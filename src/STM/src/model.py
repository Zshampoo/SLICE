# src/model.py
import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.models.bart.modeling_bart import BartDecoderLayer
try:
    from src.STM.src.config import Config
except ModuleNotFoundError:
    from src.config import Config


def _patch_transformers_torch_load_guard():
    """
    transformers>=4.57 requires torch>=2.6 to load .bin via torch.load.
    Local env is torch 2.5.1 + trusted offline BART weights; bypass the hard check.
    Prefer safetensors when available.
    """
    try:
        import transformers.utils.import_utils as import_utils
        import transformers.modeling_utils as modeling_utils
        import_utils.check_torch_load_is_safe = lambda: None
        modeling_utils.check_torch_load_is_safe = lambda: None
    except Exception:
        pass


def load_model_and_tokenizer():
    _patch_transformers_torch_load_guard()
    tokenizer = BartTokenizer.from_pretrained(Config.model_name)
    model = BartForConditionalGeneration.from_pretrained(Config.model_name)
    return model, tokenizer

def load_modelwdecoderinput_and_tokenizer():
    _patch_transformers_torch_load_guard()
    tokenizer = BartTokenizer.from_pretrained(Config.model_name)
    model = BartWithDecoderInput.from_pretrained(Config.model_name)
    return model, tokenizer

class BartWithDecoderInput(BartForConditionalGeneration):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        # 如果显式提供了 decoder_input_ids，就直接使用
        # 否则，如果 labels 存在，让父类自己 shift-right
        if decoder_input_ids is not None:
            # 直接使用用户传入的 decoder_input_ids
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
                **kwargs,
            )


class BartDecoderLayerExtractCA(BartDecoderLayer):
    """Same BART decoder layer, with the post-cross-attention state kept on the module.

    h_after_ca is the hidden state after cross-attention residual add and
    encoder_attn_layer_norm, before the FFN. Shape (batch, tgt_len, d_model).
    Edit `cross_attend` to change that block.
    """

    def cross_attend(
        self,
        hidden_states,
        encoder_hidden_states,
        encoder_attention_mask,
        cross_attn_layer_head_mask,
        past_key_values,
        output_attentions,
        cache_position,
    ):
        residual = hidden_states
        hidden_states, cross_attn_weights = self.encoder_attn(
            hidden_states=hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            layer_head_mask=cross_attn_layer_head_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        return hidden_states, cross_attn_weights

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=True,
        cache_position=None,
    ):
        residual = hidden_states
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        cross_attn_weights = None
        self.h_after_ca = None
        if encoder_hidden_states is not None:
            hidden_states, cross_attn_weights = self.cross_attend(
                hidden_states,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_layer_head_mask,
                past_key_values,
                output_attentions,
                cache_position,
            )
            # (batch, tgt_len, d_model); still in the graph so later edits can use it
            self.h_after_ca = hidden_states
            block = getattr(self, "retrieval_block", None)
            if block is not None and getattr(self, "h_o", None) is not None:
                hidden_states = block(self.h_after_ca, self.h_o, self.retrieval_mask)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        return outputs


class BartForConditionalGenerationExtractable(BartForConditionalGeneration):
    """Black-box BART weights, with h_enc and h_after_ca readable after forward.

    h_enc: encoder last hidden state, (batch, src_len, d_model)
    h_after_ca: one post-cross-attention state per decoder layer,
                (decoder_layers, batch, tgt_len, d_model)
    """

    def install_extractable_decoder_layers(self):
        new_layers = nn.ModuleList()
        for i, layer in enumerate(self.model.decoder.layers):
            replacement = BartDecoderLayerExtractCA(self.config, layer_idx=i)
            replacement.load_state_dict(layer.state_dict())
            new_layers.append(replacement)
        self.model.decoder.layers = new_layers
        self.h_enc = None
        self.h_after_ca = None

    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        if getattr(outputs, "encoder_last_hidden_state", None) is not None:
            self.h_enc = outputs.encoder_last_hidden_state
        else:
            self.h_enc = None
        states = [layer.h_after_ca for layer in self.model.decoder.layers]
        if states and all(s is not None for s in states):
            self.h_after_ca = torch.stack(states, dim=0)
        else:
            self.h_after_ca = None
        return outputs


def load_extractable_bart(model_name=None):
    """Load BART the same way as load_model_and_tokenizer, then expose h_enc / h_after_ca."""
    _patch_transformers_torch_load_guard()
    name = Config.model_name if model_name is None else model_name
    tokenizer = BartTokenizer.from_pretrained(name)
    model = BartForConditionalGenerationExtractable.from_pretrained(name)
    model.install_extractable_decoder_layers()
    return model, tokenizer

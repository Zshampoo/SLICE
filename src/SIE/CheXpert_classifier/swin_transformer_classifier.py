from pathlib import Path
from pprint import pprint

import torch
from transformers import AutoModel, SwinForImageClassification

from src.SIE.CheXpert_classifier.swin_transformer import SwinTransformer

def rename_key(hf_key):
    # Embedding 层
    if hf_key.startswith("embeddings.patch_embeddings.projection."):
        return hf_key.replace("embeddings.patch_embeddings.projection", "patch_embed.proj")
    if hf_key.startswith("embeddings.norm."):
        return hf_key.replace("embeddings.norm", "patch_embed.norm")

    # Encoder 层
    if hf_key.startswith("encoder.layers."):
        key = hf_key.replace("encoder.layers.", "layers.")
        key = key.replace(".layernorm_before.", ".norm1.")
        key = key.replace(".layernorm_after.", ".norm2.")
        key = key.replace(".attention.self.relative_position_bias_table", ".attn.relative_position_bias_table")
        key = key.replace(".attention.self.relative_position_index", ".attn.relative_position_index")
        key = key.replace(".attention.self.query.", ".attn.q.")
        key = key.replace(".attention.self.key.", ".attn.k.")
        key = key.replace(".attention.self.value.", ".attn.v.")
        key = key.replace(".attention.output.dense.", ".attn.proj.")
        key = key.replace(".intermediate.dense.", ".mlp.fc1.")
        key = key.replace(".output.dense.", ".mlp.fc2.")
        return key

    return None  # unmatched key



def build_model_swintransformer():
    # load = torch.load('./data/<path>')
    model_type = 'swin' # model_name: swin_tiny_patch4_window7_224
    if model_type == 'swin':
        model = SwinForImageClassification.from_pretrained(
            str(Path(__import__('os').environ.get('DATA_ROOT', './data'))),
            num_labels=14,  # 设置你自己的类别数
            ignore_mismatched_sizes=True
        )
        # model = SwinTransformer(img_size=224, # 224
        #                         patch_size=4,
        #                         in_chans=3,
        #                         num_classes=14,
        #                         embed_dim=128, # 96
        #                         depths=[2,2,6,2],
        #                         num_heads=[3, 6, 12, 24],
        #                         window_size=7,
        #                         mlp_ratio=4.,
        #                         qkv_bias=True,
        #                         qk_scale=None,
        #                         drop_rate=0.0,
        #                         drop_path_rate=0.1,
        #                         ape=False,
        #                         patch_norm=True,
        #                         use_checkpoint=False,
        #                         num_mlp_heads=3)
        # #### load huging face
        # # Step 1: 加载 Hugging Face 的 Swin 模型
        # hf_model = AutoModel.from_pretrained("./data/<path>")
        # # Step 2: 获取 state_dict
        # hf_state_dict = hf_model.state_dict()
        # # Step3. 重命名 Hugging Face 权重并筛选匹配项
        # renamed_state_dict = {}
        # for k, v in hf_state_dict.items():
        #     new_k = rename_key(k)
        #     if new_k and new_k in model.state_dict() and v.shape == model.state_dict()[new_k].shape:
        #         renamed_state_dict[new_k] = v
        #     else:
        #         print(f"[Unmatched or shape mismatch] {k} -> {new_k}")
        # # Step 4: 过滤掉分类头
        # filtered_state_dict = {
        #     k: v for k, v in renamed_state_dict.items()
        #     if k in model.state_dict() and v.shape == model.state_dict()[k].shape
        # }
        # print("=== Hugging Face keys ===")
        # for k in list(renamed_state_dict.keys())[:10]:
        #     print(k)
        #
        # print("\n=== Your model keys ===")
        # for k in list(model.state_dict().keys())[:10]:
        #     print(k)
        #
        # # Step 5: 加载权重（strict=False 以容忍不匹配层）
        # missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
        #
        # print("Loaded successfully with:")
        # print(f"  - missing keys: {missing}")
        # print(f"  - unexpected keys: {unexpected}")
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model


# if __name__ == '__main__':
#     model = build_model()
#     pprint(model)
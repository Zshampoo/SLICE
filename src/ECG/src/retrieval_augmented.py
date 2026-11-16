"""Retrieval gating and retrieval-augmented cross-attention on extractable BART.

Control vector per disease is N_ctrl=128. It is concatenated with
h_llm_tokavg (1024) and mapped by Linear(N_ctrl + 1024, 512).
"""
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from transformers.models.bart.modeling_bart import BartAttention

DISEASES = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]
N_CTRL = 128
D_MODEL = 1024
RETRIEVAL_DIM = 512
JSON_PATH = (
    Path(__import__("os").environ.get("DATA_ROOT", "./data"))
    / "Phase2_retrieval_phrase_encoded_by_biocliBERT_only_not_conflict"
    / "phase2_retrieval_phrase_only_not_conflict_encoded_by_biocliBERT_tokenavg.json"
)


class PositionalEncoding(nn.Module):
    """Sinusoidal encoding, same construction as the standard Transformer PE."""

    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class DiseaseRetrievalGating(nn.Module):
    """Map K phrase vectors and build one 512-d vector per disease."""

    def __init__(self, n_ctrl=N_CTRL, dropout=0.1):
        super().__init__()
        self.n_ctrl = n_ctrl
        self.control_pos = nn.Parameter(torch.randn(len(DISEASES), n_ctrl) * 0.02)
        self.control_neg = nn.Parameter(torch.randn(len(DISEASES), n_ctrl) * 0.02)
        self.h_bla = nn.Parameter(torch.randn(1, RETRIEVAL_DIM) * 0.02)
        self.s_proj = nn.Linear(768, RETRIEVAL_DIM)
        self.q_proj = nn.Linear(n_ctrl + D_MODEL, RETRIEVAL_DIM)
        self.pos = PositionalEncoding(RETRIEVAL_DIM, dropout)

    def forward(self, h_llm, h_llm_mask, polarity, s_pad, s_valid):
        """
        h_llm: (B, src_len, 1024)
        h_llm_mask: (B, src_len) 1 = token
        polarity: (B, 14)  1 POS, 0 NEG, -1 BLA
        s_pad: (B, 14, K, 768)
        s_valid: (B, 14, K) bool
        returns h_o (B, 14, 512), disease_mask (B, 14) with 1 where S_k exists
        """
        mask = h_llm_mask.to(h_llm.dtype).unsqueeze(-1)
        h_llm_tokavg = (h_llm * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        disease_mask = s_valid.any(dim=-1).to(h_llm.dtype)
        rows = []
        weights = []
        for i in range(len(DISEASES)):
            use_pos = polarity[:, i] == 1
            has_s = disease_mask[:, i] > 0
            gate = ((polarity[:, i] == 1) | (polarity[:, i] == 0)) & has_s
            ctrl = torch.where(use_pos.unsqueeze(-1), self.control_pos[i], self.control_neg[i])
            q = torch.cat([ctrl, h_llm_tokavg], dim=-1)
            q_mm = self.q_proj(q)
            s_mm = self.s_proj(s_pad[:, i])
            logits = torch.matmul(q_mm.unsqueeze(1), s_mm.transpose(1, 2)).squeeze(1)
            logits = logits / math.sqrt(RETRIEVAL_DIM)
            logits = logits.masked_fill(~s_valid[:, i], torch.finfo(logits.dtype).min)
            w = torch.softmax(logits, dim=-1)
            w = torch.where(has_s.unsqueeze(-1), w, torch.zeros_like(w))
            h = torch.matmul(w.unsqueeze(1), s_mm).squeeze(1)
            h = torch.where(gate.unsqueeze(-1), h, self.h_bla.expand_as(h))
            rows.append(h)
            weights.append(w)
        h = torch.stack(rows, dim=1)
        h_o = self.pos(h)
        return h_o, disease_mask, h_llm_tokavg, weights


class RetrievalAugmentedCrossAttention(nn.Module):
    """Same residual cross-attention block as BartDecoderLayer.cross_attend.

    Query is h_after_ca (B, tgt_len, 1024). Memory is h_o (B, 14, 512),
    projected to 1024 so the attention width matches BART.
    """

    def __init__(self, config):
        super().__init__()
        self.kv_proj = nn.Linear(RETRIEVAL_DIM, config.d_model)
        self.attn = BartAttention(
            embed_dim=config.d_model,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        self.dropout = config.dropout
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, h_after_ca, h_o, disease_mask):
        if h_o.size(0) != h_after_ca.size(0):
            reps = h_after_ca.size(0) // h_o.size(0)
            h_o = h_o.repeat_interleave(reps, dim=0)
            disease_mask = disease_mask.repeat_interleave(reps, dim=0)
        kv = self.kv_proj(h_o)
        all_blocked = disease_mask.reshape(disease_mask.size(0), -1).sum(dim=-1) == 0
        if bool(all_blocked.all()):
            return h_after_ca
        additive = torch.zeros(
            disease_mask.size(0),
            1,
            1,
            disease_mask.size(1),
            device=h_after_ca.device,
            dtype=h_after_ca.dtype,
        )
        additive = additive.masked_fill(disease_mask[:, None, None, :] == 0, torch.finfo(h_after_ca.dtype).min)
        if bool(all_blocked.any()):
            additive = additive.clone()
            additive[all_blocked, :, :, 0] = 0
        attn_out, _ = self.attn(
            hidden_states=h_after_ca,
            key_value_states=kv,
            attention_mask=additive,
            output_attentions=False,
        )
        attn_out = nn.functional.dropout(attn_out, p=self.dropout, training=self.training)
        out = self.layer_norm(h_after_ca + attn_out)
        if bool(all_blocked.any()):
            out = torch.where(all_blocked.view(-1, 1, 1), h_after_ca, out)
        return out


def attach_retrieval(model, block):
    for layer in model.model.decoder.layers:
        layer.retrieval_block = block


def bind_retrieval_memory(model, h_o, disease_mask):
    for layer in model.model.decoder.layers:
        layer.h_o = h_o
        layer.retrieval_mask = disease_mask


def polarity_from_y(value):
    if value is None:
        return -1
    try:
        number = float(value)
    except (TypeError, ValueError):
        return -1
    if math.isnan(number):
        return -1
    if number == 0.0:
        return 0
    return 1


def disease_mask_from_record(record):
    mask = []
    for name in DISEASES:
        entry = record.get(name)
        ok = isinstance(entry, list) and len(entry) >= 4 and isinstance(entry[3], list) and len(entry[3]) > 0
        mask.append(1 if ok else 0)
    return mask


def vectors_from_record(record):
    """Return list of 14 float tensors (K, 768) or None."""
    out = []
    for name in DISEASES:
        entry = record.get(name)
        if not (isinstance(entry, list) and len(entry) >= 4 and entry[3]):
            out.append(None)
            continue
        rows = [item[-1] for item in entry[3] if isinstance(item[-1], list)]
        out.append(torch.tensor(rows, dtype=torch.float32) if rows else None)
    return out


def pad_phrase_batch(per_sample):
    """per_sample: list over batch of list-14 tensors or None -> s_pad, s_valid."""
    batch = len(per_sample)
    k_max = 1
    for sample in per_sample:
        for item in sample:
            if item is not None:
                k_max = max(k_max, item.size(0))
    s_pad = torch.zeros(batch, len(DISEASES), k_max, 768)
    s_valid = torch.zeros(batch, len(DISEASES), k_max, dtype=torch.bool)
    for b, sample in enumerate(per_sample):
        for i, item in enumerate(sample):
            if item is None:
                continue
            k = item.size(0)
            s_pad[b, i, :k] = item
            s_valid[b, i, :k] = True
    return s_pad, s_valid


def write_disease_masks(json_path=JSON_PATH):
    """Insert disease_attn_mask into each dicom object without reserializing vectors."""
    json_path = Path(json_path)
    tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    n = 0
    with json_path.open("r", encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        first = src.readline()
        if first.strip() != "{":
            raise RuntimeError(f"unexpected JSON start: {first[:20]!r}")
        dst.write("{\n")
        for line in src:
            stripped = line.strip()
            if stripped in ("", "}"):
                continue
            brace = line.find(":{")
            if brace < 0:
                raise RuntimeError("dicom line has no object")
            obj = line[brace + 1 :]
            if '"disease_attn_mask"' not in obj[:80]:
                mask = [1 if f'"{name}":[' in obj else 0 for name in DISEASES]
                obj = '{"disease_attn_mask":' + json.dumps(mask) + "," + obj[1:]
            dst.write(line[: brace + 1])
            dst.write(obj if obj.endswith("\n") else obj + "\n")
            n += 1
            if n % 2000 == 0:
                print(f"mask written {n}", flush=True)
        dst.write("}\n")
    tmp.replace(json_path)
    print(f"mask write done n={n} -> {json_path}", flush=True)
    return n

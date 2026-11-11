"""CheXbert loader used by DST prompt construction."""
from __future__ import annotations
import os
from pathlib import Path
import torch
from transformers import BertTokenizer
from src.CheXbert.src.models.bert_labeler import bert_labeler


def get_chexbert(checkpoint: str | None = None, device: str | None = None):
    device = device or os.environ.get("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = checkpoint or os.environ.get(
        "CHEXBERT_WEIGHTS",
        str(Path(os.environ.get("DATA_ROOT", "./data")) / "weights" / "chexbert.pth"),
    )
    model = bert_labeler()
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer

"""Clinical Efficacy (CheXbert) P / R / F1 — F1-all 14 and F1-conflict 5."""

from __future__ import annotations

import csv
import os
import tempfile
from collections import OrderedDict
from typing import Dict, Sequence

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

CHEXBERT_WEIGHTS = os.environ.get(
    "CHEXBERT_PATH",
    os.environ.get(
        "CHEXBERT_WEIGHTS",
        os.path.join(os.environ.get("DATA_ROOT", "./data"), "weights", "chexbert.pth"),
    ),
)

# Paper table: F1-conflict 5
CONDITIONS_CONFLICT5 = [
    "Cardiomegaly",
    "Atelectasis",
    "Edema",
    "Pleural Effusion",
    "Lung Opacity",
]
# Miura 5 (legacy)
CONDITIONS_5 = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]
CONDITIONS_C6 = [
    "Cardiomegaly",
    "Enlarged Cardiomediastinum",
    "Atelectasis",
    "Pleural Effusion",
    "Edema",
    "Lung Opacity",
]


def _load_chexbert(chexbert_path: str | None = None):
    from src.EVAL.factual_accuracy.CheXbert.src.models.bert_labeler import bert_labeler

    weights = chexbert_path or CHEXBERT_WEIGHTS
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = bert_labeler()
    ckpt = torch.load(weights, map_location="cpu", weights_only=True)
    state = OrderedDict()
    for k, v in ckpt["model_state_dict"].items():
        state[k[7:] if k.startswith("module.") else k] = v
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    try:
        import src.EVAL.factual_accuracy.CheXbert.src.label as label_mod

        label_mod.device = device
    except Exception:
        pass
    return model, device


def _label_reports(model, reports: Sequence[str]):
    from src.EVAL.factual_accuracy.CheXbert.src.label import label

    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, "reports.csv")
        with open(csv_path, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["Report Impression"])
            for r in reports:
                w.writerow([str(r) if r is not None else ""])
        return label(model, csv_path)  # List[14][N]


def _miura_binarize(preds_reports: list) -> np.ndarray:
    """preds: List[14 lists of N] -> (N, 14) with Miura mapping 2->0, 3->1."""
    n = len(preds_reports[0])
    mat = np.zeros((n, 14), dtype=np.int64)
    for c in range(14):
        for i, lab in enumerate(preds_reports[c]):
            if lab == 2:
                lab = 0
            elif lab == 3:
                lab = 1
            mat[i, c] = int(lab)
    return mat


def _micro_prf(gt: np.ndarray, pred: np.ndarray, idxs: list[int]) -> tuple[float, float, float]:
    gt_bin = (gt[:, idxs] == 1).astype(np.int64).reshape(-1)
    pred_bin = (pred[:, idxs] == 1).astype(np.int64).reshape(-1)
    p, r, f1, _ = precision_recall_fscore_support(
        gt_bin, pred_bin, average="binary", zero_division=0
    )
    return float(p), float(r), float(f1)


def _macro_f1(gt: np.ndarray, pred: np.ndarray, idxs: list[int]) -> float:
    scores = []
    for j in idxs:
        gt_j = (gt[:, j] == 1).astype(np.int64)
        pred_j = (pred[:, j] == 1).astype(np.int64)
        _, _, f, _ = precision_recall_fscore_support(
            gt_j, pred_j, average="binary", zero_division=0
        )
        scores.append(float(f))
    return float(np.mean(scores)) if scores else 0.0


def compute_ce(
    refs: Sequence[str],
    hyps: Sequence[str],
    chexbert_path: str | None = None,
) -> Dict[str, float]:
    """Return F1-all 14 and F1-conflict 5 (micro + macro), plus legacy c5/c6 keys."""
    from src.EVAL.factual_accuracy.CheXbert.src.constants import CONDITIONS

    model, _ = _load_chexbert(chexbert_path)
    pred = _miura_binarize(_label_reports(model, hyps))
    gt = _miura_binarize(_label_reports(model, refs))

    idx_all = list(range(14))
    p, r, f1 = _micro_prf(gt, pred, idx_all)
    out: Dict[str, float] = {
        "CE-P": p,
        "CE-R": r,
        "CE-F1": f1,
        "F1-all_14_micro": f1,
        "F1-all_14_macro": _macro_f1(gt, pred, idx_all),
    }

    idx_c5 = [CONDITIONS.index(c) for c in CONDITIONS_CONFLICT5 if c in CONDITIONS]
    pc5, rc5, f1c5 = _micro_prf(gt, pred, idx_c5)
    out.update(
        {
            "F1-conflict_5_micro": f1c5,
            "F1-conflict_5_macro": _macro_f1(gt, pred, idx_c5),
            "CE-P_conflict5": pc5,
            "CE-R_conflict5": rc5,
            "CE-F1_conflict5": f1c5,
        }
    )

    idx5 = [CONDITIONS.index(c) for c in CONDITIONS_5 if c in CONDITIONS]
    p5, r5, f15 = _micro_prf(gt, pred, idx5)
    out.update({"CE-P5": p5, "CE-R5": r5, "CE-F15": f15})

    idx6 = [CONDITIONS.index(c) for c in CONDITIONS_C6 if c in CONDITIONS]
    p6, r6, f16 = _micro_prf(gt, pred, idx6)
    out.update({"CE-Pc6": p6, "CE-Rc6": r6, "CE-F1c6": f16})

    alias = {
        "Enlarged Cardiomediastinum": "ECM",
        "Pleural Effusion": "Pleural_Effusion",
        "Lung Opacity": "Lung_Opacity",
    }
    for name in CONDITIONS_C6:
        if name not in CONDITIONS:
            continue
        j = CONDITIONS.index(name)
        gt_j = (gt[:, j] == 1).astype(np.int64)
        pred_j = (pred[:, j] == 1).astype(np.int64)
        _p, _r, _f, _ = precision_recall_fscore_support(
            gt_j, pred_j, average="binary", zero_division=0
        )
        key = name.replace(" ", "_")
        out[f"CE/{key}/precision"] = float(_p)
        out[f"CE/{key}/recall"] = float(_r)
        out[f"CE/{key}/f1"] = float(_f)
        short = alias.get(name, name)
        out[f"CE-{short}-F1"] = float(_f)
    return out


def chexbert_label_vectors(
    reports: Sequence[str],
    chexbert_path: str | None = None,
) -> torch.Tensor:
    model, _ = _load_chexbert(chexbert_path)
    mat = _miura_binarize(_label_reports(model, reports))
    return torch.tensor(mat, dtype=torch.float32)


def semb_cosine(refs: Sequence[str], hyps: Sequence[str], chexbert_path: str | None = None):
    ref_v = chexbert_label_vectors(refs, chexbert_path)
    hyp_v = chexbert_label_vectors(hyps, chexbert_path)
    eps = 1e-8
    num = (ref_v * hyp_v).sum(dim=1)
    den = ref_v.norm(dim=1) * hyp_v.norm(dim=1) + eps
    return (num / den).cpu().numpy()

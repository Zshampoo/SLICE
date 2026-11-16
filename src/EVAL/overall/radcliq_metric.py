"""RadCliQ-v1 and 1/RadCliQ (higher better)."""

from __future__ import annotations

import os
import pickle
import sys
import types
from typing import Dict, Sequence

import numpy as np

CXR_METRIC_ROOT = os.environ.get(
    "CXR_METRIC_ROOT",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "third_party",
        "CXR-Report-Metric",
    ),
)
RADCLIQ_PKL = os.path.join(CXR_METRIC_ROOT, "CXRMetric", "radcliq-v1.pkl")


class CompositeMetric:
    """Stub matching CXR-Report-Metric for unpickling RadCliQ-v1."""

    def __init__(self, scaler, coefs):
        self.scaler = scaler
        self.coefs = coefs

    def predict(self, x):
        norm_x = self.scaler.transform(x)
        norm_x = np.concatenate((norm_x, np.ones((norm_x.shape[0], 1))), axis=1)
        return norm_x @ self.coefs


def _load_radcliq_model(pkl_path: str = RADCLIQ_PKL):
    # Ensure pickle can resolve CompositeMetric
    mod = types.ModuleType("CXRMetric.run_eval")
    mod.CompositeMetric = CompositeMetric
    sys.modules.setdefault("CXRMetric", types.ModuleType("CXRMetric"))
    sys.modules["CXRMetric.run_eval"] = mod
    # Some pickles were dumped from __main__
    import __main__

    __main__.CompositeMetric = CompositeMetric
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def compute_radcliq(
    refs: Sequence[str],
    hyps: Sequence[str],
    *,
    bleu2_per_sample: np.ndarray | None = None,
    bertscore_per_sample: np.ndarray | None = None,
    radgraph_per_sample: np.ndarray | None = None,
    semb_per_sample: np.ndarray | None = None,
    cuda: int = 0,
    chexbert_path: str | None = None,
) -> Dict[str, float]:
    """
    Columns expected by RadCliQ-v1 (Yu et al.):
      [radgraph_combined, bertscore, semb_score, bleu_score]
    Returns mean RadCliQ-v1 (lower better) and 1/mean (higher better).
    """
    from src.EVAL.factual_accuracy.ce_metrics import semb_cosine
    from src.EVAL.textual_similarity.nlg import compute_bertscore, compute_bleu2_per_sample
    from src.EVAL.textual_similarity.radgraph_metric import compute_radgraph

    n = len(refs)
    if bleu2_per_sample is None:
        bleu2_per_sample = compute_bleu2_per_sample(refs, hyps)
    if bertscore_per_sample is None:
        bs = compute_bertscore(
            refs,
            hyps,
            model_type=os.environ.get(
                "RADCLIQ_BERTSCORE_MODEL",
                "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            ),
            num_layers=9,
            rescale_with_baseline=False,
            device=f"cuda:{cuda}" if cuda is not None and cuda >= 0 else "cpu",
        )
        bertscore_per_sample = bs["BertScore_per_sample"]
    if semb_per_sample is None:
        semb_per_sample = semb_cosine(refs, hyps, chexbert_path)
    if radgraph_per_sample is None:
        rg = compute_radgraph(refs, hyps, cuda=cuda)
        if "RadGraph_combined_per_sample" in rg:
            radgraph_per_sample = rg["RadGraph_combined_per_sample"]
        else:
            radgraph_per_sample = np.full(n, rg["RadGraph-Complete"], dtype=float)

    def _fit(a, name):
        arr = np.asarray(a, dtype=float).reshape(-1)
        if arr.shape[0] != n:
            if arr.shape[0] == 0:
                return np.zeros(n, dtype=float)
            # pad / truncate
            out = np.zeros(n, dtype=float)
            m = min(n, arr.shape[0])
            out[:m] = arr[:m]
            return out
        return arr

    X = np.stack(
        [
            _fit(radgraph_per_sample, "radgraph"),
            _fit(bertscore_per_sample, "bertscore"),
            _fit(semb_per_sample, "semb"),
            _fit(bleu2_per_sample, "bleu2"),
        ],
        axis=1,
    )

    model = _load_radcliq_model()
    raw = np.asarray(model.predict(X), dtype=float)
    mean_raw = float(np.mean(raw))
    inv = float(1.0 / mean_raw) if abs(mean_raw) > 1e-12 else float("nan")
    return {
        "RadCliQ-v1": mean_raw,
        "1/RadCliQ": inv,
        "RadCliQ_per_sample": raw,
    }

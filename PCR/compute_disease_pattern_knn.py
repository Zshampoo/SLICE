"""
Per-row disease pattern KNN on train_file_0 / train_file_1 cluster members.

Adds pred_pattern_knn_norm_101 and pred_pattern_knn_bal_100 to disease_pattern_similarity.
"""

from __future__ import annotations

from paths import DATA_ROOT, KNN_ROOT

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd

from compute_cardiomegaly_pattern_similarity import (
    GT_COL,
    GPT_PRED_COL,
    metrics_binary,
    norm_gt,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

META_KEYS = {"average", "geometric_median", "gaussian_params", "gaussian_param"}
K_NORM = 101
K_BAL = 100
HALF_BAL = K_BAL // 2

TRAIN_DIRS = [
    KNN_ROOT / "train_clusters_0",
    KNN_ROOT / "train_clusters_1",
]


def resolve_train_path(fname: str) -> Optional[Path]:
    for base in TRAIN_DIRS:
        p = base / fname
        if p.is_file():
            return p
    return None


def load_train_members(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    feats: List[np.ndarray] = []
    for k, v in obj.items():
        if k in META_KEYS:
            continue
        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim == 1 and arr.shape[0] == 2048:
            feats.append(arr)
    if not feats:
        return np.zeros((0, 2048), dtype=np.float32)
    return np.stack(feats, axis=0)


class TrainMemberCache:
    def __init__(self) -> None:
        self._cache: Dict[str, np.ndarray] = {}

    def get(self, fname: str) -> Optional[np.ndarray]:
        if fname in self._cache:
            return self._cache[fname]
        path = resolve_train_path(fname)
        if path is None:
            return None
        mat = load_train_members(path)
        self._cache[fname] = mat
        return mat


class PairIndexCache:
    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def get(self, f0: str, f1: str, member_cache: TrainMemberCache) -> Optional[Dict[str, Any]]:
        key = (f0, f1)
        if key in self._cache:
            return self._cache[key]
        m0 = member_cache.get(f0)
        m1 = member_cache.get(f1)
        if m0 is None or m1 is None:
            return None
        n0, n1 = len(m0), len(m1)
        if n0 == 0 and n1 == 0:
            return None
        combined = np.vstack([m0, m1]) if n0 and n1 else (m0 if n0 else m1)
        labels = ["negative"] * n0 + ["positive"] * n1
        index_all = _build_faiss_index(combined)
        index_neg = _build_faiss_index(m0) if n0 else None
        index_pos = _build_faiss_index(m1) if n1 else None
        obj = {
            "n0": n0,
            "n1": n1,
            "index_all": index_all,
            "index_neg": index_neg,
            "index_pos": index_pos,
            "labels": labels,
        }
        self._cache[key] = obj
        return obj


def _build_faiss_index(mat: np.ndarray) -> faiss.IndexFlatIP:
    x = mat.astype(np.float32).copy()
    faiss.normalize_L2(x)
    index = faiss.IndexFlatIP(2048)
    index.add(x)
    return index


def _softmax_pair(score_pos: float, score_neg: float) -> Tuple[float, str]:
    ep, en = np.exp(score_pos), np.exp(score_neg)
    p_pos = float(ep / (ep + en)) if (ep + en) > 0 else 0.5
    pred = "positive" if p_pos > 0.5 else "negative"
    return p_pos, pred


def knn_pattern_norm(
    feat: np.ndarray,
    index: faiss.IndexFlatIP,
    labels: List[str],
    n0: int,
    n1: int,
    k: int = K_NORM,
) -> Dict[str, Any]:
    """
    Imbalance-normalized top-K over combined pool.

    Insufficient samples:
      - k_eff = min(k, n0+n1); retrieve fewer neighbors when total < k.
      - Scores divide by n0/n1 (full pool sizes), same as region knn_norm.
    """
    out: Dict[str, Any] = {
        f"pattern_knn_norm_k": k,
        f"pattern_knn_norm_k_eff_{k}": 0,
        f"pattern_knn_norm_score_negative_{k}": np.nan,
        f"pattern_knn_norm_score_positive_{k}": np.nan,
        f"pattern_knn_norm_P_positive_{k}": np.nan,
        f"pred_pattern_knn_norm_{k}": np.nan,
        f"pattern_knn_norm_topk_negative_count_{k}": 0,
        f"pattern_knn_norm_topk_positive_count_{k}": 0,
        f"pattern_knn_norm_insufficient_total": False,
    }
    total = n0 + n1
    if total == 0 or index.ntotal == 0:
        return out
    k_eff = min(k, total)
    out[f"pattern_knn_norm_k_eff_{k}"] = k_eff
    out["pattern_knn_norm_insufficient_total"] = total < k

    q = feat.astype(np.float32).reshape(1, -1).copy()
    faiss.normalize_L2(q)
    D, I = index.search(q, k_eff)

    score_neg = 0.0
    score_pos = 0.0
    hit_neg = 0
    hit_pos = 0
    for sim, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        if labels[idx] == "negative":
            score_neg += float(sim)
            hit_neg += 1
        else:
            score_pos += float(sim)
            hit_pos += 1
    score_neg /= max(n0, 1)
    score_pos /= max(n1, 1)
    p_pos, pred = _softmax_pair(score_pos, score_neg)
    out.update(
        {
            f"pattern_knn_norm_score_negative_{k}": score_neg,
            f"pattern_knn_norm_score_positive_{k}": score_pos,
            f"pattern_knn_norm_P_positive_{k}": p_pos,
            f"pred_pattern_knn_norm_{k}": pred,
            f"pattern_knn_norm_topk_negative_count_{k}": hit_neg,
            f"pattern_knn_norm_topk_positive_count_{k}": hit_pos,
        }
    )
    return out


def knn_pattern_balanced(
    feat: np.ndarray,
    index_neg: Optional[faiss.IndexFlatIP],
    index_pos: Optional[faiss.IndexFlatIP],
    n0: int,
    n1: int,
    k: int = K_BAL,
) -> Dict[str, Any]:
    """
    Strict balanced KNN (scheme B): both sides share the same k_eff.

    k_eff = min(k/2, n0, n1); each side takes top-k_eff neighbors and
    score = mean similarity (sum / k_eff).
    """
    half = k // 2
    out: Dict[str, Any] = {
        f"pattern_knn_bal_k": k,
        f"pattern_knn_bal_topk_per_group_{k}": half,
        f"pattern_knn_bal_k_eff_{k}": 0,
        f"pattern_knn_bal_k_neg_eff_{k}": 0,
        f"pattern_knn_bal_k_pos_eff_{k}": 0,
        f"pattern_knn_bal_score_negative_{k}": 0.0,
        f"pattern_knn_bal_score_positive_{k}": 0.0,
        f"pattern_knn_bal_P_positive_{k}": np.nan,
        f"pred_pattern_knn_bal_{k}": np.nan,
        f"pattern_knn_bal_insufficient_neg": n0 < half,
        f"pattern_knn_bal_insufficient_pos": n1 < half,
        f"pattern_knn_bal_k_eff_lt_half": False,
    }
    if n0 == 0 and n1 == 0:
        return out

    q = feat.astype(np.float32).reshape(1, -1).copy()
    faiss.normalize_L2(q)

    if n0 == 0 or n1 == 0:
        out[f"pred_pattern_knn_bal_{k}"] = "positive" if n1 > n0 else "negative"
        out[f"pattern_knn_bal_P_positive_{k}"] = 1.0 if n1 > n0 else 0.0
        return out

    k_eff = min(half, n0, n1)
    out[f"pattern_knn_bal_k_eff_{k}"] = k_eff
    out[f"pattern_knn_bal_k_neg_eff_{k}"] = k_eff
    out[f"pattern_knn_bal_k_pos_eff_{k}"] = k_eff
    out["pattern_knn_bal_k_eff_lt_half"] = k_eff < half

    score_neg = 0.0
    score_pos = 0.0
    if index_neg is not None:
        Dn, _ = index_neg.search(q, k_eff)
        score_neg = float(Dn[0].sum()) / k_eff
    if index_pos is not None:
        Dp, _ = index_pos.search(q, k_eff)
        score_pos = float(Dp[0].sum()) / k_eff

    p_pos, pred = _softmax_pair(score_pos, score_neg)
    out.update(
        {
            f"pattern_knn_bal_score_negative_{k}": score_neg,
            f"pattern_knn_bal_score_positive_{k}": score_pos,
            f"pattern_knn_bal_P_positive_{k}": p_pos,
            f"pred_pattern_knn_bal_{k}": pred,
        }
    )
    return out


def compute_row_knn(
    feat: np.ndarray,
    f0: Any,
    f1: Any,
    member_cache: TrainMemberCache,
    pair_cache: PairIndexCache,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "pattern_knn_error": np.nan,
        "pattern_knn_skip_reason": np.nan,
    }
    if feat is None or (isinstance(feat, float) and math.isnan(feat)):
        out["pattern_knn_skip_reason"] = "missing_test_feature"
        return out
    if f0 is None or f1 is None or (isinstance(f0, float) and np.isnan(f0)) or (isinstance(f1, float) and np.isnan(f1)):
        out["pattern_knn_skip_reason"] = "missing_train_file"
        return out
    f0s, f1s = str(f0), str(f1)
    pair = pair_cache.get(f0s, f1s, member_cache)
    if pair is None:
        if resolve_train_path(f0s) is None or resolve_train_path(f1s) is None:
            out["pattern_knn_skip_reason"] = "train_file_not_found"
        else:
            out["pattern_knn_skip_reason"] = "empty_train_pool"
        return out
    try:
        out.update(
            knn_pattern_norm(
                feat,
                pair["index_all"],
                pair["labels"],
                pair["n0"],
                pair["n1"],
            )
        )
        out.update(
            knn_pattern_balanced(
                feat,
                pair["index_neg"],
                pair["index_pos"],
                pair["n0"],
                pair["n1"],
            )
        )
        out["pattern_knn_n0"] = pair["n0"]
        out["pattern_knn_n1"] = pair["n1"]
    except Exception as e:
        out["pattern_knn_error"] = str(e)
    return out


def attach_gpt_pred(df: pd.DataFrame, gpt_df: pd.DataFrame, gpt_col: str) -> pd.DataFrame:
    if "id" not in df.columns:
        return df
    out = df.drop(columns=[gpt_col], errors="ignore").copy()
    out["id"] = out["id"].astype(str)
    merged = out.merge(gpt_df, on="id", how="left", validate="many_to_one")
    cols = [c for c in merged.columns if c != gpt_col]
    if gpt_col in merged.columns:
        insert_at = cols.index("id") + 1 if "id" in cols else 0
        cols = cols[:insert_at] + [gpt_col] + cols[insert_at:]
    return merged[cols]


def build_knn_summary(df: pd.DataFrame, gt_col: str) -> pd.DataFrame:
    y_true = df[gt_col].map(norm_gt)
    rows = []
    for col in (f"pred_pattern_knn_norm_{K_NORM}", f"pred_pattern_knn_bal_{K_BAL}"):
        m = metrics_binary(y_true, df[col])
        rows.append({"method": col.replace("pred_pattern_", ""), "pred_col": col, **m})
    return pd.DataFrame(rows)


def build_insufficient_stats(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["pattern_knn_n0"].notna() & df["pattern_knn_n1"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()
    n0 = valid["pattern_knn_n0"].astype(int)
    n1 = valid["pattern_knn_n1"].astype(int)
    total = n0 + n1
    rows = [
        {"metric": "valid_knn_rows", "count": len(valid)},
        {"metric": "sum_lt_101", "count": int((total < 101).sum())},
        {"metric": "n0_lt_50", "count": int((n0 < 50).sum())},
        {"metric": "n1_lt_50", "count": int((n1 < 50).sum())},
        {"metric": "either_lt_50", "count": int(((n0 < 50) | (n1 < 50)).sum())},
        {"metric": "k_eff_lt_50", "count": int((valid[f"pattern_knn_bal_k_eff_lt_half"] == True).sum()) if f"pattern_knn_bal_k_eff_lt_half" in valid.columns else 0},
        {"metric": "n0_min", "count": int(n0.min())},
        {"metric": "n1_min", "count": int(n1.min())},
        {"metric": "total_min", "count": int(total.min())},
        {"metric": "skip_missing_train_file", "count": int((df["pattern_knn_skip_reason"] == "missing_train_file").sum())},
        {"metric": "skip_train_not_found", "count": int((df["pattern_knn_skip_reason"] == "train_file_not_found").sum())},
        {"metric": "skip_empty_pool", "count": int((df["pattern_knn_skip_reason"] == "empty_train_pool").sum())},
        {"metric": "skip_missing_feat", "count": int((df["pattern_knn_skip_reason"] == "missing_test_feature").sum())},
    ]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conflict-xlsx",
        default=str(KNN_ROOT),
    )
    parser.add_argument(
        "--test-features-json",
        default=str(KNN_ROOT),
    )
    parser.add_argument(
        "--wacc-xlsx",
        default=str(KNN_ROOT),
    )
    parser.add_argument("--gt-col", default=GT_COL)
    parser.add_argument("--gpt-pred-col", default=GPT_PRED_COL)
    args = parser.parse_args()

    conflict_path = Path(args.conflict_xlsx)
    df = pd.read_excel(conflict_path, sheet_name="disease_pattern_similarity", engine="openpyxl")
    df["id"] = df["id"].astype(str)

    with open(args.test_features_json, "r", encoding="utf-8") as f:
        raw_test = json.load(f)
    feat_map = {str(k): np.asarray(v, dtype=np.float32) for k, v in raw_test.items() if isinstance(v, list)}

    member_cache = TrainMemberCache()
    pair_cache = PairIndexCache()

    knn_cols = [
        f"pred_pattern_knn_norm_{K_NORM}",
        f"pred_pattern_knn_bal_{K_BAL}",
        f"pattern_knn_norm_score_negative_{K_NORM}",
        f"pattern_knn_norm_score_positive_{K_NORM}",
        f"pattern_knn_norm_P_positive_{K_NORM}",
        f"pattern_knn_norm_topk_negative_count_{K_NORM}",
        f"pattern_knn_norm_topk_positive_count_{K_NORM}",
        f"pattern_knn_norm_k_eff_{K_NORM}",
        f"pattern_knn_norm_insufficient_total",
        f"pattern_knn_bal_score_negative_{K_BAL}",
        f"pattern_knn_bal_score_positive_{K_BAL}",
        f"pattern_knn_bal_P_positive_{K_BAL}",
        f"pattern_knn_bal_k_eff_{K_BAL}",
        f"pattern_knn_bal_k_neg_eff_{K_BAL}",
        f"pattern_knn_bal_k_pos_eff_{K_BAL}",
        f"pattern_knn_bal_k_eff_lt_half",
        f"pattern_knn_bal_insufficient_neg",
        f"pattern_knn_bal_insufficient_pos",
        f"pattern_knn_n0",
        f"pattern_knn_n1",
        f"pattern_knn_skip_reason",
        f"pattern_knn_error",
    ]
    for c in knn_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    extra_rows: List[Dict[str, Any]] = []
    for i, row in df.iterrows():
        sid = str(row["id"])
        feat = feat_map.get(sid)
        extra = compute_row_knn(
            feat,
            row.get("train_file_0"),
            row.get("train_file_1"),
            member_cache,
            pair_cache,
        )
        extra_rows.append(extra)
        if (len(extra_rows) % 500) == 0:
            log.info("knn progress %d/%d, cached_files=%d, cached_pairs=%d", len(extra_rows), len(df), len(member_cache._cache), len(pair_cache._cache))

    extra_df = pd.DataFrame(extra_rows)
    result_df = pd.concat([df.reset_index(drop=True), extra_df], axis=1)

    summary_df = build_knn_summary(result_df, args.gt_col)
    insuf_df = build_insufficient_stats(result_df)
    log.info("KNN summary:\n%s", summary_df.to_string(index=False))
    log.info("Insufficient stats:\n%s", insuf_df.to_string(index=False))
    log.info("cached train files=%d, cached pairs=%d", len(member_cache._cache), len(pair_cache._cache))

    wacc = pd.read_excel(
        args.wacc_xlsx,
        sheet_name="Sheet1",
        usecols=["id", args.gpt_pred_col],
        engine="openpyxl",
    )
    wacc["id"] = wacc["id"].astype(str)
    gpt_df = wacc.drop_duplicates("id")

    with pd.ExcelFile(conflict_path, engine="openpyxl") as xl:
        existing = {name: pd.read_excel(xl, sheet_name=name) for name in xl.sheet_names}

    result_df = attach_gpt_pred(result_df, gpt_df, args.gpt_pred_col)

    with pd.ExcelWriter(conflict_path, engine="openpyxl") as writer:
        for name, sheet_df in existing.items():
            if name in (
                "disease_pattern_similarity",
                "disease_pattern_similarity_summary",
                "disease_pattern_knn_summary",
                "disease_pattern_knn_insufficient_stats",
            ):
                continue
            attach_gpt_pred(sheet_df, gpt_df, args.gpt_pred_col).to_excel(
                writer, sheet_name=name, index=False
            )
        result_df.to_excel(writer, sheet_name="disease_pattern_similarity", index=False)
        if "disease_pattern_similarity_summary" in existing:
            existing["disease_pattern_similarity_summary"].to_excel(
                writer, sheet_name="disease_pattern_similarity_summary", index=False
            )
        summary_df.to_excel(writer, sheet_name="disease_pattern_knn_summary", index=False)
        insuf_df.to_excel(writer, sheet_name="disease_pattern_knn_insufficient_stats", index=False)

    log.info("Updated %s", conflict_path)


if __name__ == "__main__":
    main()

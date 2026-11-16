"""
Cardiomegaly disease-level pattern similarity on conflict tuples.

Uses pred_disease_15w_pattern candidates vs train cluster files (first 14 filename chars)
and test8601 2048-d image features.
"""

from __future__ import annotations

from paths import DATA_ROOT, KNN_ROOT

import argparse
import ast
import json
import logging
import math
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DISEASE_ORDER_14 = [
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
TARGET_DISEASE = "Cardiomegaly"
TARGET_DISEASE_IDX = DISEASE_ORDER_14.index(TARGET_DISEASE)
GT_COL = "gt_report_clean_Cardiomegaly"
GPT_PRED_COL = "gpt_pred_15w_clean_Cardiomegaly"


def safe_eval(v: Any) -> Any:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (list, dict, tuple)):
        return v
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def pred_info_to_pattern(pred_info_val: Any) -> str:
    status_map: Dict[str, str] = {}
    obj = safe_eval(pred_info_val)
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                name = str(item[0])
                status = str(item[1]).strip().lower()
                if status == "positive":
                    status_map[name] = "1"
                elif status == "negative":
                    status_map[name] = "0"
                else:
                    status_map[name] = "0"
    bits = "".join(status_map.get(d, "0") for d in DISEASE_ORDER_14)
    if len(bits) != 14:
        raise ValueError(f"pattern must be 14 bits, got {len(bits)}")
    return "s" + bits


def validate_pattern(pattern: str) -> bool:
    return (
        isinstance(pattern, str)
        and len(pattern) == 15
        and pattern.startswith("s")
        and set(pattern[1:]) <= {"0", "1"}
    )


def candidate_patterns(pattern_s: str, disease_idx: int) -> Tuple[str, str]:
    """x-mask14: x + bits[1:13] + x (compare to train filename chars 1-14)."""
    bits14 = pattern_s[1:]
    if len(bits14) != 14:
        raise ValueError(f"expected 14 bits after s, got {len(bits14)}")
    b0 = list(bits14)
    b1 = list(bits14)
    b0[disease_idx] = "0"
    b1[disease_idx] = "1"
    return "x" + "".join(b0)[1:13] + "x", "x" + "".join(b1)[1:13] + "x"


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx <= 1e-12 or ny <= 1e-12:
        return float("nan")
    return float(np.dot(x, y) / (nx * ny))


def mahalanobis_diagonal(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    v = np.maximum(var.astype(np.float64), 1e-12)
    d2 = np.sum(((x.astype(np.float64) - mean.astype(np.float64)) ** 2) / v)
    if not np.isfinite(d2):
        return float("nan")
    return float(np.sqrt(max(d2, 0.0)))


CACHE_NAME = "_train_cluster_reps_cache.pkl"
DEFAULT_NC_X0000_JSON = KNN_ROOT / "cluster_meta.json"


def _load_one_train_file(fp_str: str) -> Tuple[str, Dict[str, Any]]:
    fp = Path(fp_str)
    prefix = fp.name.split("_p0_")[0]
    with fp.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    gp = obj.get("gaussian_params") or obj.get("gaussian_param") or {}
    return prefix, {
        "file": fp.name,
        "average": np.asarray(obj.get("average", []), dtype=np.float64),
        "geometric_median": np.asarray(obj.get("geometric_median", []), dtype=np.float64),
        "g_mean": np.asarray(gp.get("mean", []), dtype=np.float64),
        "g_var": np.asarray(gp.get("variance", []), dtype=np.float64),
    }


def build_train_cache(train_dir: Path, workers: int = 8) -> Dict[str, Dict[str, Any]]:
    files = sorted(str(p) for p in train_dir.glob("*.json") if not p.name.startswith("_"))
    index: Dict[str, Dict[str, Any]] = {}
    log.info("Building train cache from %d files (workers=%d)...", len(files), workers)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_load_one_train_file, fp): fp for fp in files}
        for i, fut in enumerate(as_completed(futures), start=1):
            prefix, obj = fut.result()
            if len(prefix) == 14:
                index[prefix] = obj
            if i % 100 == 0 or i == len(files):
                log.info("cache build progress %d/%d", i, len(files))
    cache_path = train_dir / CACHE_NAME
    with cache_path.open("wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("Saved train cache: %s", cache_path)
    return index


def load_train_index(train_dir: Path, workers: int = 8) -> Dict[str, Dict[str, Any]]:
    cache_path = train_dir / CACHE_NAME
    if cache_path.exists():
        with cache_path.open("rb") as f:
            index = pickle.load(f)
        log.info("Loaded train cache: %d clusters from %s", len(index), cache_path)
        return index
    return build_train_cache(train_dir, workers=workers)


def build_train_middle_index(
    train_index: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    """Map filename chars 2-13 (12 bits) -> train prefix (chars 1-14)."""
    mid_index: Dict[str, str] = {}
    for prefix in train_index:
        if len(prefix) != 14:
            continue
        mid = prefix[1:13]
        if mid in mid_index and mid_index[mid] != prefix:
            raise ValueError(f"ambiguous train middle {mid!r}: {mid_index[mid]!r} vs {prefix!r}")
        mid_index[mid] = prefix
    return mid_index


def resolve_train_reps(
    pattern: str,
    train_index: Dict[str, Dict[str, Any]],
    middle_index: Dict[str, str],
    nc_x0000: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Resolve x-mask14 pattern via filename chars 2-13 match to train prefix."""
    if len(pattern) == 14 and pattern[0] == "x" and pattern[-1] == "x":
        if pattern == "x000000000000x" and nc_x0000 is not None:
            return nc_x0000, f"normal_conditions:{nc_x0000['file']}"
        mid = pattern[1:13]
        prefix = middle_index.get(mid)
        if prefix is not None:
            return train_index[prefix], f"train:{train_index[prefix]['file']}"
        return None, "missing"
    if pattern in train_index:
        return train_index[pattern], f"train:{train_index[pattern]['file']}"
    return None, "missing"


def load_reps_json(path: Path, label: str = "normal_conditions") -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    gp = obj.get("gaussian_params") or obj.get("gaussian_param") or {}
    return {
        "file": path.name,
        "source": label,
        "average": np.asarray(obj.get("average", []), dtype=np.float64),
        "geometric_median": np.asarray(obj.get("geometric_median", []), dtype=np.float64),
        "g_mean": np.asarray(gp.get("mean", []), dtype=np.float64),
        "g_var": np.asarray(gp.get("variance", []), dtype=np.float64),
    }


def load_test_features(path: Path) -> Dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, np.ndarray] = {}
    for k, v in raw.items():
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim == 1:
            out[str(k)] = arr
    log.info("Loaded test features: %d", len(out))
    return out


def compare_candidates(
    feat: np.ndarray,
    train0: Optional[Dict[str, Any]],
    train1: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for suffix, tr in (("0", train0), ("1", train1)):
        if tr is None or feat.shape[0] != tr["average"].shape[0]:
            out[f"sim_cosine_average_{suffix}"] = np.nan
            out[f"sim_cosine_geometric_median_{suffix}"] = np.nan
            out[f"sim_mahalanobis_{suffix}"] = np.nan
            out[f"train_file_{suffix}"] = np.nan
            continue
        out[f"train_file_{suffix}"] = tr["file"]
        out[f"sim_cosine_average_{suffix}"] = cosine_similarity(feat, tr["average"])
        out[f"sim_cosine_geometric_median_{suffix}"] = cosine_similarity(feat, tr["geometric_median"])
        out[f"sim_mahalanobis_{suffix}"] = mahalanobis_diagonal(feat, tr["g_mean"], tr["g_var"])

    for method, higher_is_closer in (
        ("cosine_average", True),
        ("cosine_geometric_median", True),
        ("mahalanobis", False),
    ):
        if method.startswith("cosine"):
            k0, k1 = f"sim_{method}_0", f"sim_{method}_1"
            s0, s1 = out.get(k0), out.get(k1)
            if s0 is None or s1 is None or (isinstance(s0, float) and np.isnan(s0)) or (isinstance(s1, float) and np.isnan(s1)):
                out[f"pred_pattern_{method}"] = np.nan
            elif s0 > s1:
                out[f"pred_pattern_{method}"] = "negative"
            elif s1 > s0:
                out[f"pred_pattern_{method}"] = "positive"
            else:
                out[f"pred_pattern_{method}"] = "negative"
        else:
            s0, s1 = out.get("sim_mahalanobis_0"), out.get("sim_mahalanobis_1")
            if s0 is None or s1 is None or (isinstance(s0, float) and np.isnan(s0)) or (isinstance(s1, float) and np.isnan(s1)):
                out[f"pred_pattern_{method}"] = np.nan
            elif s0 < s1:
                out[f"pred_pattern_{method}"] = "negative"
            elif s1 < s0:
                out[f"pred_pattern_{method}"] = "positive"
            else:
                out[f"pred_pattern_{method}"] = "negative"
    return out


def norm_gt(v: Any) -> Optional[str]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip().lower()
    if s in ("negative", "blank"):
        return "negative"
    if s in ("positive", "uncertain"):
        return "positive"
    return None


def metrics_binary(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, Any]:
    mask = y_true.notna() & y_pred.notna()
    yt = y_true[mask]
    yp = y_pred[mask]
    n = int(len(yt))
    if n == 0:
        return {"n": 0, "accuracy": np.nan, "f1": np.nan, "TP": 0, "FP": 0, "TN": 0, "FN": 0}
    tp = int(((yt == "positive") & (yp == "positive")).sum())
    fp = int(((yt == "negative") & (yp == "positive")).sum())
    tn = int(((yt == "negative") & (yp == "negative")).sum())
    fn = int(((yt == "positive") & (yp == "negative")).sum())
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"n": n, "accuracy": acc, "f1": f1, "TP": tp, "FP": fp, "TN": tn, "FN": fn}


def attach_gpt_pred_by_id(df: pd.DataFrame, gpt_df: pd.DataFrame) -> pd.DataFrame:
    if "id" not in df.columns:
        return df
    out = df.drop(columns=[GPT_PRED_COL], errors="ignore").copy()
    out["id"] = out["id"].astype(str)
    merged = out.merge(gpt_df, on="id", how="left", validate="many_to_one")
    cols = [c for c in merged.columns if c != GPT_PRED_COL]
    if GPT_PRED_COL in merged.columns:
        insert_at = cols.index("id") + 1 if "id" in cols else 0
        cols = cols[:insert_at] + [GPT_PRED_COL] + cols[insert_at:]
    return merged[cols]


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    y_true = df[GT_COL].map(norm_gt)
    rows = []
    for method in ("cosine_average", "cosine_geometric_median", "mahalanobis"):
        col = f"pred_pattern_{method}"
        m = metrics_binary(y_true, df[col])
        rows.append({"method": method, "pred_col": col, **m})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conflict-xlsx",
        default="str(KNN_ROOT)",
    )
    parser.add_argument(
        "--test-features-json",
        default="str(KNN_ROOT)",
    )
    parser.add_argument(
        "--train-cluster-dir",
        default="str(KNN_ROOT)",
    )
    parser.add_argument(
        "--wacc-xlsx",
        default="str(KNN_ROOT)",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--nc-x0000-json",
        default=str(DEFAULT_NC_X0000_JSON),
        help="normal_conditions fallback for x000000000000x (default: nc100)",
    )
    args = parser.parse_args()

    conflict_path = Path(args.conflict_xlsx)
    base_df = pd.read_excel(conflict_path, sheet_name=0, engine="openpyxl")
    base_df["id"] = base_df["id"].astype(str)

    wacc = pd.read_excel(
        args.wacc_xlsx,
        sheet_name="Sheet1",
        usecols=["id", "pred_disease_info_chexber_uone_15w", GPT_PRED_COL],
        engine="openpyxl",
    )
    wacc["id"] = wacc["id"].astype(str)
    gpt_df = wacc[["id", GPT_PRED_COL]].drop_duplicates("id")
    wacc["pred_disease_15w_pattern"] = wacc["pred_disease_info_chexber_uone_15w"].map(pred_info_to_pattern)
    bad = wacc[~wacc["pred_disease_15w_pattern"].map(validate_pattern)]
    if len(bad):
        raise ValueError(f"invalid pred_disease_15w_pattern rows: {len(bad)}")

    feat_map = load_test_features(Path(args.test_features_json))
    train_index = load_train_index(Path(args.train_cluster_dir), workers=args.workers)
    middle_index = build_train_middle_index(train_index)
    nc_x0000_path = Path(args.nc_x0000_json)
    nc_x0000 = load_reps_json(nc_x0000_path) if nc_x0000_path.is_file() else None
    if nc_x0000 is not None:
        log.info("x000000000000x fallback: %s", nc_x0000_path)
    else:
        log.warning("x000000000000x fallback disabled: %s not found", nc_x0000_path)

    merge_cols = ["id", "pred_disease_info_chexber_uone_15w", "pred_disease_15w_pattern"]
    work = base_df.merge(wacc[merge_cols], on="id", how="left", validate="many_to_one")

    rows: List[Dict[str, Any]] = []
    missing_feat = missing_train = 0
    for i, row in work.iterrows():
        sid = str(row["id"])
        pattern_s = row["pred_disease_15w_pattern"]
        out = row.to_dict()
        try:
            c0, c1 = candidate_patterns(str(pattern_s), TARGET_DISEASE_IDX)
            out["candidate_pattern_0"] = c0
            out["candidate_pattern_1"] = c1
            if sid not in feat_map:
                missing_feat += 1
                raise KeyError(f"missing test feature for id={sid}")
            feat = feat_map[sid]
            tr0, src0 = resolve_train_reps(c0, train_index, middle_index, nc_x0000)
            tr1, src1 = resolve_train_reps(c1, train_index, middle_index, nc_x0000)
            out["resolve_src_0"] = src0
            out["resolve_src_1"] = src1
            if tr0 is None or tr1 is None:
                missing_train += 1
            out.update(compare_candidates(feat, tr0, tr1))
        except Exception as e:
            out["pattern_similarity_error"] = str(e)
            for method in ("cosine_average", "cosine_geometric_median", "mahalanobis"):
                out.setdefault(f"pred_pattern_{method}", np.nan)
        rows.append(out)
        if (len(rows) % 500) == 0:
            log.info("processed %d/%d", len(rows), len(work))

    result_df = pd.DataFrame(rows)
    summary_df = build_summary(result_df)
    log.info("missing_feat=%d missing_train=%d", missing_feat, missing_train)
    log.info("summary:\n%s", summary_df.to_string(index=False))

    with pd.ExcelFile(conflict_path, engine="openpyxl") as xl:
        existing = {name: pd.read_excel(xl, sheet_name=name) for name in xl.sheet_names}

    result_df = attach_gpt_pred_by_id(result_df, gpt_df)

    with pd.ExcelWriter(conflict_path, engine="openpyxl") as writer:
        for name, df in existing.items():
            if name in ("disease_pattern_similarity", "disease_pattern_similarity_summary"):
                continue
            attach_gpt_pred_by_id(df, gpt_df).to_excel(writer, sheet_name=name, index=False)
        result_df.to_excel(writer, sheet_name="disease_pattern_similarity", index=False)
        summary_df.to_excel(writer, sheet_name="disease_pattern_similarity_summary", index=False)

    log.info("Saved sheets to %s", conflict_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build conflict5 / conflict5_wregion info columns on CheXpert+ Sheet1."""

from __future__ import annotations

from paths import DATA_ROOT, KNN_ROOT

import argparse
import ast
import copy
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

DEFAULT_XLSX = KNN_ROOT / "test_input.xlsx"
# disease-region conflict5 (step 1)
DISEASE_SPECS: List[dict] = [
    {
        "disease": "Cardiomegaly",
        "pair_col": "pair_knn_norm_101_Cardiomegaly_cardiac_silhouette",
        "disease_pred_col": "pred_pattern_knn_norm_101_Cardiomegaly",
        "regions": [
            ("cardiac silhouette", "knn_pred_norm_101_cardiac_silhouette"),
        ],
    },
    {
        "disease": "Atelectasis",
        "pair_col": "pair_knn_norm_101_Atelectasis_left_lower_lung_zone+right_lower_lung_zone",
        "disease_pred_col": "pred_pattern_knn_norm_101_Atelectasis",
        "regions": [
            (
                "left lower lung zone",
                "knn_pred_norm_101_left_lower_lung_zone+right_lower_lung_zone",
            ),
            (
                "right lower lung zone",
                "knn_pred_norm_101_left_lower_lung_zone+right_lower_lung_zone",
            ),
        ],
    },
    {
        "disease": "Enlarged Cardiomediastinum",
        "pair_col": (
            "pair_knn_norm_101_Enlarged_Cardiomediastinum_"
            "upper_mediastinum+mediastinum"
        ),
        "disease_pred_col": "pred_pattern_knn_norm_101_Enlarged_Cardiomediastinum",
        "regions": [
            (
                "upper mediastinum",
                "knn_pred_norm_101_upper_mediastinum+mediastinum",
            ),
            (
                "mediastinum",
                "knn_pred_norm_101_upper_mediastinum+mediastinum",
            ),
        ],
    },
    {
        "disease": "Edema",
        "pair_col": "pair_knn_norm_101_Edema_left_hilar_structures+right_hilar_structures",
        "disease_pred_col": "pred_pattern_knn_norm_101_Edema",
        "regions": [
            (
                "left hilar structures",
                "knn_pred_norm_101_left_hilar_structures+right_hilar_structures",
            ),
            (
                "right hilar structures",
                "knn_pred_norm_101_left_hilar_structures+right_hilar_structures",
            ),
        ],
    },
    {
        "disease": "Pleural Effusion",
        "pair_col": (
            "pair_knn_norm_101_Pleural_Effusion_"
            "left_costophrenic_angle+right_costophrenic_angle"
        ),
        "disease_pred_col": "pred_pattern_knn_norm_101_Pleural_Effusion",
        "regions": [
            (
                "left costophrenic angle",
                "knn_pred_norm_101_left_costophrenic_angle+right_costophrenic_angle",
            ),
            (
                "right costophrenic angle",
                "knn_pred_norm_101_left_costophrenic_angle+right_costophrenic_angle",
            ),
        ],
    },
]

# region-only conflicts (step 2)
REGION_ONLY_SPECS: List[dict] = [
    {
        "pair_col": "pair_knn_norm_101_mediastinum_cardiac_silhouette",
        "regions": [
            ("mediastinum", "knn_pred_norm_101_mediastinum"),
            ("cardiac silhouette", "knn_pred_norm_101_cardiac_silhouette"),
        ],
    },
    {
        "pair_col": "pair_knn_norm_101_upper_mediastinum_aortic_arch",
        "regions": [
            ("upper mediastinum", "knn_pred_norm_101_upper_mediastinum"),
            ("aortic arch", "knn_pred_norm_101_aortic_arch"),
        ],
    },
]

VALID_PAIRS = {"pos_pair", "neg_pair"}


def safe_eval_list(v: Any) -> List[Any]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return []
    if isinstance(v, list):
        return copy.deepcopy(v)
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return []
    try:
        obj = ast.literal_eval(s)
        return copy.deepcopy(obj) if isinstance(obj, list) else []
    except Exception:
        return []


def list_repr(item: Sequence[Any]) -> str:
    return str(list(item))


def slug_region(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def slug_disease(name: str) -> str:
    return name.strip().replace(" ", "_")


def norm_dis(v: Any) -> Optional[str]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = str(v).strip().lower()
    if s in ("positive", "pos"):
        return "positive"
    if s in ("negative", "neg"):
        return "negative"
    return None


def norm_reg(v: Any) -> Optional[str]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = str(v).strip().lower()
    if s == "abnormal":
        return "abnormal"
    if s == "normal":
        return "normal"
    return None


def dis_label(v: str) -> str:
    return "Positive" if v == "positive" else "Negative"


def reg_label(v: str) -> str:
    return "Abnormal" if v == "abnormal" else "Normal"


def is_valid_pair(v: Any) -> bool:
    return str(v).strip() in VALID_PAIRS


def find_idx(info: List[Any], name: str) -> int:
    for i, item in enumerate(info):
        if isinstance(item, (list, tuple)) and len(item) >= 2 and str(item[0]) == name:
            return i
    return -1


def apply_status_replace(
    info: List[Any],
    name: str,
    new_status: Optional[str],
    *,
    kind: str,
) -> Tuple[List[Any], str, bool]:
    """Return (info, note, did_change). kind in {'disease','region'}."""
    idx = find_idx(info, name)
    if idx < 0:
        return info, f"{name} not found in original info; cannot replace", False
    old_item = list(info[idx])
    if kind == "disease":
        old_s, to_lab, norm_fn = norm_dis(old_item[1]), dis_label, norm_dis
    else:
        old_s, to_lab, norm_fn = norm_reg(old_item[1]), reg_label, norm_reg
    if new_status is None:
        return info, f"missing knn prediction; keep original: {list_repr(old_item)}", False
    if old_s == new_status:
        return (
            info,
            f"info consistent; keep original with confidence: {list_repr(old_item)}",
            False,
        )
    new_item = [name, to_lab(new_status)]
    info[idx] = new_item
    return info, f"replaced {list_repr(old_item)} with {list_repr(new_item)}", True


def process_sheet(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    n = len(out)

    # init outputs
    pred_c5: List[str] = []
    reg_c5: List[str] = []
    reg_w: List[str] = []

    note_cols: Dict[str, List[Any]] = {}
    for spec in DISEASE_SPECS:
        dslug = slug_disease(spec["disease"])
        note_cols[f"conflict5_note_disease_{dslug}"] = [pd.NA] * n
        for rname, _ in spec["regions"]:
            note_cols[f"conflict5_note_region_{slug_region(rname)}"] = [pd.NA] * n
    for spec in REGION_ONLY_SPECS:
        for rname, _ in spec["regions"]:
            key = f"conflict5_wregion_note_region_{slug_region(rname)}"
            if key not in note_cols:
                note_cols[key] = [pd.NA] * n

    # regions touched by step1 valid disease pairs (per row)
    step1_touched_regions: List[set] = [set() for _ in range(n)]

    for i, row in out.iterrows():
        # pandas iterrows index may not be 0..n-1 positional; use enumerate
        pass

    for pos, (_, row) in enumerate(out.iterrows()):
        dis_info = safe_eval_list(row.get("pred_info_mimic15w"))
        reg_info = safe_eval_list(row.get("region_info_only_detected_mimic15w"))
        touched: set = set()

        for spec in DISEASE_SPECS:
            dname = spec["disease"]
            dslug = slug_disease(dname)
            dnote_col = f"conflict5_note_disease_{dslug}"
            if not is_valid_pair(row.get(spec["pair_col"])):
                continue
            # disease
            dis_info, dnote, _ = apply_status_replace(
                dis_info,
                dname,
                norm_dis(row.get(spec["disease_pred_col"])),
                kind="disease",
            )
            note_cols[dnote_col][pos] = dnote
            # regions
            for rname, rcol in spec["regions"]:
                touched.add(rname)
                reg_info, rnote, _ = apply_status_replace(
                    reg_info,
                    rname,
                    norm_reg(row.get(rcol)),
                    kind="region",
                )
                note_cols[f"conflict5_note_region_{slug_region(rname)}"][pos] = rnote

        step1_touched_regions[pos] = touched
        pred_c5.append(str(dis_info))
        reg_c5.append(str(reg_info))

        # step 2: further region-only on a copy of conflict5 region info
        reg_w_info = copy.deepcopy(reg_info)
        for spec in REGION_ONLY_SPECS:
            if not is_valid_pair(row.get(spec["pair_col"])):
                continue
            for rname, rcol in spec["regions"]:
                wnote_col = f"conflict5_wregion_note_region_{slug_region(rname)}"
                if rname in touched:
                    note_cols[wnote_col][pos] = (
                        f"skip: {rname} already handled in conflict5; do not override step-1 result"
                    )
                    continue
                reg_w_info, rnote, _ = apply_status_replace(
                    reg_w_info,
                    rname,
                    norm_reg(row.get(rcol)),
                    kind="region",
                )
                note_cols[wnote_col][pos] = rnote
        reg_w.append(str(reg_w_info))

    out["pred_info_mimic15w_conflict5"] = pred_c5
    out["region_info_only_detected_mimic15w_conflict5"] = reg_c5
    out["region_info_only_detected_mimic15w_conflict5_wregion"] = reg_w
    for c, vals in note_cols.items():
        out[c] = vals
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default=str(DEFAULT_XLSX))
    args = ap.parse_args()
    xlsx = Path(args.xlsx)

    bak = xlsx.with_suffix(".xlsx.bak_before_conflict5_info")
    if not bak.exists():
        shutil.copy2(xlsx, bak)
        print(f"backup -> {bak}")

    print(f"loading {xlsx}")
    book = pd.read_excel(xlsx, sheet_name=None, engine="openpyxl")
    s1 = book["Sheet1"]
    required = ["pred_info_mimic15w", "region_info_only_detected_mimic15w"]
    for spec in DISEASE_SPECS + REGION_ONLY_SPECS:
        required.append(spec["pair_col"])
        if "disease_pred_col" in spec:
            required.append(spec["disease_pred_col"])
        for _, c in spec["regions"]:
            required.append(c)
    missing = [c for c in required if c not in s1.columns]
    if missing:
        raise SystemExit(f"missing columns: {missing}")

    print("applying conflict5 replacements …")
    book["Sheet1"] = process_sheet(s1)

    # quick stats
    s1n = book["Sheet1"]
    for spec in DISEASE_SPECS:
        dslug = slug_disease(spec["disease"])
        col = f"conflict5_note_disease_{dslug}"
        vc = s1n[col].dropna().astype(str)
        n_rep = int(vc.str.startswith("replaced ").sum())
        n_keep = int(vc.str.startswith("info consistent").sum())
        print(f"  disease {spec['disease']}: replace={n_rep} keep={n_keep} notes={vc.shape[0]}")
    for rname in sorted(
        {
            r
            for spec in DISEASE_SPECS
            for r, _ in spec["regions"]
        }
    ):
        col = f"conflict5_note_region_{slug_region(rname)}"
        vc = s1n[col].dropna().astype(str)
        n_rep = int(vc.str.startswith("replaced ").sum())
        n_keep = int(vc.str.startswith("info consistent").sum())
        print(f"  region {rname}: replace={n_rep} keep={n_keep} notes={vc.shape[0]}")
    for rname in ("mediastinum", "cardiac silhouette", "upper mediastinum", "aortic arch"):
        col = f"conflict5_wregion_note_region_{slug_region(rname)}"
        vc = s1n[col].dropna().astype(str)
        n_rep = int(vc.str.startswith("replaced ").sum())
        n_skip = int(vc.str.startswith("skip:").sum())
        n_keep = int(vc.str.startswith("info consistent").sum())
        print(
            f"  wregion {rname}: replace={n_rep} keep={n_keep} skip={n_skip} notes={vc.shape[0]}"
        )

    tmp = xlsx.with_suffix(".xlsx.tmp_conflict5")
    print(f"writing {tmp} …")
    with pd.ExcelWriter(tmp, engine="openpyxl") as w:
        for name, df in book.items():
            df.to_excel(w, sheet_name=name[:31], index=False)
    tmp.replace(xlsx)
    print(f"done -> {xlsx} ncols={book['Sheet1'].shape[1]}")


if __name__ == "__main__":
    main()

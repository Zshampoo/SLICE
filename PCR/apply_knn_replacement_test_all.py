"""
Apply knn_norm_101 replacements on test_all_KNNnorm101.xlsx Sheet6 (all tuples).

Merge 5 disease sheets by id, update pred_disease/region info lists,
extract knn_tuple_* columns, validate replacements.
"""

from __future__ import annotations

from paths import DATA_ROOT, KNN_ROOT

import argparse
import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

DEFAULT_XLSX = KNN_ROOT / "input.xlsx"
ALL_TUPLES_SHEETS = ("Sheet6", "all_tuples")
OUT_SHEET = "all_tuples"

SHEET_CONFIG = {
    "EDEMA": {
        "disease": "Edema",
        "disease_col": "edema_knn_norm_101",
        "regions": [
            ("left hilar structures", "left hilar structures_knn_pred_norm_101"),
            ("right hilar structures", "right hilar structures_knn_pred_norm_101"),
        ],
        "pair_rule": "both",
    },
    "CARDIOMEGALY": {
        "disease": "Cardiomegaly",
        "disease_col": "cardiomegaly_knn_norm_101",
        "regions": [("cardiac silhouette", "cardiac silhouette_knn_pred_norm_101")],
        "pair_rule": "single",
    },
    "Enlarged Cardiomediastinum": {
        "disease": "Enlarged Cardiomediastinum",
        "disease_col": "Enlarged Cardiomediastinum_knn_norm_101",
        "regions": [
            ("mediastinum", "mediastinum_knn_pred_norm_101"),
            ("upper mediastinum", "upper mediastinum_knn_pred_norm_101"),
        ],
        "pair_rule": "both",
    },
    "Pleural effusion": {
        "disease": "Pleural Effusion",
        "disease_col": "pleural effusion_knn_norm_101",
        "regions": [
            ("left costophrenic angle", "left costophrenic angle_knn_pred_norm_101"),
            ("right costophrenic angle", "right costophrenic angle_knn_pred_norm_101"),
        ],
        "pair_rule": "any",
    },
    "Atelectasis": {
        "disease": "Atelectasis",
        "disease_col": "Atelectasis_knn_norm_101",
        "regions": [
            ("left lower lung zone", "left lower lung zone_knn_pred_norm_101"),
            ("right lower lung zone", "right lower lung zone_knn_pred_norm_101"),
        ],
        "pair_rule": "any",
    },
}

DISEASE_NAMES = [cfg["disease"] for cfg in SHEET_CONFIG.values()]
REGION_NAMES = [r for cfg in SHEET_CONFIG.values() for r, _ in cfg["regions"]]


def safe_eval_list(v: Any) -> List[Any]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    if isinstance(v, list):
        return v
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return []
    try:
        obj = ast.literal_eval(s)
        return obj if isinstance(obj, list) else []
    except Exception:
        return []


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def norm_dis(v: Any) -> Optional[str]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip().lower()
    if s in ("positive", "pos"):
        return "positive"
    if s in ("negative", "neg"):
        return "negative"
    return None


def norm_reg(v: Any) -> Optional[str]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip().lower()
    if s == "abnormal":
        return "abnormal"
    if s == "normal":
        return "normal"
    return None


def dis_to_label(v: str) -> str:
    return "Positive" if v == "positive" else "Negative"


def reg_to_label(v: str) -> str:
    return "Abnormal" if v == "abnormal" else "Normal"


def find_item(info: List[Any], name: str) -> Optional[List[Any]]:
    for item in info:
        if isinstance(item, (list, tuple)) and len(item) >= 2 and str(item[0]) == name:
            return list(item)
    return None


def replace_item(
    item: List[Any],
    new_status: Optional[str],
    to_label_fn,
    old_norm_fn,
    scope_label: str,
) -> Tuple[List[Any], str]:
    old_status = old_norm_fn(item[1])
    if new_status is None:
        return list(item), f"keep_based_on_{scope_label}(no_knn)"
    if old_status == new_status:
        return list(item), f"keep_based_on_{scope_label}"
    return [item[0], to_label_fn(new_status)], f"replace_based_on_{scope_label}"


def merge_knn_sheets(xlsx_path: Path) -> pd.DataFrame:
    frames = []
    for sheet_name, cfg in SHEET_CONFIG.items():
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl")
        df["id"] = df["id"].astype(str)
        keep = ["id", cfg["disease_col"]] + [c for _, c in cfg["regions"]]
        keep = [c for c in keep if c in df.columns]
        frames.append(df[keep].copy())
    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="id", how="outer")
    return merged


def extract_named_tuples(info: List[Any], names: List[str]) -> List[List[Any]]:
    extracted: List[List[Any]] = []
    for name in names:
        item = find_item(info, name)
        if item is not None:
            extracted.append(list(item))
    return extracted


def apply_row(
    row: pd.Series,
    dis_src: Any,
    reg_src: Any,
) -> Dict[str, Any]:
    dis_info = safe_eval_list(dis_src)
    reg_info = safe_eval_list(reg_src)
    dis_out: List[Any] = []
    reg_out: List[Any] = []
    out: Dict[str, Any] = {}

    out["pred_disease_info_chexber_uone_15w_five_diseases"] = str(
        extract_named_tuples(dis_info, DISEASE_NAMES)
    )
    out["pred_region_info_15w_five_regions"] = str(
        extract_named_tuples(reg_info, REGION_NAMES)
    )

    for item in dis_info:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            dis_out.append(item)
            continue
        name = str(item[0])
        cfg = next((c for c in SHEET_CONFIG.values() if c["disease"] == name), None)
        if cfg is None:
            dis_out.append(list(item))
            continue
        new_raw = row.get(cfg["disease_col"])
        new_status = norm_dis(new_raw)
        new_item, action = replace_item(
            item, new_status, dis_to_label, norm_dis, f"{name}_disease"
        )
        dis_out.append(new_item)
        out[f"disease_knn_action_{name}"] = action

    for item in reg_info:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            reg_out.append(item)
            continue
        name = str(item[0])
        src_col = None
        for cfg in SHEET_CONFIG.values():
            for rname, col in cfg["regions"]:
                if rname == name:
                    src_col = col
                    break
        if src_col is None:
            reg_out.append(list(item))
            continue
        new_raw = row.get(src_col)
        new_status = norm_reg(new_raw)
        new_item, action = replace_item(
            item, new_status, reg_to_label, norm_reg, f"{name}_region"
        )
        reg_out.append(new_item)
        out[f"region_knn_action_{slug(name)}"] = action

    out["pred_disease_info_chexber_uone_15w_knn"] = str(dis_out)
    out["pred_region_info_15w_knn"] = str(reg_out)

    dis_knn = safe_eval_list(out["pred_disease_info_chexber_uone_15w_knn"])
    reg_knn = safe_eval_list(out["pred_region_info_15w_knn"])

    for cfg in SHEET_CONFIG.values():
        d = cfg["disease"]
        d_item = find_item(dis_knn, d)
        out[f"knn_tuple_{d.replace(' ', '_')}"] = str(d_item) if d_item else ""
        if d_item:
            out[f"knn_tuple_{d.replace(' ', '_')}_condition"] = str(d_item[1]).strip().lower()
            out[f"knn_tuple_{d.replace(' ', '_')}_cs"] = (
                float(d_item[2]) if len(d_item) >= 3 and d_item[2] is not None else None
            )
        else:
            out[f"knn_tuple_{d.replace(' ', '_')}_condition"] = None
            out[f"knn_tuple_{d.replace(' ', '_')}_cs"] = None

        for rname, _ in cfg["regions"]:
            r_item = find_item(reg_knn, rname)
            col = f"knn_tuple_{slug(rname)}"
            out[col] = str(r_item) if r_item else ""
            out[f"{col}_condition"] = (
                str(r_item[1]).strip().lower() if r_item else None
            )
            out[f"{col}_cs"] = (
                float(r_item[2]) if r_item and len(r_item) >= 3 and r_item[2] is not None else None
            )

        out.update(check_pair_consistency(cfg, d_item, reg_knn))
        for rname, _ in cfg["regions"]:
            r_item = find_item(reg_knn, rname)
            pair_key = f"{d.replace(' ', '_')}_{slug(rname)}"
            out[f"consistency_{pair_key}"] = check_single_pair_consistency(d_item, r_item)

    action_cols = [
        c
        for c in out
        if c.startswith("disease_knn_action_") or c.startswith("region_knn_action_")
    ]
    any_replaced = any(str(out[c]).startswith("replace_based_on_") for c in action_cols)
    out["knn_five_disease_region_none_replaced"] = "no" if any_replaced else "yes"

    return out


def check_single_pair_consistency(
    disease_item: Optional[List[Any]],
    region_item: Optional[List[Any]],
) -> str:
    if disease_item is None or region_item is None:
        return "missing"
    d = norm_dis(disease_item[1])
    r = norm_reg(region_item[1])
    if d is None or r is None:
        return "invalid"
    if d == "positive" and r == "abnormal":
        return "consistent(pos-abnormal)"
    if d == "negative" and r == "normal":
        return "consistent(neg-normal)"
    if d == "positive" and r == "normal":
        return "inconsistent(disease_pos_region_normal)"
    if d == "negative" and r == "abnormal":
        return "inconsistent(disease_neg_region_abnormal)"
    return "invalid"


def check_pair_consistency(
    cfg: Dict[str, Any],
    d_item: Optional[List[Any]],
    reg_knn: List[Any],
) -> Dict[str, Any]:
    disease = cfg["disease"]
    d_key = disease.replace(" ", "_")
    out: Dict[str, Any] = {}

    if d_item is None:
        out[f"consistency_{d_key}"] = "missing_disease_tuple"
        return out

    d = norm_dis(d_item[1])
    regions = []
    for rname, _ in cfg["regions"]:
        r_item = find_item(reg_knn, rname)
        r = norm_reg(r_item[1]) if r_item else None
        regions.append((rname, r))

    if cfg["pair_rule"] == "single":
        r = regions[0][1]
        if d is None or r is None:
            out[f"consistency_{d_key}"] = "invalid"
        elif d == "positive" and r == "abnormal":
            out[f"consistency_{d_key}"] = "consistent(pos-abnormal)"
        elif d == "negative" and r == "normal":
            out[f"consistency_{d_key}"] = "consistent(neg-normal)"
        else:
            out[f"consistency_{d_key}"] = "inconsistent"
    elif cfg["pair_rule"] == "both":
        if any(r is None for _, r in regions):
            out[f"consistency_{d_key}"] = "invalid"
        elif d == "positive" and all(r == "abnormal" for _, r in regions):
            out[f"consistency_{d_key}"] = "consistent(pos-both_abnormal)"
        elif d == "negative" and all(r == "normal" for _, r in regions):
            out[f"consistency_{d_key}"] = "consistent(neg-both_normal)"
        else:
            out[f"consistency_{d_key}"] = "inconsistent"
    else:  # any
        if any(r is None for _, r in regions):
            out[f"consistency_{d_key}"] = "invalid"
        elif d == "positive" and any(r == "abnormal" for _, r in regions):
            out[f"consistency_{d_key}"] = "consistent(pos-at_least_one_abnormal)"
        elif d == "negative" and all(r == "normal" for _, r in regions):
            out[f"consistency_{d_key}"] = "consistent(neg-both_normal)"
        else:
            out[f"consistency_{d_key}"] = "inconsistent"
    return out


def build_validation(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    n = len(df)
    rows.append({"scope": "all", "metric": "num_samples", "value": n})

    if "knn_five_disease_region_none_replaced" in df.columns:
        for cat, cnt in df["knn_five_disease_region_none_replaced"].value_counts().items():
            rows.append(
                {
                    "scope": "all",
                    "metric": "no_replacement_on_five_disease_regions",
                    "category": cat,
                    "value": int(cnt),
                }
            )

    for cfg in SHEET_CONFIG.values():
        disease = cfg["disease"]
        act_col = f"disease_knn_action_{disease}"
        if act_col in df.columns:
            acts = df[act_col].astype(str)
            replaced = int(acts.str.startswith("replace_based_on_").sum())
            kept = int(acts.str.startswith("keep_based_on_").sum())
            no_knn = int(acts.str.contains("no_knn", na=False).sum())
            rows.append(
                {
                    "scope": disease,
                    "metric": "disease_replacements",
                    "value": replaced,
                    "kept": kept,
                    "no_knn": no_knn,
                }
            )
        for rname, _ in cfg["regions"]:
            ract = f"region_knn_action_{slug(rname)}"
            if ract in df.columns:
                replaced = int(df[ract].astype(str).str.startswith("replace_based_on_").sum())
                rows.append(
                    {
                        "scope": f"{disease}/{rname}",
                        "metric": "region_replacements",
                        "value": replaced,
                    }
                )
        ccol = f"consistency_{disease.replace(' ', '_')}"
        if ccol in df.columns:
            for cat, cnt in df[ccol].value_counts().items():
                rows.append(
                    {
                        "scope": disease,
                        "metric": "pair_consistency",
                        "category": cat,
                        "value": int(cnt),
                    }
                )

    for cfg in SHEET_CONFIG.values():
        disease = cfg["disease"]
        d_key = disease.replace(" ", "_")
        for rname, _ in cfg["regions"]:
            pair_col = f"consistency_{d_key}_{slug(rname)}"
            if pair_col not in df.columns:
                continue
            ok = int(df[pair_col].astype(str).str.startswith("consistent").sum())
            checked = int(df[pair_col].notna().sum())
            rows.append(
                {
                    "scope": f"check/{disease}+{rname}",
                    "metric": "single_region_pair_consistent_samples",
                    "value": ok,
                    "checked": checked,
                }
            )
            for cat, cnt in df[pair_col].value_counts().items():
                rows.append(
                    {
                        "scope": f"check/{disease}+{rname}",
                        "metric": "single_region_pair_distribution",
                        "category": cat,
                        "value": int(cnt),
                    }
                )
    return pd.DataFrame(rows)


def process(xlsx_path: Path, out_path: Path) -> None:
    knn = merge_knn_sheets(xlsx_path)
    knn["id"] = knn["id"].astype(str)

    with pd.ExcelFile(xlsx_path, engine="openpyxl") as xl:
        sheets = {n: pd.read_excel(xl, sheet_name=n) for n in xl.sheet_names}

    src_sheet = next((s for s in ALL_TUPLES_SHEETS if s in sheets), None)
    if src_sheet is None:
        raise KeyError(f"Source sheet not found; expected one of: {ALL_TUPLES_SHEETS}")

    base = sheets[src_sheet].copy()
    base["id"] = base["id"].astype(str)
    knn_cols = {c for cfg in SHEET_CONFIG.values() for c in [cfg["disease_col"]] + [rc for _, rc in cfg["regions"]]}
    drop_cols = [
        c
        for c in base.columns
        if c in knn_cols
        or c.startswith(
            (
                "disease_knn_action_",
                "region_knn_action_",
                "knn_tuple_",
                "consistency_",
            )
        )
        or c
        in (
            "pred_disease_info_chexber_uone_15w_knn",
            "pred_region_info_15w_knn",
            "pred_disease_info_chexber_uone_15w_five_diseases",
            "pred_region_info_15w_five_regions",
            "knn_five_disease_region_none_replaced",
        )
    ]
    if drop_cols:
        base = base.drop(columns=drop_cols, errors="ignore")
    merged = base.merge(knn, on="id", how="left")

    extra_rows: List[Dict[str, Any]] = []
    for _, row in merged.iterrows():
        extra_rows.append(
            apply_row(
                row,
                row.get("pred_disease_info_chexber_uone_15w"),
                row.get("pred_region_info_15w"),
            )
        )
    extra_df = pd.DataFrame(extra_rows)
    out = pd.concat([merged.reset_index(drop=True), extra_df], axis=1)

    validation = build_validation(out)
    sheets[OUT_SHEET] = out
    sheets["knn_replacement_validation"] = validation
    for old in ALL_TUPLES_SHEETS:
        if old in sheets and old != OUT_SHEET:
            del sheets[old]

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)

    print(f"processed rows: {len(out)}")
    print("\n=== Per-disease replacement counts ===")
    for cfg in SHEET_CONFIG.values():
        col = f"disease_knn_action_{cfg['disease']}"
        if col in out.columns:
            print(cfg["disease"], out[col].value_counts().to_string())
    print(f"\nSaved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-xlsx", default=str(DEFAULT_XLSX))
    parser.add_argument("--output-xlsx", default=str(DEFAULT_XLSX))
    args = parser.parse_args()
    process(Path(args.input_xlsx), Path(args.output_xlsx))


if __name__ == "__main__":
    main()

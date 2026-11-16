#!/usr/bin/env python3
"""ReXGradient: rename disease-sheet pair_knn cols; append knn pair cols to Sheet1;
build pred_dis_info_rexgradientCP_knn / region_info_only_detected_mimic15w_knn + notes.

Mirrors CheXpert+ apply_chexpertplus_conflict5_info_replace.py, but:
  - disease baseline source = pred_dis_info_rexgradientCP
  - region source = region_info_only_detected_mimic (Sheet1)
  - only 5 disease conflicts (no region-only step)
  - note cols named conflict5_note_knn_disease_* / conflict5_note_knn_region_*
"""
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

DEFAULT_XLSX = KNN_ROOT / "input.xlsx"

# 2048 disease sheets → Sheet1 column naming (match CheXpert+ style)
DISEASE_SPECS: List[dict] = [
    {
        "sheet": "Cardiomegaly",
        "disease": "Cardiomegaly",
        "regions": ("cardiac silhouette",),
        "pair_suffix": "Cardiomegaly_cardiac_silhouette",
        "region_pred_suffix": "cardiac_silhouette",  # Sheet1 knn_pred col suffix
    },
    {
        "sheet": "Atelectasis",
        "disease": "Atelectasis",
        "regions": ("left lower lung zone", "right lower lung zone"),
        "pair_suffix": "Atelectasis_left_lower_lung_zone+right_lower_lung_zone",
        "region_pred_suffix": "left_lower_lung_zone+right_lower_lung_zone",
    },
    {
        "sheet": "Enlarged_CM",
        "disease": "Enlarged Cardiomediastinum",
        "regions": ("upper mediastinum", "mediastinum"),
        "pair_suffix": "Enlarged_Cardiomediastinum_upper_mediastinum+mediastinum",
        "region_pred_suffix": "upper_mediastinum+mediastinum",
    },
    {
        "sheet": "Edema",
        "disease": "Edema",
        "regions": ("left hilar structures", "right hilar structures"),
        "pair_suffix": "Edema_left_hilar_structures+right_hilar_structures",
        "region_pred_suffix": "left_hilar_structures+right_hilar_structures",
    },
    {
        "sheet": "Pleural_Effusion",
        "disease": "Pleural Effusion",
        "regions": ("left costophrenic angle", "right costophrenic angle"),
        "pair_suffix": "Pleural_Effusion_left_costophrenic_angle+right_costophrenic_angle",
        "region_pred_suffix": "left_costophrenic_angle+right_costophrenic_angle",
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
    idx = find_idx(info, name)
    if idx < 0:
        return info, f"{name} not found in original info; cannot replace", False
    old_item = list(info[idx])
    if kind == "disease":
        old_s, to_lab = norm_dis(old_item[1]), dis_label
    else:
        old_s, to_lab = norm_reg(old_item[1]), reg_label
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


def rename_pair_col_on_disease_sheets(book: Dict[str, pd.DataFrame]) -> None:
    for spec in DISEASE_SPECS:
        sheet = spec["sheet"]
        if sheet not in book:
            raise KeyError(f"missing disease sheet {sheet}")
        df = book[sheet]
        old = "pair_knn_norm_101"
        new = f"pair_knn_norm_101_{spec['pair_suffix']}"
        if old in df.columns:
            df = df.rename(columns={old: new})
        elif new not in df.columns:
            raise KeyError(f"{sheet}: neither {old} nor {new}")
        book[sheet] = df
        print(f"  {sheet}: pair col -> {new}")


def append_knn_cols_to_sheet1(book: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    s1 = book["Sheet1"].copy()
    id_col = "id" if "id" in s1.columns else "ID"
    s1["_id_key"] = s1[id_col].astype(str)

    for spec in DISEASE_SPECS:
        sheet = spec["sheet"]
        dslug = slug_disease(spec["disease"])
        pair_col = f"pair_knn_norm_101_{spec['pair_suffix']}"
        dis_pred_col = f"pred_pattern_knn_norm_101_{dslug}"
        reg_pred_col = f"knn_pred_norm_101_{spec['region_pred_suffix']}"

        ddf = book[sheet].copy()
        ddf["_id_key"] = ddf["ID"].astype(str)
        # source cols on disease sheet
        src_pair = pair_col if pair_col in ddf.columns else "pair_knn_norm_101"
        need = {
            src_pair: pair_col,
            "pred_pattern_knn_norm_101": dis_pred_col,
            "knn_pred_norm_101": reg_pred_col,  # combined region pred
        }
        for src, dst in need.items():
            if src not in ddf.columns:
                raise KeyError(f"{sheet} missing {src}")
        sub = ddf[["_id_key", src_pair, "pred_pattern_knn_norm_101", "knn_pred_norm_101"]].copy()
        sub = sub.rename(
            columns={
                src_pair: pair_col,
                "pred_pattern_knn_norm_101": dis_pred_col,
                "knn_pred_norm_101": reg_pred_col,
            }
        )
        # only keep pos/neg pair labels on Sheet1 pair col; others -> NA
        sub[pair_col] = sub[pair_col].where(sub[pair_col].astype(str).isin(VALID_PAIRS), pd.NA)
        # drop existing then merge
        drop = [c for c in (pair_col, dis_pred_col, reg_pred_col) if c in s1.columns]
        if drop:
            s1 = s1.drop(columns=drop)
        s1 = s1.merge(sub, on="_id_key", how="left")
        n_pair = int(s1[pair_col].isin(list(VALID_PAIRS)).sum())
        print(f"  Sheet1 append {pair_col}: valid_pairs={n_pair}")

    return s1


def apply_knn_updates(s1: pd.DataFrame) -> pd.DataFrame:
    out = s1.copy()
    n = len(out)
    dis_src = "pred_dis_info_rexgradientCP"
    reg_src = (
        "region_info_only_detected_mimic15w"
        if "region_info_only_detected_mimic15w" in out.columns
        else "region_info_only_detected_mimic"
    )
    if dis_src not in out.columns:
        raise KeyError(dis_src)
    if reg_src not in out.columns:
        raise KeyError(reg_src)

    pred_knn: List[str] = []
    reg_knn: List[str] = []
    note_cols: Dict[str, List[Any]] = {}
    for spec in DISEASE_SPECS:
        dslug = slug_disease(spec["disease"])
        note_cols[f"conflict5_note_knn_disease_{dslug}"] = [pd.NA] * n
        for rname in spec["regions"]:
            note_cols[f"conflict5_note_knn_region_{slug_region(rname)}"] = [pd.NA] * n

    for pos, (_, row) in enumerate(out.iterrows()):
        dis_info = safe_eval_list(row.get(dis_src))
        reg_info = safe_eval_list(row.get(reg_src))

        for spec in DISEASE_SPECS:
            dname = spec["disease"]
            dslug = slug_disease(dname)
            pair_col = f"pair_knn_norm_101_{spec['pair_suffix']}"
            dis_pred_col = f"pred_pattern_knn_norm_101_{dslug}"
            reg_pred_col = f"knn_pred_norm_101_{spec['region_pred_suffix']}"
            if not is_valid_pair(row.get(pair_col)):
                continue
            dis_info, dnote, _ = apply_status_replace(
                dis_info,
                dname,
                norm_dis(row.get(dis_pred_col)),
                kind="disease",
            )
            note_cols[f"conflict5_note_knn_disease_{dslug}"][pos] = dnote
            # combined region pred applied to each listed region (CheXpert+ style)
            rpred = norm_reg(row.get(reg_pred_col))
            for rname in spec["regions"]:
                reg_info, rnote, _ = apply_status_replace(
                    reg_info, rname, rpred, kind="region"
                )
                note_cols[f"conflict5_note_knn_region_{slug_region(rname)}"][pos] = rnote

        pred_knn.append(str(dis_info))
        reg_knn.append(str(reg_info))

    out["pred_dis_info_rexgradientCP_knn"] = pred_knn
    out["region_info_only_detected_mimic15w_knn"] = reg_knn
    for c, vals in note_cols.items():
        out[c] = vals
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    args = ap.parse_args()
    xlsx = args.xlsx

    bak = xlsx.with_suffix(".xlsx.bak_before_conflict5_knn")
    if not bak.exists():
        shutil.copy2(xlsx, bak)
        print(f"backup -> {bak}")

    print(f"loading {xlsx}")
    book = pd.read_excel(xlsx, sheet_name=None, engine="openpyxl")

    print("1) rename pair_knn_norm_101 on disease sheets …")
    rename_pair_col_on_disease_sheets(book)

    print("2) append pair / pred_pattern / region knn cols to Sheet1 …")
    s1 = append_knn_cols_to_sheet1(book)

    print("3) build *_knn info + conflict5_note_knn_* …")
    s1 = apply_knn_updates(s1)
    if "_id_key" in s1.columns:
        s1 = s1.drop(columns=["_id_key"])
    book["Sheet1"] = s1

    # stats
    for spec in DISEASE_SPECS:
        dslug = slug_disease(spec["disease"])
        col = f"conflict5_note_knn_disease_{dslug}"
        vc = s1[col].dropna().astype(str)
        n_rep = int(vc.str.startswith("replaced ").sum())
        n_keep = int(vc.str.startswith("info consistent").sum())
        print(f"  disease {spec['disease']}: replace={n_rep} keep={n_keep} notes={len(vc)}")
        for rname in spec["regions"]:
            rcol = f"conflict5_note_knn_region_{slug_region(rname)}"
            rvc = s1[rcol].dropna().astype(str)
            print(
                f"    region {rname}: replace={int(rvc.str.startswith('replaced ').sum())} "
                f"keep={int(rvc.str.startswith('info consistent').sum())} notes={len(rvc)}"
            )

    tmp = xlsx.with_suffix(".xlsx.tmp_rex_c5knn")
    print(f"writing {tmp} …")
    with pd.ExcelWriter(tmp, engine="openpyxl") as w:
        for name, df in book.items():
            df.to_excel(w, sheet_name=str(name)[:31], index=False)
    tmp.replace(xlsx)
    print(f"done -> {xlsx} Sheet1 ncols={book['Sheet1'].shape[1]}")


if __name__ == "__main__":
    main()

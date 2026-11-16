"""Shared helpers for knn_norm_101 (unmodified) annotation from similarity tables."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from apply_knn_replacement_test_all import SHEET_CONFIG, norm_dis, norm_reg

SIM_FILES = {
    "EDEMA": "similarity_full_edema.xlsx",
    "CARDIOMEGALY": "similarity_full_cardiomegaly.xlsx",
    "Enlarged Cardiomediastinum": "similarity_full_enlarged_cardiomediastinum.xlsx",
    "Pleural effusion": "similarity_full_pleural effusion.xlsx",
    "Atelectasis": "similarity_full_atelectasis.xlsx",
}

REGION_SRC_COL = {
    ("EDEMA", "left hilar structures"): "left_knn_pred_norm_101",
    ("EDEMA", "right hilar structures"): "right_knn_pred_norm_101",
    ("CARDIOMEGALY", "cardiac silhouette"): "knn_pred_norm_101",
    ("Enlarged Cardiomediastinum", "mediastinum"): "med_knn_pred_norm_101",
    ("Enlarged Cardiomediastinum", "upper mediastinum"): "upper_knn_pred_norm_101",
    ("Pleural effusion", "left costophrenic angle"): "left_knn_pred_norm_101",
    ("Pleural effusion", "right costophrenic angle"): "right_knn_pred_norm_101",
    ("Atelectasis", "left lower lung zone"): "left_knn_pred_norm_101",
    ("Atelectasis", "right lower lung zone"): "right_knn_pred_norm_101",
}

DISEASE_SRC_COL = {
    "EDEMA": "pred_pattern_knn_norm_101",
    "CARDIOMEGALY": "pred_pattern_knn_norm_101",
    "Enlarged Cardiomediastinum": "pred_pattern_knn_norm_101",
    "Pleural effusion": "pleural effusion_knn_norm_101",
    "Atelectasis": "Atelectasis_knn_norm_101",
}

SHEET_KEY_BY_DISEASE = {
    "Edema": "EDEMA",
    "Cardiomegaly": "CARDIOMEGALY",
    "Enlarged Cardiomediastinum": "Enlarged Cardiomediastinum",
    "Atelectasis": "Atelectasis",
    "Pleural Effusion": "Pleural effusion",
}


def load_similarity_tables(sim_dir: Path) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for sheet_key, fname in SIM_FILES.items():
        fp = sim_dir / fname
        df = pd.read_excel(fp, sheet_name="merged_region_disease", engine="openpyxl")
        df["id"] = df["id"].astype(str)
        out[sheet_key] = df.set_index("id", drop=False)
    return out


def get_disease_val(row: pd.Series, sheet_key: str) -> Optional[str]:
    primary = DISEASE_SRC_COL[sheet_key]
    val = norm_dis(row.get(primary))
    if val is None and primary != "pred_pattern_knn_norm_101":
        val = norm_dis(row.get("pred_pattern_knn_norm_101"))
    return val


def fetch_knn_values(
    sim_tables: Dict[str, pd.DataFrame],
    tuple_id: str,
    disease: str,
) -> Tuple[Optional[str], List[Tuple[str, Optional[str]]]]:
    sheet_key = SHEET_KEY_BY_DISEASE[disease]
    cfg = SHEET_CONFIG[sheet_key]
    sim_df = sim_tables[sheet_key]
    if tuple_id not in sim_df.index:
        return None, []

    row = sim_df.loc[tuple_id]
    dis_val = get_disease_val(row, sheet_key)
    region_vals: List[Tuple[str, Optional[str]]] = []
    for region_name, _ in cfg["regions"]:
        r_src = REGION_SRC_COL[(sheet_key, region_name)]
        region_vals.append((region_name, norm_reg(row.get(r_src))))
    return dis_val, region_vals


def format_knn_txt_line(
    disease: str,
    dis_val: Optional[str],
    region_vals: List[Tuple[str, Optional[str]]],
    *,
    indent: str = "  ",
) -> str:
    """Format: knn：[Disease] : positive, [region]: normal"""
    dis_str = dis_val if dis_val is not None else "NA"
    parts = [f"[{disease}] : {dis_str}"]
    for region, reg_val in region_vals:
        parts.append(f"[{region}]: {reg_val if reg_val is not None else 'NA'}")
    return f"{indent}knn：" + ", ".join(parts)

"""
Apply knn_norm_101 region conflict updates from region_pair_conflict_similarity_results.xlsx
to test_all_KNNnorm101_clean_abnormal_merged_llm.xlsx (llm_check + llm_check_region_conflicts_knn).
"""

from __future__ import annotations

from paths import DATA_ROOT, KNN_ROOT

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from apply_region_conflict_mahalanobis_update import (
    DIS_COL,
    SRC_COL,
    format_change_log,
    get_protected_regions,
    pair_is_consistent,
    reg_to_label,
    safe_eval_list,
)
from apply_knn_replacement_test_all import find_item, norm_reg

DEFAULT_KNN_XLSX = KNN_ROOT / "region_pair_conflict_similarity_results.xlsx"
DEFAULT_SOURCE_XLSX = KNN_ROOT / "test_source.xlsx"
DEFAULT_OUT_XLSX = KNN_ROOT / "test_out.xlsx"
KNN_REGION_MAP = {
    "knn_norm_101_mediastinum": "mediastinum",
    "knn_norm_101_cardiac silhouette": "cardiac silhouette",
    "knn_norm_101_upper mediastinum": "upper mediastinum",
    "knn_norm_101_aortic arch": "aortic arch",
}
KNN_COLS = list(KNN_REGION_MAP.keys())
KNN_REGIONS = list(KNN_REGION_MAP.values())

OUT_COL = "pred_region_info_15w_knn_new_region_conflicts_knn"
OUT_CHANGE_LOG_COL = "region_conflict_knn_change_log"
MOD_COL = "region_conflict_knn_modified_regions"
SKIP_CONSISTENT_COL = "region_conflict_knn_skipped_consistent_regions"

LLM_CHECK_SHEET = "llm_check"
TEMPLATE_SHEET = "llm_check_with_region_conflicts"
OUT_SUBSET_SHEET = "llm_check_region_conflicts_knn"
STATS_SHEET = "region_conflict_knn_stats"

TEMPLATE_KEEP_COLS = [
    "id",
    "gpt_output_pred_15w_clean",
    "r2gen_report_clean",
    "pred_disease_info_chexber_uone_15w_knn_new",
    "llm_report_merge",
    "reference_report_new",
]
TEMPLATE_KNN_BLOCK = [
    OUT_COL,
    OUT_CHANGE_LOG_COL,
    MOD_COL,
    SKIP_CONSISTENT_COL,
] + KNN_COLS
TEMPLATE_TAIL_COLS = ["r2gen_report"]


def apply_knn_to_region_info(
    reg_info: List[Any],
    knn_row: Optional[pd.Series],
    *,
    skip_regions: Set[str],
) -> Tuple[List[Any], List[str], List[str], List[str]]:
    if not reg_info:
        return [], [], [], []

    out: List[Any] = [list(x) if isinstance(x, tuple) else x for x in reg_info]
    modified: List[str] = []
    change_logs: List[str] = []
    skipped: List[str] = []

    if knn_row is None:
        return out, modified, change_logs, skipped

    for knn_col, region_name in KNN_REGION_MAP.items():
        if knn_col not in knn_row.index:
            continue
        new_status = norm_reg(knn_row.get(knn_col))
        if new_status is None:
            continue

        item = find_item(out, region_name)
        if item is None:
            continue

        old_status = norm_reg(item[1])
        if old_status is None or old_status == new_status:
            continue

        if region_name in skip_regions:
            skipped.append(region_name)
            continue

        new_item = [region_name, reg_to_label(new_status)]
        for idx, existing in enumerate(out):
            if (
                isinstance(existing, (list, tuple))
                and len(existing) >= 2
                and str(existing[0]) == region_name
            ):
                out[idx] = new_item
                modified.append(region_name)
                change_logs.append(format_change_log(list(existing), new_item))
                break

    return out, modified, change_logs, skipped


def load_knn_lookup(xlsx_path: Path) -> pd.DataFrame:
    med = pd.read_excel(xlsx_path, sheet_name="knn_med_cs", engine="openpyxl")
    um = pd.read_excel(xlsx_path, sheet_name="knn_um_aa", engine="openpyxl")
    med["id"] = med["id"].astype(str)
    um["id"] = um["id"].astype(str)

    med_cols = ["id"] + [c for c in KNN_COLS if c in med.columns]
    um_cols = ["id"] + [c for c in KNN_COLS if c in um.columns]
    merged = med[med_cols].merge(um[um_cols], on="id", how="outer", suffixes=("", "_um"))
    for col in KNN_COLS:
        um_col = f"{col}_um"
        if col not in merged.columns and um_col in merged.columns:
            merged[col] = merged[um_col]
        elif col in merged.columns and um_col in merged.columns:
            merged[col] = merged[col].combine_first(merged[um_col])
            merged = merged.drop(columns=[um_col])
    keep = ["id"] + [c for c in KNN_COLS if c in merged.columns]
    return merged[keep].set_index("id")


def compute_row_updates(
    row: pd.Series,
    lookup: pd.DataFrame,
) -> Dict[str, Any]:
    reg_info = safe_eval_list(row.get(SRC_COL))
    knn_row = lookup.loc[row["id"]] if row["id"] in lookup.index else None
    protected = get_protected_regions(row, reg_info)

    updated, modified, logs, skipped = apply_knn_to_region_info(
        reg_info, knn_row, skip_regions=protected
    )

    return {
        OUT_COL: str(updated),
        OUT_CHANGE_LOG_COL: "；".join(logs) if logs else pd.NA,
        MOD_COL: str(modified) if modified else pd.NA,
        SKIP_CONSISTENT_COL: str(skipped) if skipped else pd.NA,
        **{
            col: knn_row.get(col) if knn_row is not None and col in knn_row.index else pd.NA
            for col in KNN_COLS
        },
        "_modified_count": len(modified),
        "_skipped_count": len(skipped),
    }


def append_knn_columns(df: pd.DataFrame, lookup: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if SRC_COL not in df.columns or DIS_COL not in df.columns:
        raise KeyError(f"Missing {SRC_COL} or {DIS_COL}")

    out = df.copy()
    out["id"] = out["id"].astype(str)

    drop_cols = [
        c
        for c in out.columns
        if c
        in [OUT_COL, OUT_CHANGE_LOG_COL, MOD_COL, SKIP_CONSISTENT_COL] + KNN_COLS
    ]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    extras = [compute_row_updates(row, lookup) for _, row in out.iterrows()]
    extra_df = pd.DataFrame(extras)

    new_block = extra_df[
        [OUT_COL, OUT_CHANGE_LOG_COL, MOD_COL, SKIP_CONSISTENT_COL] + KNN_COLS
    ]
    out = pd.concat([out.reset_index(drop=True), new_block], axis=1)

    stats_rows = [
        {
            "metric": "knn_updatable_rows",
            "category": "rows",
            "count": int(out[KNN_COLS[0]].notna().sum()),
        },
        {
            "metric": "rows_modified",
            "category": "rows",
            "count": int(out[MOD_COL].notna().sum()),
        },
        {
            "metric": "tuples_modified",
            "category": "tuples",
            "count": int(extra_df["_modified_count"].sum()),
        },
        {
            "metric": "consistent_tuples_skipped",
            "category": "tuples",
            "count": int(extra_df["_skipped_count"].sum()),
        },
    ]
    return out, pd.DataFrame(stats_rows)


def build_subset_sheet(
    llm_check: pd.DataFrame,
    lookup: pd.DataFrame,
) -> pd.DataFrame:
    knn_ids = lookup.reset_index()["id"].astype(str).tolist()
    base = llm_check.copy()
    base["id"] = base["id"].astype(str)
    base = base[base["id"].isin(knn_ids)].copy()

    ordered = []
    for col in TEMPLATE_KEEP_COLS:
        if col in base.columns:
            ordered.append(col)
    ordered.extend([c for c in TEMPLATE_KNN_BLOCK if c in base.columns])
    ordered.extend([c for c in TEMPLATE_TAIL_COLS if c in base.columns and c not in ordered])
    return base[ordered]


def process_workbook(
    source_xlsx: Path,
    knn_xlsx: Path,
    out_xlsx: Path,
) -> None:
    lookup = load_knn_lookup(knn_xlsx)
    xl = pd.ExcelFile(source_xlsx, engine="openpyxl")
    sheets = {name: pd.read_excel(xl, sheet_name=name) for name in xl.sheet_names}

    llm_check, stats = append_knn_columns(sheets[LLM_CHECK_SHEET], lookup)
    sheets[LLM_CHECK_SHEET] = llm_check
    sheets[STATS_SHEET] = stats
    sheets[OUT_SUBSET_SHEET] = build_subset_sheet(llm_check, lookup)

    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        for name, sdf in sheets.items():
            sdf.to_excel(writer, sheet_name=name[:31], index=False)

    print(f"Wrote: {out_xlsx}")
    print(f"llm_check rows: {len(llm_check)}")
    print(f"{OUT_COL} modified rows: {llm_check[MOD_COL].notna().sum()}")
    print(f"{OUT_SUBSET_SHEET} rows: {len(sheets[OUT_SUBSET_SHEET])}")
    for col in KNN_COLS:
        print(f"  {col} non-null in llm_check: {llm_check[col].notna().sum()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply knn_norm_101 region conflict updates.")
    parser.add_argument("--source-xlsx", type=Path, default=DEFAULT_SOURCE_XLSX)
    parser.add_argument("--knn-xlsx", type=Path, default=DEFAULT_KNN_XLSX)
    parser.add_argument("--out-xlsx", type=Path, default=DEFAULT_OUT_XLSX)
    args = parser.parse_args()
    process_workbook(args.source_xlsx, args.knn_xlsx, args.out_xlsx)


if __name__ == "__main__":
    main()

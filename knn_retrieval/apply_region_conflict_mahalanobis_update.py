"""
Update pred_region_info_15w_knn_new using mahalanobis predictions from
region_pair_conflict_similarity_results.xlsx (CSV or Excel sheet).

Produces:
  - pred_region_info_15w_knn_new_region_conflicts (+ change log)
  - pred_region_info_15w_knn_new_region_conflicts_new (skip disease-region consistent tuples)
"""

from __future__ import annotations

from paths import DATA_ROOT, KNN_ROOT

import argparse
import ast
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from apply_knn_replacement_test_all import SHEET_CONFIG, find_item, norm_dis, safe_eval_list

DEFAULT_MAHALANOBIS_XLSX = KNN_ROOT / "region_pair_conflict_similarity_results.xlsx"
DEFAULT_CSV = KNN_ROOT / "input.csv"
DEFAULT_TARGET_XLSX = KNN_ROOT / "test_source.xlsx"

MAHALANOBIS_REGION_MAP = {
    "mahalanobis_mediastinum": "mediastinum",
    "mahalanobis_cardiac silhouette": "cardiac silhouette",
    "mahalanobis_upper mediastinum": "upper mediastinum",
    "mahalanobis_aortic arch": "aortic arch",
}
MAHALANOBIS_COLS = list(MAHALANOBIS_REGION_MAP.keys())
MAHALANOBIS_REGIONS = list(MAHALANOBIS_REGION_MAP.values())

SRC_COL = "pred_region_info_15w_knn_new"
DIS_COL = "pred_disease_info_chexber_uone_15w_knn_new"
OUT_COL = "pred_region_info_15w_knn_new_region_conflicts"
OUT_CHANGE_LOG_COL = "region_conflict_mahalanobis_change_log"
MOD_COL = "region_conflict_mahalanobis_modified_regions"
HIT_CONSISTENT_COL = "region_conflict_mahalanobis_hit_consistent_regions"
OUT_NEW_COL = "pred_region_info_15w_knn_new_region_conflicts_new"
OUT_NEW_CHANGE_LOG_COL = "region_conflict_mahalanobis_change_log_new"
MOD_NEW_COL = "region_conflict_mahalanobis_modified_regions_new"
SKIP_CONSISTENT_COL = "region_conflict_mahalanobis_skipped_consistent_regions"

REGION_TO_DISEASE: Dict[str, str] = {}
for cfg in SHEET_CONFIG.values():
    for rname, _ in cfg["regions"]:
        REGION_TO_DISEASE[rname] = cfg["disease"]

OUTPUT_COL_ORDER = [
    OUT_COL,
    OUT_CHANGE_LOG_COL,
    MOD_COL,
    HIT_CONSISTENT_COL,
    OUT_NEW_COL,
    OUT_NEW_CHANGE_LOG_COL,
    MOD_NEW_COL,
    SKIP_CONSISTENT_COL,
] + MAHALANOBIS_COLS


def norm_reg(v: Any) -> Optional[str]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip().lower()
    if s == "abnormal":
        return "abnormal"
    if s == "normal":
        return "normal"
    return None


def reg_to_label(v: str) -> str:
    return "Abnormal" if v == "abnormal" else "Normal"


def tuple_repr(item: List[Any]) -> str:
    return str(list(item))


def format_change_log(old_item: List[Any], new_item: List[Any]) -> str:
    return f"将{tuple_repr(old_item)}修改为{tuple_repr(new_item)}"


def pair_is_consistent(d_item: Optional[List[Any]], r_item: Optional[List[Any]]) -> bool:
    if d_item is None or r_item is None:
        return False
    d = norm_dis(d_item[1])
    r = norm_reg(r_item[1])
    if d is None or r is None:
        return False
    return (d == "positive" and r == "abnormal") or (d == "negative" and r == "normal")


def get_protected_regions(row: pd.Series, reg_info: List[Any]) -> Set[str]:
    """Regions tied to a five-disease KNN mapping and already disease-region consistent."""
    protected: Set[str] = set()
    dis_info = safe_eval_list(row.get(DIS_COL))
    for region_name in MAHALANOBIS_REGIONS:
        disease = REGION_TO_DISEASE.get(region_name)
        if disease is None:
            continue
        d_item = find_item(dis_info, disease)
        r_item = find_item(reg_info, region_name)
        if pair_is_consistent(d_item, r_item):
            protected.add(region_name)
    return protected


def apply_mahalanobis_to_region_info(
    reg_info: List[Any],
    mah_row: Optional[pd.Series],
    *,
    skip_regions: Optional[Set[str]] = None,
) -> Tuple[List[Any], List[str], List[str], List[str]]:
    """
    Returns (updated_info, modified_regions, change_logs, skipped_consistent_regions).
    """
    if not reg_info:
        return [], [], [], []

    out: List[Any] = [list(x) if isinstance(x, tuple) else x for x in reg_info]
    modified: List[str] = []
    change_logs: List[str] = []
    skipped: List[str] = []
    skip_regions = skip_regions or set()

    if mah_row is None:
        return out, modified, change_logs, skipped

    for mah_col, region_name in MAHALANOBIS_REGION_MAP.items():
        if mah_col not in mah_row.index:
            continue
        new_status = norm_reg(mah_row.get(mah_col))
        if new_status is None:
            continue

        item = find_item(out, region_name)
        if item is None:
            continue

        old_status = norm_reg(item[1])
        if old_status is None:
            continue

        if old_status == new_status:
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


def load_mahalanobis_lookup(xlsx_path: Path) -> pd.DataFrame:
    med = pd.read_excel(xlsx_path, sheet_name="mahalanobis_med_cs", engine="openpyxl")
    um = pd.read_excel(xlsx_path, sheet_name="mahalanobis_um_aa", engine="openpyxl")
    med["id"] = med["id"].astype(str)
    um["id"] = um["id"].astype(str)

    med_cols = ["id"] + [c for c in MAHALANOBIS_COLS if c in med.columns]
    um_cols = ["id"] + [c for c in MAHALANOBIS_COLS if c in um.columns]
    merged = med[med_cols].merge(um[um_cols], on="id", how="outer", suffixes=("", "_um"))
    for col in MAHALANOBIS_COLS:
        um_col = f"{col}_um"
        if col not in merged.columns and um_col in merged.columns:
            merged[col] = merged[um_col]
        elif col in merged.columns and um_col in merged.columns:
            merged[col] = merged[col].combine_first(merged[um_col])
            merged = merged.drop(columns=[um_col])
    keep = ["id"] + [c for c in MAHALANOBIS_COLS if c in merged.columns]
    return merged[keep].set_index("id")


def reorder_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in OUTPUT_COL_ORDER if c in df.columns]
    if OUT_COL not in df.columns:
        return df
    anchor_idx = df.columns.get_loc(OUT_COL)
    before = [c for c in df.columns[:anchor_idx] if c not in present]
    after = [c for c in df.columns[anchor_idx + 1 :] if c not in present]
    ordered = before + present + after
    return df[ordered]


def apply_updates_to_dataframe(
    df: pd.DataFrame,
    lookup: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if SRC_COL not in df.columns:
        raise KeyError(f"Missing column: {SRC_COL}")
    if DIS_COL not in df.columns:
        raise KeyError(f"Missing column: {DIS_COL}")

    out = df.copy()
    out["id"] = out["id"].astype(str)

    full_infos: List[str] = []
    full_logs: List[object] = []
    full_modified: List[object] = []
    hit_consistent: List[object] = []

    new_infos: List[str] = []
    new_logs: List[object] = []
    new_modified: List[object] = []
    skipped_consistent: List[object] = []

    stats_rows: List[Dict[str, Any]] = []
    consistent_tuple_total = 0
    full_modified_total = 0
    full_hit_consistent_total = 0
    new_modified_total = 0
    full_hit_by_region: Counter[str] = Counter()
    consistent_by_region: Counter[str] = Counter()
    skipped_consistent_total = 0

    for _, row in out.iterrows():
        reg_info = safe_eval_list(row.get(SRC_COL))
        mah_row = lookup.loc[row["id"]] if row["id"] in lookup.index else None
        protected = get_protected_regions(row, reg_info)
        for region in protected:
            consistent_by_region[region] += 1
        consistent_tuple_total += len(protected)

        updated_full, modified_full, logs_full, _ = apply_mahalanobis_to_region_info(
            reg_info, mah_row
        )
        hit = [r for r in modified_full if r in protected]
        full_hit_by_region.update(hit)

        updated_new, modified_new, logs_new, skipped = apply_mahalanobis_to_region_info(
            reg_info, mah_row, skip_regions=protected
        )

        full_infos.append(str(updated_full))
        full_logs.append("；".join(logs_full) if logs_full else pd.NA)
        full_modified.append(str(modified_full) if modified_full else pd.NA)
        hit_consistent.append(str(hit) if hit else pd.NA)

        new_infos.append(str(updated_new))
        new_logs.append("；".join(logs_new) if logs_new else pd.NA)
        new_modified.append(str(modified_new) if modified_new else pd.NA)
        skipped_consistent.append(str(skipped) if skipped else pd.NA)

        full_modified_total += len(modified_full)
        full_hit_consistent_total += len(hit)
        new_modified_total += len(modified_new)
        skipped_consistent_total += len(skipped)

    out[OUT_COL] = full_infos
    out[OUT_CHANGE_LOG_COL] = full_logs
    out[MOD_COL] = full_modified
    out[HIT_CONSISTENT_COL] = hit_consistent
    out[OUT_NEW_COL] = new_infos
    out[OUT_NEW_CHANGE_LOG_COL] = new_logs
    out[MOD_NEW_COL] = new_modified
    out[SKIP_CONSISTENT_COL] = skipped_consistent

    for col in MAHALANOBIS_COLS:
        out[col] = out["id"].map(lookup[col]) if col in lookup.columns else pd.NA

    out = reorder_output_columns(out)

    stats_rows.extend(
        [
            {
                "metric": "mahalanobis可更新行数",
                "category": "rows",
                "count": int(out[MAHALANOBIS_COLS[0]].notna().sum()),
            },
            {
                "metric": "原始更新(全量)修改行数",
                "category": "rows",
                "count": int(out[MOD_COL].notna().sum()),
            },
            {
                "metric": "原始更新(全量)修改元组总数",
                "category": "tuples",
                "count": full_modified_total,
            },
            {
                "metric": "原始更新前已满足疾病-区域一致性的元组总数",
                "category": "tuples",
                "count": consistent_tuple_total,
            },
            {
                "metric": "原始更新涉及的一致元组数",
                "category": "tuples",
                "count": full_hit_consistent_total,
            },
            {
                "metric": "原始更新涉及的一致元组占全部一致元组比例(%)",
                "category": "pct",
                "count": round(
                    full_hit_consistent_total / consistent_tuple_total * 100, 2
                )
                if consistent_tuple_total
                else 0.0,
            },
            {
                "metric": "新逻辑修改行数",
                "category": "rows",
                "count": int(out[MOD_NEW_COL].notna().sum()),
            },
            {
                "metric": "新逻辑修改元组总数",
                "category": "tuples",
                "count": new_modified_total,
            },
            {
                "metric": "新逻辑跳过的一致元组数",
                "category": "tuples",
                "count": skipped_consistent_total,
            },
        ]
    )

    for region in MAHALANOBIS_REGIONS:
        stats_rows.append(
            {
                "metric": "原始更新前一致元组数(按区域)",
                "category": region,
                "count": consistent_by_region[region],
            }
        )
        stats_rows.append(
            {
                "metric": "原始更新涉及的一致元组数(按区域)",
                "category": region,
                "count": full_hit_by_region[region],
            }
        )

    stats = pd.DataFrame(stats_rows)
    return out, stats


def print_stats(stats: pd.DataFrame) -> None:
    print("\n=== 区域更新统计 ===")
    for _, row in stats.iterrows():
        print(f"  {row['metric']}: {row['count']} ({row['category']})")


def process_csv(
    csv_path: Path,
    mahalanobis_xlsx_path: Path,
    out_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    lookup = load_mahalanobis_lookup(mahalanobis_xlsx_path)
    df = pd.read_csv(csv_path, low_memory=False)
    df, stats = apply_updates_to_dataframe(df, lookup)

    out = out_path or csv_path
    df.to_csv(out, index=False)
    return df, stats


def process_xlsx_sheet(
    target_xlsx_path: Path,
    sheet_name: str,
    mahalanobis_xlsx_path: Path,
    out_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    lookup = load_mahalanobis_lookup(mahalanobis_xlsx_path)
    xl = pd.ExcelFile(target_xlsx_path, engine="openpyxl")
    sheet_names = xl.sheet_names

    if sheet_name not in sheet_names:
        raise ValueError(f"Sheet {sheet_name!r} not found in {target_xlsx_path}")

    updated, stats = apply_updates_to_dataframe(
        pd.read_excel(xl, sheet_name=sheet_name), lookup
    )

    out = out_path or target_xlsx_path
    if out == target_xlsx_path:
        sheets = {name: pd.read_excel(xl, sheet_name=name) for name in sheet_names}
        sheets[sheet_name] = updated
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            for name in sheet_names:
                sheets[name].to_excel(writer, sheet_name=name, index=False)
            stats.to_excel(writer, sheet_name="mahalanobis_region_stats", index=False)
    else:
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            updated.to_excel(writer, sheet_name=sheet_name, index=False)
            stats.to_excel(writer, sheet_name="mahalanobis_region_stats", index=False)

    return updated, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply mahalanobis region conflict updates to KNNnorm CSV/Excel."
    )
    parser.add_argument("--mahalanobis-xlsx", type=Path, default=DEFAULT_MAHALANOBIS_XLSX)
    parser.add_argument("--csv", type=Path, default=None, help="Target CSV path")
    parser.add_argument(
        "--target-xlsx",
        type=Path,
        default=None,
        help="Target Excel workbook path",
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default="llm_check",
        help="Sheet name when --target-xlsx is used",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.target_xlsx is not None:
        df, stats = process_xlsx_sheet(
            args.target_xlsx,
            args.sheet,
            args.mahalanobis_xlsx,
            args.out,
        )
        out_desc = args.out or args.target_xlsx
        print(f"Wrote sheet {args.sheet!r} to {out_desc}")
    elif args.csv is not None:
        df, stats = process_csv(args.csv, args.mahalanobis_xlsx, args.out)
        out_desc = args.out or args.csv
        print(f"Wrote {out_desc}")
    else:
        df, stats = process_csv(DEFAULT_CSV, args.mahalanobis_xlsx, args.out)
        out_desc = args.out or DEFAULT_CSV
        print(f"Wrote {out_desc}")

    print(f"Rows: {len(df)}")
    print(f"Rows with full region updates: {df[MOD_COL].notna().sum()}")
    print(f"Rows with selective region updates: {df[MOD_NEW_COL].notna().sum()}")
    for col in MAHALANOBIS_COLS:
        if col in df.columns:
            print(f"  {col} non-null: {df[col].notna().sum()}")
    print_stats(stats)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Unified evaluation entry for the paper metric groups.

Overall:              1/RadCliQ
Disease-level factual: F1-all 14, F1-conflict 5  (micro / macro)
Textual similarity:    BLEU-4, ROUGE-L, RadGraph F1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="SLICE EVAL (paper metric groups)")
    parser.add_argument("--input", required=True, help="CSV/XLSX with hyp & ref columns")
    parser.add_argument("--hyp_col", required=True)
    parser.add_argument("--ref_col", required=True)
    parser.add_argument("--outdir", default="./eval_out")
    parser.add_argument(
        "--groups",
        default="overall,factual,textual",
        help="Comma list: overall,factual,textual",
    )
    parser.add_argument("--nrows", type=int, default=None)
    args = parser.parse_args()

    path = Path(args.input)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    if args.nrows:
        df = df.head(args.nrows)

    refs = df[args.ref_col].fillna("").astype(str).tolist()
    hyps = df[args.hyp_col].fillna("").astype(str).tolist()
    groups = {g.strip() for g in args.groups.split(",") if g.strip()}
    out: dict = {"n": len(refs)}

    if "textual" in groups:
        from src.EVAL.textual_similarity.nlg import compute_nlg

        out.update(compute_nlg(refs, hyps))
        try:
            from src.EVAL.textual_similarity.radgraph_f1 import calc_radGraghF1

            out.update(calc_radGraghF1(hyps, refs))
        except Exception as e:  # optional heavy dep
            out["radgraph_error"] = str(e)

    if "factual" in groups:
        from src.EVAL.factual_accuracy.ce_metrics import compute_ce

        out.update(compute_ce(refs, hyps))

    if "overall" in groups:
        from src.EVAL.overall.radcliq_metric import compute_radcliq

        try:
            out.update(compute_radcliq(refs, hyps))
        except Exception as e:
            out["radcliq_error"] = str(e)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

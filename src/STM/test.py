"""Prompted transfer (DST) inference + optional metric computation."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.STM.eval_to_ts import predict_and_save_wchexbertinputprompt


def main():
    parser = argparse.ArgumentParser(description="Run DST inference on LLM draft reports")
    parser.add_argument("--input_csv", required=True, help="CSV/XLSX with LLM draft reports")
    parser.add_argument("--output_csv", required=True, help="Output CSV path")
    parser.add_argument("--checkpoint_dir", required=True, help="DST checkpoint directory")
    parser.add_argument("--llmpred_col", default="llm_report")
    parser.add_argument("--transfer_col", default="dst_report")
    parser.add_argument("--prompt_col", default=None, help="Optional precomputed prompt column")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    predict_and_save_wchexbertinputprompt(
        csv_input_path=args.input_csv,
        csv_output_path=args.output_csv,
        checkpoint_dir=args.checkpoint_dir,
        llmpred_col=args.llmpred_col,
        transfer_col=args.transfer_col,
        batch_size=args.batch_size,
        prompt_col=args.prompt_col,
    )


if __name__ == "__main__":
    main()

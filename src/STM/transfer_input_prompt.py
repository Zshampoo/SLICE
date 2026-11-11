"""Batch helper: run DST over multiple baseline LLM CSVs.

Configure via environment:
  DATA_ROOT, TRANSFER_READ_ROOT, TRANSFER_CHECKPOINT
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.STM.eval_to_ts import predict_and_save_wchexbertinputprompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--read_root", default=os.environ.get(
        "TRANSFER_READ_ROOT",
        str(Path(os.environ.get("DATA_ROOT", "./data")) / "transfer" / "baselines"),
    ))
    parser.add_argument("--checkpoint_dir", default=os.environ.get(
        "TRANSFER_CHECKPOINT",
        str(Path(os.environ.get("DATA_ROOT", "./data")) / "runs" / "transfer_dst" / "checkpoint"),
    ))
    parser.add_argument("--baselines", nargs="+", default=["R2Gen", "PromptMRG", "TGRG"])
    parser.add_argument("--input_type", default="pred15winfo")
    parser.add_argument("--llmpred_col_template", default="llm_infer_w{input_type}_{baseline}")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    read_root = Path(args.read_root)
    cp = Path(args.checkpoint_dir)
    for baseline in args.baselines:
        llm_readpath = read_root / baseline / f"llm_infer_w{args.input_type}_{baseline}.csv"
        out_dir = read_root / baseline / cp.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"Transfer_inputprompt_w{args.input_type}_{baseline}_cp_{cp.name}.csv"
        if not llm_readpath.is_file():
            print(f"skip missing {llm_readpath}")
            continue
        if out_path.is_file():
            print(f"skip existing {out_path}")
            continue
        col = args.llmpred_col_template.format(input_type=args.input_type, baseline=baseline)
        predict_and_save_wchexbertinputprompt(
            csv_input_path=str(llm_readpath),
            csv_output_path=str(out_path),
            checkpoint_dir=str(cp),
            llmpred_col=col,
            transfer_col=f"transfer_w{args.input_type}_{baseline}",
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()

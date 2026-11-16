"""Transfer (DST) training / inference configuration.

Override via environment variables or edit defaults below.
"""

from __future__ import annotations

import os
from pathlib import Path


class Config:
    # Prefer public HF id; optionally point MODEL_NAME to a local mirror.
    model_name = os.environ.get("MODEL_NAME", "facebook/bart-large")

    data_root = Path(os.environ.get("DATA_ROOT", "./data"))
    train_file = Path(os.environ.get("TRAIN_FILE", data_root / "transfer" / "train.csv"))
    eval_file = Path(os.environ.get("EVAL_FILE", data_root / "transfer" / "eval.csv"))
    test_file = Path(os.environ.get("TEST_FILE", data_root / "transfer" / "test.csv"))

    output_dir = Path(os.environ.get("OUTPUT_DIR", data_root / "runs" / "transfer_dst"))
    logging_dir = output_dir / "logs"

    # Prompted DST: use disease / region info columns as conditioning text.
    use_prompt = True
    prompt_disease_col = os.environ.get("PROMPT_DISEASE_COL", "pred_disease_info_knn")
    prompt_region_col = os.environ.get("PROMPT_REGION_COL", "pred_region_info_knn")
    input_report_col = os.environ.get("INPUT_REPORT_COL", "llm_report")
    target_report_col = os.environ.get("TARGET_REPORT_COL", "reference_report")

    max_source_length = 512
    max_target_length = 256
    per_device_train_batch_size = int(os.environ.get("BATCH_SIZE", "8"))
    learning_rate = float(os.environ.get("LR", "3e-5"))
    num_train_epochs = float(os.environ.get("EPOCHS", "3"))
    seed = 42

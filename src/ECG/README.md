# ECG — Evidence-guided Controllable Generator

BART-based controllable report generator conditioned on disease / region evidence and an input report (often the **LRO** refined text or a base **RG** report).

```bash
export PYTHONPATH=/path/to/SLICE
export MODEL_NAME=facebook/bart-large
export TRAIN_FILE=$DATA_ROOT/transfer/train.csv
export EVAL_FILE=$DATA_ROOT/transfer/eval.csv
export OUTPUT_DIR=$DATA_ROOT/runs/ecg

python -m src.ECG.train
python -m src.ECG.test \
  --input_csv $DATA_ROOT/transfer/test.csv \
  --output_csv $DATA_ROOT/transfer/ecg_out.csv \
  --checkpoint_dir $OUTPUT_DIR/checkpoint \
  --llmpred_col llm_report \
  --transfer_col dst_report
```

Prompt column names: `src/ECG/src/config.py` (`PROMPT_DISEASE_COL`, `PROMPT_REGION_COL`, `INPUT_REPORT_COL`).

Training / inference evaluation hooks call `src/EVAL` for CE / NLG / RadGraph.

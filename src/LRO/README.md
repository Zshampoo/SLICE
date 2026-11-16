# LRO — Prompted LLM Report Refinement

LRO refines a base report generator output (default: **R2Gen** under `src/RG/`) with disease / region prompts (from SIE + PCR).

```bash
export PYTHONPATH=/path/to/SLICE
export OPENAI_API_KEY=YOUR_KEY
export LLM_MODEL=gpt-4.1-mini   # optional

python -m src.LRO.LLM_inference \
  --input_csv $DATA_ROOT/reports/input.csv \
  --output_csv $DATA_ROOT/reports/lro_out.csv \
  --report_column baseline_report \
  --global_column pred_disease_info \
  --local_column pred_region_info \
  --result_column llm_report \
  --batch_size 10
```

Metrics for refined reports live under `src/EVAL/` (not in this folder).

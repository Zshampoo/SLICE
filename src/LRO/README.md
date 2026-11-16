# LRO — Prompted LLM Report Refinement

LRO refines a base report generator output (default: **R2Gen** under `src/RG/`) using disease / region evidence from **SIE** (+ optional **PCR** updates).

## Prompt types (`prompt_type`)

Default: **`PRED_NEWPROMPT_DISEASE_REGION`**.

| `prompt_type` | Evidence source | Evidence scope |
|---------------|-----------------|----------------|
| `PRED_NEWPROMPT_DISEASE_ONLY` | **pred** = SIE model predictions (usually with confidence) | disease only (`global_column`) |
| `PRED_NEWPROMPT_DISEASE_REGION` | **pred** = SIE predictions | disease + region (`global_column` + `local_column`) |
| `LABEL_NEWPROMPT_DISEASE_ONLY` | **label** = ground-truth annotations (no confidence) | disease only |
| `LABEL_NEWPROMPT_DISEASE_REGION` | **label** = ground truth | disease + region |

- **pred vs label**: `PRED_*` consumes SIE predicted disease/region info; `LABEL_*` consumes ground-truth disease/region labels.  
- **DISEASE_REGION vs DISEASE_ONLY**: `*_DISEASE_REGION` uses both disease and anatomical-region evidence; `*_DISEASE_ONLY` uses disease evidence only (region ignored / unused).

## Textualization

SIE raw outputs (class ids / scores) must be converted to the list-of-tuples format expected by `global_column` / `local_column` before calling LRO:

```python
from src.LRO.LLM_inference import Textualization

# pred: include confidence when available
global_col, local_col = Textualization(
    disease_preds=(disease_class_ids, disease_confidences),
    region_preds=(region_scores, region_confidences),
    source="pred",
    region_names=region_name_list,
)

# label: ground truth, no confidence
global_col, local_col = Textualization(
    disease_preds=disease_class_ids,
    region_preds=region_binary,
    source="label",
    region_names=region_name_list,
)
```

Returned shapes:

- `global_col`: `[[disease_name, status(, confidence)], ...]`
- `local_col`: `[[region_name, Abnormal|Normal(, confidence)], ...]` or `None` for disease-only prompts

## Run

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
  --prompt_type PRED_NEWPROMPT_DISEASE_REGION \
  --batch_size 10
```

Metrics for refined reports live under `src/EVAL/` (not in this folder).

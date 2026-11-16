# EVAL — Paper metric groups

Organized to match the evaluation table:

| Group | Metrics | Folder |
|-------|---------|--------|
| **Overall** | `1/RadCliQ` (higher better; raw RadCliQ-v1 lower better) | `overall/` |
| **Disease-level factual accuracy** | `F1-all 14` micro/macro; `F1-conflict 5` micro/macro | `factual_accuracy/` |
| **Textual similarity** | `BLEU-4`, `ROUGE-L`, `RadGraph F1` | `textual_similarity/` |

Conflict-5 diseases: Cardiomegaly, Atelectasis, Edema, Pleural Effusion, Lung Opacity.

## Quick run

```bash
export PYTHONPATH=/path/to/SLICE
export DATA_ROOT=/path/to/data
export CHEXBERT_WEIGHTS=$DATA_ROOT/weights/chexbert.pth
# optional: RADGRAPH_ARCHIVE, CXR_METRIC_ROOT (defaults to EVAL/third_party/CXR-Report-Metric)

python -m src.EVAL.run_eval \
  --input preds.xlsx \
  --hyp_col llm_report --ref_col reference_report_new \
  --outdir ./eval_out \
  --groups overall,factual,textual
```

## Module map

- `factual_accuracy/ce_metrics.py` — lightweight CE / F1-all / F1-conflict5  
- `factual_accuracy/CE_calc.py` — full CE + NLG helper used by ECG  
- `factual_accuracy/CheXbert/` — report labeler  
- `textual_similarity/nlg.py` — BLEU / ROUGE / METEOR / BertScore  
- `textual_similarity/radgraph_f1.py` / `radgraph_metric.py` — RadGraph F1  
- `overall/radcliq_metric.py` — RadCliQ-v1 & `1/RadCliQ` (needs `third_party/CXR-Report-Metric`)

Upstream RadCliQ reference: https://github.com/rajpurkarlab/CXR-Report-Metric

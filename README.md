# SLICE

Sanitized reference implementation accompanying the paper. Pipeline modules:

1. **SIE** — Image classifier training / testing  
2. **SIE** — Anatomical Region abnormality recognition (detector + region CLS) training / testing  
3. **PCR** — KNN retrieval & disease–region conflict resolution  
4. **LRO** — Prompted LLM report refinement  
5. **ECG** — Evidence-guided Controllable generator (BART-based) training / inference  

Default **report generator (RG)** is [R2Gen](https://github.com/zhjohnchan/R2Gen) (memory-driven Transformer), the representative baseline radiology report generation method used by SLICE.

All machine-specific absolute paths, API keys, and proxy endpoints have been removed. Configure data via environment variables.

---

## Layout

```text
SLICE/
├── README.md
├── requirements.txt
├── configs/paths.example.yaml
├── PCR/                           # (3) KNN + conflict resolution
└── src/
    ├── path_datasets_and_weights.py
    ├── dataset/                   # shared constants / splits helpers
    ├── SIE/                       # (1)(2) image + region
    │   ├── CheXpert_classifier/
    │   ├── object_detector/
    │   └── binary_classifier/
    ├── LRO/                       # (4) prompted LLM refinement
    ├── ECG/                       # (5) BART controllable generator
    ├── EVAL/                      # paper metric groups (Fig. metrics)
    │   ├── overall/               # 1/RadCliQ
    │   ├── factual_accuracy/      # F1-all 14, F1-conflict 5 (+ CheXbert)
    │   └── textual_similarity/    # BLEU-4, ROUGE-L, RadGraph F1
    └── RG/                        # R2Gen (default report generator)
```

```bash
cd SLICE
export PYTHONPATH=$PWD
export DATA_ROOT=/path/to/your/data
cp configs/paths.example.yaml configs/paths.yaml   # optional reference
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export DATA_ROOT=/path/to/your/data
export CHEXBERT_WEIGHTS=$DATA_ROOT/weights/chexbert.pth
```

LLM calls for **LRO** (never commit keys):

```bash
export OPENAI_API_KEY=YOUR_KEY
export OPENAI_BASE_URL=https://api.openai.com/v1   # optional
export LLM_MODEL=gpt-4.1-mini                      # optional
```

---

## 1–2) SIE — Image classifier & anatomical region abnormality

```bash
# Image classifier
python -m src.SIE.CheXpert_classifier.train_image_classifier
python -m src.SIE.CheXpert_classifier.test_image_classifier

# Detector + region CLS
python -m src.SIE.object_detector.training_script_object_detector
python -m src.SIE.object_detector.training_script_region_abnormality_classifier
python -m src.SIE.object_detector.test_script_region_abnormality
```

See [`src/SIE/README.md`](src/SIE/README.md).

---

## 3) PCR — KNN retrieval & conflict resolution

```bash
cd PCR
export KNN_ROOT=$DATA_ROOT/knn

python compute_disease_pattern_knn.py --help
python apply_knn_replacement_test_all.py --input-xlsx $KNN_ROOT/test.xlsx --output-xlsx $KNN_ROOT/test_knn.xlsx
python apply_region_conflict_knn_update.py --source-xlsx ... --knn-xlsx ... --out-xlsx ...
python apply_rexgradient_conflict5_knn.py --xlsx ...
python apply_chexpertplus_conflict5_info_replace.py --xlsx ...
```

See [`PCR/README.md`](PCR/README.md).

---

## 4) LRO — Prompted LLM report refinement

```bash
python -m src.LRO.LLM_inference \
  --input_csv $DATA_ROOT/reports/input.csv \
  --output_csv $DATA_ROOT/reports/lro_out.csv \
  --report_column baseline_report \
  --global_column pred_disease_info \
  --local_column pred_region_info \
  --result_column llm_report \
  --batch_size 10
```

See [`src/LRO/README.md`](src/LRO/README.md).

---

## 5) ECG — Evidence-guided Controllable generator

```bash
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

Prompt columns: `src/ECG/src/config.py` (`PROMPT_DISEASE_COL`, `PROMPT_REGION_COL`, `INPUT_REPORT_COL`).  
See [`src/ECG/README.md`](src/ECG/README.md).

---

## RG — Default report generator (R2Gen)

SLICE defaults to **R2Gen** as the base report generator. Code under `src/RG/` (vendored from the official repo).

- Upstream: https://github.com/zhjohnchan/R2Gen  
- Docs: [`src/RG/README.md`](src/RG/README.md)

---

## EVAL — Metrics (paper table groups)

| Group | Metrics | Code |
|-------|---------|------|
| Overall | `1/RadCliQ` | `src/EVAL/overall/` |
| Disease-level factual accuracy | `F1-all 14` (micro/macro), `F1-conflict 5` (micro/macro) | `src/EVAL/factual_accuracy/` |
| Textual similarity | `BLEU-4`, `ROUGE-L`, `RadGraph F1` | `src/EVAL/textual_similarity/` |

```bash
python -m src.EVAL.run_eval \
  --input $DATA_ROOT/reports/preds.xlsx \
  --hyp_col llm_report --ref_col reference_report_new \
  --outdir $DATA_ROOT/eval_out \
  --groups overall,factual,textual
```

See [`src/EVAL/README.md`](src/EVAL/README.md).

---

## Data & weights (not redistributed)

Place under `$DATA_ROOT` (see `configs/paths.example.yaml`).

### Dataset statistics & conflict-5 positive rates

|  | MIMIC-CXR |  |  |  | CheXpert Plus |  |  |  | ReXGradient |  |  |  |
|--|----------:|--:|--:|--:|--------------:|--:|--:|--:|------------:|--:|--:|--:|
|  | Train-SIE | Train-ECG | Val | Test | Train-SIE | Train-ECG | Val | Test | Train-SIE | Train-ECG | Val | Test |
| **# of samples** | 111,936 | 33,091 | 5,958 | 8,601 | 33,763 | 10,000 | 1,778 | 2,595 | 83,161 | 24,608 | 7,684 | 7,705 |
| **Avg. report length** | 49.23 | 49.09 | 48.49 | 49.34 | 70.33 | 69.63 | 70.96 | 69.98 | 51.92 | 52.04 | 50.81 | 51.28 |
| **Positive-label rate of conflict-5 (%)** |  |  |  |  |  |  |  |  |  |  |  |  |
| Cardiomegaly (CA) | 27.74 | 27.10 | 27.14 | 29.03 | 21.88 | 21.88 | 21.99 | 21.89 | 22.66 | 23.04 | 21.41 | 22.28 |
| Atelectasis (AT) | 27.90 | 27.35 | 27.73 | 29.92 | 31.13 | 31.11 | 31.16 | 31.14 | 19.40 | 19.21 | 18.49 | 19.01 |
| Edema (ED) | 13.99 | 13.61 | 14.62 | 14.24 | 28.67 | 28.67 | 28.63 | 28.67 | 10.98 | 11.02 | 10.09 | 11.01 |
| Pleural Effusion (PE) | 23.58 | 22.77 | 22.91 | 25.66 | 47.21 | 47.23 | 47.19 | 47.21 | 12.29 | 12.21 | 11.26 | 11.67 |
| Lung Opacity (LO) | 26.87 | 26.52 | 26.97 | 30.03 | 58.19 | 58.19 | 58.04 | 58.07 | 24.25 | 23.47 | 22.64 | 23.35 |

### Dataset download / access

| Dataset | Access |
|---------|--------|
| **MIMIC-CXR-JPG** | https://physionet.org/content/mimic-cxr-jpg/2.1.0/ |
| **Chest ImaGenome** | https://physionet.org/content/chest-imagenome/1.0.0/ |
| **CheXpert Plus** | https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9d3c3561a |
| **ReXGradient-160K** | https://huggingface.co/datasets/rajpurkarlab/ReXGradient-160K |

### Suggested local layout

| Item | Suggested path |
|------|----------------|
| MIMIC-CXR-JPG | `$DATA_ROOT/mimic-cxr-jpg` |
| Chest ImaGenome | `$DATA_ROOT/chest-imagenome` |
| CheXpert Plus | `$DATA_ROOT/chexpert-plus` |
| ReXGradient | `$DATA_ROOT/rexgradient` |
| Detector / classifier annotations | `$DATA_ROOT/annotations/...` |
| CheXbert weights | `$DATA_ROOT/weights/chexbert.pth` |
| BioViL / image encoder | `$DATA_ROOT/weights/biovil` |
| ECG / transfer CSV splits | `$DATA_ROOT/transfer/*.csv` |
| PCR / KNN tables & features | `$DATA_ROOT/knn/` |

Download CheXbert / BioViL / BART from their official sources. Checkpoints and private result dumps are not shipped.

---

## Sanitization notes

- Absolute workstation paths removed; use `DATA_ROOT` / CLI args.  
- API keys and lab proxy IPs removed.  
- Logs, checkpoints, and private result dumps are not included.  
- Experiment column names are kept so the code matches the paper pipeline; adapt as needed.

---

## Citation

Please cite the main paper if you use this code. Also cite [R2Gen](https://github.com/zhjohnchan/R2Gen) when using the default report generator.

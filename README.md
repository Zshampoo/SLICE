# SLICE

Sanitized reference implementation accompanying the paper. Includes:

1. Image classifier training / testing  
2. Region abnormality recognition (detector + region CLS)  
3. KNN retrieval & disease–region conflict resolution  
4. Prompted LLM report refinement  
5. Prompted transfer model (DST) training / inference  

All machine-specific absolute paths, API keys, and proxy endpoints have been removed. Configure data via environment variables.

---

## Layout

```text
SLICE/
├── README.md
├── requirements.txt
├── configs/paths.example.yaml
├── knn_retrieval/                 # standalone KNN / conflict scripts
└── src/
    ├── path_datasets_and_weights.py
    ├── dataset/constants.py
    ├── CheXpert_classifier/       # (1) image classifier
    ├── object_detector/           # (2) region detector / abnormality CLS
    ├── binary_classifier/
    ├── knn helpers live in ../../knn_retrieval
    ├── LLM/                       # (4) LLM inference + CE metrics
    ├── STM/                       # (5) DST train / test / prompts
    └── CheXbert/                  # report labeler used by CE / DST prompts
```

Set `PYTHONPATH` to the package root so `import src.*` resolves:

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

LLM calls (never commit keys):

```bash
export OPENAI_API_KEY=YOUR_KEY
export OPENAI_BASE_URL=https://api.openai.com/v1   # optional
export LLM_MODEL=gpt-4.1-mini                      # optional
```

---

## 1) Image classifier

```bash
python -m src.CheXpert_classifier.train_image_classifier
python -m src.CheXpert_classifier.test_image_classifier
```

Edit hyper-parameters near the top of the training script. Dataset / run directories come from `src/path_datasets_and_weights.py` (`DATA_ROOT`).

---

## 2) Region abnormality

```bash
python -m src.object_detector.training_script_object_detector
python -m src.object_detector.training_script_region_abnormality_classifier
python -m src.object_detector.test_script_region_abnormality
```

Produces anatomical boxes and Normal/Abnormal region labels used by conflict checking and prompting.

---

## 3) KNN retrieval & conflict resolution

```bash
cd knn_retrieval
export KNN_ROOT=$DATA_ROOT/knn   # inputs / outputs

python compute_disease_pattern_knn.py --help
python apply_knn_replacement_test_all.py --input-xlsx $KNN_ROOT/test.xlsx --output-xlsx $KNN_ROOT/test_knn.xlsx
python apply_region_conflict_knn_update.py --source-xlsx ... --knn-xlsx ... --out-xlsx ...
python apply_rexgradient_conflict5_knn.py --xlsx ...
python apply_chexpertplus_conflict5_info_replace.py --xlsx ...
```

`disease-region-mapping.txt` defines disease ↔ region pairs for conflict rules. Defaults read/write under `$KNN_ROOT`.

---

## 4) LLM inference

```bash
python -m src.LLM.LLM_inference \
  --input_csv $DATA_ROOT/reports/input.csv \
  --output_csv $DATA_ROOT/reports/llm_out.csv \
  --report_column baseline_report \
  --global_column pred_disease_info \
  --local_column pred_region_info \
  --result_column llm_report \
  --batch_size 10
```

Requires `OPENAI_API_KEY`. Optional CE / NLG helpers: `src/LLM/CE_calc.py`, `src/LLM/metrics.py`.

---

## 5) Prompted transfer (DST)

```bash
export MODEL_NAME=facebook/bart-large
export TRAIN_FILE=$DATA_ROOT/transfer/train.csv
export EVAL_FILE=$DATA_ROOT/transfer/eval.csv
export OUTPUT_DIR=$DATA_ROOT/runs/transfer_dst

python -m src.STM.train
python -m src.STM.test \
  --input_csv $DATA_ROOT/transfer/test.csv \
  --output_csv $DATA_ROOT/transfer/dst_out.csv \
  --checkpoint_dir $OUTPUT_DIR/checkpoint \
  --llmpred_col llm_report \
  --transfer_col dst_report
```

Prompt columns are configured in `src/STM/src/config.py` (`PROMPT_DISEASE_COL`, `PROMPT_REGION_COL`, `INPUT_REPORT_COL`).

---

## Data & weights (not redistributed)

Place under `$DATA_ROOT` (see `configs/paths.example.yaml`):

| Item | Suggested path |
|------|----------------|
| MIMIC-CXR-JPG | `$DATA_ROOT/mimic-cxr-jpg` |
| Detector / classifier annotations | `$DATA_ROOT/annotations/...` |
| CheXbert weights | `$DATA_ROOT/weights/chexbert.pth` |
| BioViL / image encoder | `$DATA_ROOT/weights/biovil` |
| Transfer CSV splits | `$DATA_ROOT/transfer/*.csv` |
| KNN tables / features | `$DATA_ROOT/knn/` |

Download CheXbert / BioViL / BART from their official sources.

---

## Sanitization notes

- Absolute workstation paths removed; use `DATA_ROOT` / CLI args.  
- API keys and lab proxy IPs removed.  
- Logs, checkpoints, and private result dumps are not included.  
- Experiment column names are kept so the code matches the paper pipeline; adapt as needed.

---

## Citation

Please cite the main paper if you use this code.

# SLICE

Sanitized reference implementation accompanying the paper.

**Task validation.** We validate SLICE on the **Radiology Report Generation (RRG)** task across three datasets (MIMIC-CXR, CheXpert Plus, and ReXGradient), covering disease/region evidence extraction (**SIE**), conflict resolution (**PCR**), prompted LLM refinement (**LRO**), and evidence-guided controllable generation (**ECG**), with **R2Gen** as the default base report generator.

Pipeline modules:

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
    │   ├── Disease_classifier/
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
python -m src.SIE.Disease_classifier.train_image_classifier
python -m src.SIE.Disease_classifier.test_image_classifier

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
  --prompt_type PRED_NEWPROMPT_DISEASE_REGION \
  --batch_size 10
```

Supported `prompt_type` values (default `PRED_NEWPROMPT_DISEASE_REGION`):  
`PRED_NEWPROMPT_DISEASE_ONLY` / `PRED_NEWPROMPT_DISEASE_REGION` / `LABEL_NEWPROMPT_DISEASE_ONLY` / `LABEL_NEWPROMPT_DISEASE_REGION`.  
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

<table>
<thead>
<tr>
<th></th>
<th colspan="4" align="center">MIMIC&#8209;CXR</th>
<th colspan="4" align="center">CheXpert&nbsp;Plus</th>
<th colspan="4" align="center">ReXGradient</th>
</tr>
<tr>
<th></th>
<th align="center">Train&#8209;SIE</th>
<th align="center">Train&#8209;ECG</th>
<th align="center">Val</th>
<th align="center">Test</th>
<th align="center">Train&#8209;SIE</th>
<th align="center">Train&#8209;ECG</th>
<th align="center">Val</th>
<th align="center">Test</th>
<th align="center">Train&#8209;SIE</th>
<th align="center">Train&#8209;ECG</th>
<th align="center">Val</th>
<th align="center">Test</th>
</tr>
</thead>
<tbody>
<tr>
<td><b>#&nbsp;of&nbsp;samples</b></td>
<td align="right">111,936</td><td align="right">33,091</td><td align="right">5,958</td><td align="right">8,601</td>
<td align="right">33,763</td><td align="right">10,000</td><td align="right">1,778</td><td align="right">2,595</td>
<td align="right">83,161</td><td align="right">24,608</td><td align="right">7,684</td><td align="right">7,705</td>
</tr>
<tr>
<td><b>Avg.&nbsp;report&nbsp;length</b></td>
<td align="right">49.23</td><td align="right">49.09</td><td align="right">48.49</td><td align="right">49.34</td>
<td align="right">70.33</td><td align="right">69.63</td><td align="right">70.96</td><td align="right">69.98</td>
<td align="right">51.92</td><td align="right">52.04</td><td align="right">50.81</td><td align="right">51.28</td>
</tr>
<tr>
<td colspan="13"><b>Positive&#8209;label&nbsp;rate&nbsp;of&nbsp;conflict&#8209;5&nbsp;(%)</b></td>
</tr>
<tr>
<td>Cardiomegaly&nbsp;(CA)</td>
<td align="right">27.74</td><td align="right">27.10</td><td align="right">27.14</td><td align="right">29.03</td>
<td align="right">21.88</td><td align="right">21.88</td><td align="right">21.99</td><td align="right">21.89</td>
<td align="right">22.66</td><td align="right">23.04</td><td align="right">21.41</td><td align="right">22.28</td>
</tr>
<tr>
<td>Atelectasis&nbsp;(AT)</td>
<td align="right">27.90</td><td align="right">27.35</td><td align="right">27.73</td><td align="right">29.92</td>
<td align="right">31.13</td><td align="right">31.11</td><td align="right">31.16</td><td align="right">31.14</td>
<td align="right">19.40</td><td align="right">19.21</td><td align="right">18.49</td><td align="right">19.01</td>
</tr>
<tr>
<td>Edema&nbsp;(ED)</td>
<td align="right">13.99</td><td align="right">13.61</td><td align="right">14.62</td><td align="right">14.24</td>
<td align="right">28.67</td><td align="right">28.67</td><td align="right">28.63</td><td align="right">28.67</td>
<td align="right">10.98</td><td align="right">11.02</td><td align="right">10.09</td><td align="right">11.01</td>
</tr>
<tr>
<td>Pleural&nbsp;Effusion&nbsp;(PE)</td>
<td align="right">23.58</td><td align="right">22.77</td><td align="right">22.91</td><td align="right">25.66</td>
<td align="right">47.21</td><td align="right">47.23</td><td align="right">47.19</td><td align="right">47.21</td>
<td align="right">12.29</td><td align="right">12.21</td><td align="right">11.26</td><td align="right">11.67</td>
</tr>
<tr>
<td>Lung&nbsp;Opacity&nbsp;(LO)</td>
<td align="right">26.87</td><td align="right">26.52</td><td align="right">26.97</td><td align="right">30.03</td>
<td align="right">58.19</td><td align="right">58.19</td><td align="right">58.04</td><td align="right">58.07</td>
<td align="right">24.25</td><td align="right">23.47</td><td align="right">22.64</td><td align="right">23.35</td>
</tr>
</tbody>
</table>

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

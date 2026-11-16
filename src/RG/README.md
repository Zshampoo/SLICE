# RG — Report Generator (default: R2Gen)

SLICE **defaults to R2Gen** as the base radiology report generator — a widely used, representative memory-driven Transformer baseline. Downstream **LRO** (prompted LLM refinement) and **ECG** (evidence-guided controllable generation) operate on top of RG outputs (or compatible report columns).

## Upstream

- Paper: *Generating Radiology Reports via Memory-driven Transformer* (EMNLP 2020)  
- GitHub: **https://github.com/zhjohnchan/R2Gen**

This folder contains the **pure R2Gen code** (train / test / modules / pycocoevalcap). Weights and datasets are **not** redistributed; obtain them from the official repo / PhysioNet.

## Train / test (IU X-Ray / MIMIC-CXR)

```bash
cd src/RG
bash train_iu_xray.sh
bash train_mimic_cxr.sh
bash test_iu_xray.sh
bash test_mimic_cxr.sh
```

Configure data paths inside the shell scripts to point at your local IU / MIMIC splits under `$DATA_ROOT`.

## Citation

Please cite the R2Gen paper when using this module:

```bibtex
@inproceedings{chen-emnlp-2020-r2gen,
    title = "Generating Radiology Reports via Memory-driven Transformer",
    author = "Chen, Zhihong and Song, Yan and Chang, Tsung-Hui and Wan, Xiang",
    booktitle = "Proceedings of EMNLP",
    year = "2020",
}
```

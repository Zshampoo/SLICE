# PCR — KNN retrieval & disease–region conflict resolution

PCR retrieves disease-pattern neighbors with **KNN only** and resolves disease–region conflicts before **LRO** / **ECG**. Non-KNN similarity methods (cosine / Mahalanobis) are not included.

```bash
export PYTHONPATH=/path/to/SLICE   # if importing shared utils
export DATA_ROOT=/path/to/data
export KNN_ROOT=$DATA_ROOT/knn
cd PCR

python compute_disease_pattern_knn.py --help
python apply_knn_replacement_test_all.py --input-xlsx $KNN_ROOT/test.xlsx --output-xlsx $KNN_ROOT/test_knn.xlsx
python apply_region_conflict_knn_update.py --source-xlsx ... --knn-xlsx ... --out-xlsx ...
python apply_rexgradient_conflict5_knn.py --xlsx ...
python apply_chexpertplus_conflict5_info_replace.py --xlsx ...
```

Core scripts:

- `compute_disease_pattern_knn.py` — FAISS KNN over disease-pattern train clusters
- `apply_knn_replacement_test_all.py` — apply KNN disease/region replacements
- `apply_region_conflict_knn_update.py` — region conflict updates from KNN predictions
- `apply_rexgradient_conflict5_knn.py` / `apply_chexpertplus_conflict5_info_replace.py` — conflict-5 KNN info replace
- `disease-region-mapping.txt` — disease ↔ region pairs for conflict rules
- `knn_unmodified_txt_utils.py` / `paths.py` — shared helpers

Defaults read/write under `$KNN_ROOT`.

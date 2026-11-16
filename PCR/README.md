# PCR — KNN retrieval & disease–region conflict resolution

PCR retrieves disease-pattern neighbors and resolves disease–region conflicts before **LRO** / **ECG**.

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

- `compute_disease_pattern_knn.py` / `compute_cardiomegaly_pattern_similarity.py`
- `apply_knn_replacement_test_all.py`
- `apply_region_conflict_knn_update.py` (+ Mahalanobis variant)
- `apply_rexgradient_conflict5_knn.py`
- `apply_chexpertplus_conflict5_info_replace.py`
- `disease-region-mapping.txt` — disease ↔ region pairs for conflict rules

Defaults read/write under `$KNN_ROOT`.

# KNN retrieval & conflict scripts

Run from this directory (or ensure it is on `PYTHONPATH`) so `from paths import ...` works.

```bash
export DATA_ROOT=/path/to/data
export KNN_ROOT=$DATA_ROOT/knn
python compute_disease_pattern_knn.py --help
```

Core scripts:
- `compute_disease_pattern_knn.py` / `compute_cardiomegaly_pattern_similarity.py`
- `apply_knn_replacement_test_all.py`
- `apply_region_conflict_knn_update.py` (+ `apply_region_conflict_mahalanobis_update.py`)
- `apply_rexgradient_conflict5_knn.py`
- `apply_chexpertplus_conflict5_info_replace.py`
- `disease-region-mapping.txt`

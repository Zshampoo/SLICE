"""Dataset / checkpoint path configuration.

Fill these paths (or export DATA_ROOT and keep relative layout) before running.
No machine-specific absolute paths are shipped in this supplementary package.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "./data"))

# Images / annotations
path_mimic_cxr_jpg = DATA_ROOT / "mimic-cxr-jpg"
path_chest_imagenome = DATA_ROOT / "chest-imagenome"
path_full_dataset = DATA_ROOT / "annotations" / "object_detector"

# Pretrained weights
path_chexbert_weights = DATA_ROOT / "weights" / "chexbert.pth"
path_biovil_weights = DATA_ROOT / "weights" / "biovil"

# Run outputs
path_runs_image_classifier = DATA_ROOT / "runs" / "image_classifier"
path_runs_object_detector = DATA_ROOT / "runs" / "object_detector"
path_runs_region_abnormality = DATA_ROOT / "runs" / "region_abnormality"
path_runs_transfer = DATA_ROOT / "runs" / "transfer_dst"
path_runs_full_model = DATA_ROOT / "runs" / "full_model"
path_test_set_evaluation_scores_txt_files = DATA_ROOT / "evals"

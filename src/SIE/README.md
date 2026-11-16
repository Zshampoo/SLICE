# SIE — Structured Image Evidence

SIE covers the vision-side stages of SLICE:

1. **Image / disease classifier** training / testing (`Disease_classifier/`)  
2. **Anatomical Region abnormality recognition** — detector + region CLS (`object_detector/`, `binary_classifier/`)

## Disease classifier

```bash
export PYTHONPATH=/path/to/SLICE
export DATA_ROOT=/path/to/data

python -m src.SIE.Disease_classifier.train_image_classifier
python -m src.SIE.Disease_classifier.test_image_classifier
```

Hyper-parameters live near the top of the training script. Dataset / run dirs come from `src/path_datasets_and_weights.py`.

## Anatomical region abnormality

```bash
python -m src.SIE.object_detector.training_script_object_detector
python -m src.SIE.object_detector.training_script_region_abnormality_classifier
python -m src.SIE.object_detector.test_script_region_abnormality
```

Produces anatomical boxes and Normal/Abnormal region labels used by **PCR** conflict checking and by **LRO** / **ECG** prompting.

`binary_classifier/` holds the region-abnormal head consumed by the detector stack.

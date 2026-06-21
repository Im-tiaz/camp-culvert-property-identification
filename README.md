# CAMP Culvert ML Pipeline

Unified inference pipeline for NMDOT CAMP quality control. Runs all property
models (Adam's + Hemanth's + future) over the same culvert images and produces
one combined CSV: one row per culvert, prediction + confidence per property.

## Folder Structure

```
camp_ml/
├── models/                  <- ALL model weight files go here
│   ├── silting_model_v1_2026_05_19.pt      (Hemanth - included)
│   ├── corrosion_model_v1_2026_05_19.pt    (Hemanth - included)
│   ├── bestMaterialClassificationNew.pt    (Adam - COPY FROM Adam_Culvert_Model/models/)
│   └── bestCulvertandShape.pt              (Adam - COPY FROM Adam_Culvert_Model/models/)
├── pipeline/
│   ├── model_registry.py    <- config: all models defined here
│   ├── run_inference.py     <- main script
│   └── check_models.py      <- verify models load
├── outputs/                 <- versioned result CSVs land here
├── datasets/                <- future: training data for new properties
└── requirements.txt
```

## Setup (one time)

```bash
pip install -r requirements.txt
```

Copy Adam's two model files into `models/`:
- `bestMaterialClassificationNew.pt`
- `bestCulvertandShape.pt`

Then verify everything loads:

```bash
cd pipeline
python check_models.py
```

Every model should report `LOADS OK`. For Adam's YOLO models, the checker also
prints the class names stored inside the weights - confirm they match
expectations (Material: Concrete/Corrugated Metal; Shape: Box/Circular).

## Run Inference

Point at your image folder (the structure download_all_images.py creates:
one subfolder per culvert, named by Culvert ID):

```bash
python run_inference.py --images /path/to/all_images
```

Options:
- `--models silting,corrosion` : run only specific models
- `--per-image`                : also write a per-image (non-aggregated) CSV
- `--output-dir /some/path`    : change output location

Output: `outputs/camp_predictions_v{N}_{date}.csv`

Example columns:

| CulvertID | NumImages | Silting_Prediction | Silting_Confidence | Corrosion_Prediction | ... | Shape_Prediction | NumberOfCulverts_Prediction |

## Multi-Photo Aggregation Rules

When a culvert has several photos:
- **Defect properties** (silting, corrosion): worst case wins - if ANY photo
  shows the worst class, the culvert is flagged with it (conservative for QC).
- **Categorical properties** (material, shape): majority vote across photos,
  highest confidence breaks ties.
- **Culvert count**: maximum across photos (clearest photo sees all barrels).

## Adding a New Property Model (Phase 2)

1. Build dataset folders (`train/valid/test`, one subfolder per class) using
   labels from `nmdot_table_fixed.csv` joined to images by Culvert ID.
2. Train with Hemanth's `train.ipynb` (set TASK/classes) -> TorchScript `.pt`.
3. Drop the `.pt` into `models/`.
4. Add one entry to `MODEL_REGISTRY` in `pipeline/model_registry.py`
   (a commented template is provided at the bottom of the file).
5. `python check_models.py`, then run inference as usual.

No pipeline code changes needed.

## IMPORTANT - Verify Hemanth's Class Labels

The registry assumes Hemanth's silting classes are
`0=Clean/Minor, 1=Moderate, 2=Major` and corrosion `0=None/Minor, 1=Corrosion`,
based on his ImageFolder training (alphabetical folder names '0','1','2').
**Confirm against his dataset folders** (`dataset_v5`, `dataset_v10`) and edit
the `classes` / `severity_order` entries in `model_registry.py` if different.

# CAMP Culvert ML — Quality-Control Pipeline

Vision-based quality control for culvert inventory data: predicts ten physical
properties from field inspection photographs and flags disagreements with
inspector records for review.

See the accompanying paper for methods and results.

## Components
- `model_registry.py` — central per-property model configuration
- `build_dataset.py` — builds field-representative datasets from the inventory
- `retrain.py` — two-phase EfficientNetV2-S training
- `evaluate_model.py`, `threshold_sweep.py`, `inspect_dataset.py` — evaluation
- `run_inference.py`, `dimension_check.py`, `merge_results.py` — the pipeline
- `camp_app.py` — browser interface

## Note on data
The CAMP inventory and inspection photographs are not included in this
repository. See the paper's reproducibility appendix.

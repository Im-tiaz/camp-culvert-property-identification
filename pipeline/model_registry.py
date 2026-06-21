"""
============================================================================
CAMP Culvert ML Pipeline - MODEL REGISTRY
============================================================================
Central configuration for ALL property models.

To add a new property model in the future:
  1. Train it (e.g., with Hemanth's train.ipynb -> TorchScript .pt)
  2. Drop the .pt file into the models/ folder
  3. Add ONE entry to MODEL_REGISTRY below
  That's it. run_inference.py picks it up automatically.

Two loader types are supported:
  - "torchscript" : Hemanth-style EfficientNet TorchScript classifiers
  - "yolo"        : Adam-style ultralytics YOLO models
                    (detection -> shape + count, classification -> material)
============================================================================
"""

import os
from glob import glob

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PIPELINE_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")


def newest(pattern):
    """
    Resolve a model file by glob pattern, returning the NEWEST match.
    Lets the registry reference e.g. 'silting_binary_model_unfrozen_*.pt'
    without hard-coding the date stamp - retraining drops in a new dated
    file and the pipeline picks it up automatically.

    Falls back to the literal pattern (no wildcard) if nothing matches yet,
    so check_models.py reports it as MISSING rather than crashing.
    """
    full = os.path.join(MODELS_DIR, pattern)
    matches = sorted(glob(full), key=os.path.getmtime, reverse=True)
    return matches[0] if matches else full

# ---------------------------------------------------------------------------
# MODEL REGISTRY
# ---------------------------------------------------------------------------
# Each entry:
#   property        : column name prefix in the output CSV
#   model_path      : path to weights file
#   loader          : "torchscript" | "yolo_classify" | "yolo_detect"
#   classes         : index -> human-readable label
#   aggregation     : how to combine multiple photos of the SAME culvert
#       "worst_case"    -> severity order applies (defects: report the worst)
#       "majority"      -> categorical vote, confidence breaks ties (material/shape)
#       "max"           -> numeric maximum (culvert count)
#   severity_order  : ONLY for worst_case. Labels from least to most severe.
#   enabled         : set False to skip a model without deleting its entry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {

    # ----------------------------------------------------------------
    # HEMANTH'S MODELS (TorchScript EfficientNet-B0, verified working)
    # ----------------------------------------------------------------
    "silting": {
        "property": "Silting",
        "model_path": newest("silting_binary_model_unfrozen_*.pt"),
        "loader": "torchscript",
        "input_size": 224,
        # Binary silting (field-trained, GPU-unfrozen). Folder convention:
        #   0 = Clean/Minor (<60%), 1 = Major (>=60%)  [handbook "major" = >60%]
        "classes": {0: "Clean/Minor (<60%)", 1: "Major (>=60%)"},
        "aggregation": "worst_case",
        "severity_order": ["Clean/Minor (<60%)", "Major (>=60%)"],
        # Only flag "Major" when the model is at least this confident.
        # Raises precision at some cost to recall. See run_inference --help.
        # Tuned value lives here so the whole pipeline uses it consistently.
        "min_confidence": {"Major (>=60%)": 0.50},  # keep at 0.50: silting is a
        # flooding risk, so favor recall (catch more) over precision; F1 also
        # peaks here. Raising it only trades recall away ~1:1 (see threshold_sweep).
        "enabled": True,
    },

    "corrosion": {
        "property": "Corrosion",
        "model_path": newest("corrosion_model_unfrozen_*.pt"),
        "loader": "torchscript",
        "input_size": 224,
        # Confirmed by eye against train folders: 0 = no corrosion, 1 = corrosion
        "classes": {0: "None/Minor", 1: "Corrosion"},
        "aggregation": "worst_case",
        "severity_order": ["None/Minor", "Corrosion"],
        "min_confidence": {"Corrosion": 0.65},  # sweep: best F1, halves false alarms
        "enabled": True,
    },

    # ----------------------------------------------------------------
    # ADAM'S MODELS (YOLO)
    # Copy these files from Adam_Culvert_Model/models/ into models/
    # ----------------------------------------------------------------
    "material": {
        "property": "Material",
        "model_path": os.path.join(MODELS_DIR, "bestMaterialClassificationNew.pt"),
        "loader": "yolo_classify",
        "classes": None,  # YOLO stores class names inside the weights; read at load time
        "aggregation": "majority",
        "enabled": True,
    },

    "shape_count": {
        "property": "Shape",          # produces Shape + NumberOfCulverts columns
        "model_path": os.path.join(MODELS_DIR, "bestCulvertandShape.pt"),
        "loader": "yolo_detect",
        "classes": None,              # read from weights (expected: Box, Circular)
        "confidence_threshold": 0.6,  # per Adam's thesis F1-confidence analysis
        "aggregation": "majority",    # shape: majority vote; count: max across photos
        "enabled": True,
    },

    # ----------------------------------------------------------------
    # PHASE 2 PROPERTIES (field-trained, GPU)
    # ----------------------------------------------------------------
    "end_section": {
        "property": "EndSection",
        "model_path": newest("end_section_4class_model_field_*.pt"),
        "loader": "torchscript",
        "input_size": 224,
        # Trained on inlet+outlet labels combined (an end section looks the
        # same whichever end it is). 4 classes - CBC dropped: it had only ~48
        # training images and its huge class weight was hurting the others.
        # Dropping it raised macro-F1 0.64 -> 0.72 and lifted every other class.
        "classes": {0: "None/Projecting", 1: "Metal/Concrete Flared",
                    2: "Headwall", 3: "Slope Blanket"},
        "aggregation": "majority",
        # NOTE: because it was trained on both inlet & outlet, merge_results.py
        # scores a prediction as correct if it matches EITHER the inlet or the
        # outlet inspector value (see end_section either_cols there). Inventory
        # CBC culverts now score UNSUPPORTED_VALUE (model can't predict CBC).
        "enabled": True,
    },

    "physical_damage": {
        "property": "PhysicalDamage",
        "model_path": newest("physical_damage_model_field_*.pt"),
        "loader": "torchscript",
        "input_size": 224,
        # Binary damage screen. Severity tiers (minor/moderate/severe) did not
        # separate visually (4-tier macro-F1 0.36); collapsing to damage/none
        # gave a coherent screen (Damage F1 0.61 - strongest defect screen).
        #   0 = No Damage, 1 = Damage
        "classes": {0: "No Damage", 1: "Damage"},
        "aggregation": "worst_case",
        "severity_order": ["No Damage", "Damage"],
        "min_confidence": {"Damage": 0.50},  # default; tune later via threshold_sweep
        "enabled": True,
    },

    "channel_condition": {
        "property": "ChannelCondition",
        "model_path": newest("channel_condition_model_field_*.pt"),
        "loader": "torchscript",
        "input_size": 224,
        # 2-class (Degrading dropped: only 16 examples, destabilized training).
        # Whole-channel descriptive property -> majority vote across photos.
        #   0 = Good, 1 = Weeds/Debris/Vegetated
        "classes": {0: "Good", 1: "Weeds/Debris/Vegetated"},
        "aggregation": "majority",
        "enabled": True,
    },

    "erosion_control": {
        "property": "ErosionControl",
        "model_path": newest("erosion_control_model_field_*.pt"),
        "loader": "torchscript",
        "input_size": 224,
        # Binary presence screen. Rip-rap/structures are visually distinctive
        # (Control Present F1 0.49 despite 9.7x imbalance). Blank inventory =
        # none. Outlet feature, but outlet photo can't be isolated at scale
        # (~9% have Inspection Direction), so trained on all photos.
        #   0 = No Control, 1 = Control Present
        # NB: label is "No Control" (not "None") because pandas reads the literal
        # string "None" as NaN, which broke the merge comparison.
        "classes": {0: "No Control", 1: "Control Present"},
        "aggregation": "worst_case",
        "severity_order": ["No Control", "Control Present"],
        "min_confidence": {"Control Present": 0.50},
        "enabled": True,
    },

    "channel_type": {
        "property": "ChannelType",
        "model_path": newest("channel_type_model_field_*.pt"),
        "loader": "torchscript",
        "input_size": 224,
        # 4 classes. Dry Arroyo + Roadside/Median Ditch MERGED (visually
        # identical; ditch is a location call, not an appearance one). Concrete/
        # Asphalt Lined dropped (only ~8 examples). Running Water and Irrigation
        # kept but WEAK (F1 ~0.17 / ~0.07) - too few examples so far; treat their
        # predictions as provisional. More data planned -> just retrain.
        "classes": {0: "Dry/Ephemeral", 1: "No Channel",
                    2: "Running Water", 3: "Irrigation"},
        "aggregation": "majority",
        "enabled": True,
    },

    # Scour: trained but NOT deployed. Scour Present reached only F1 0.34
    # (precision 0.32, recall 0.37) - the weakest defect class in the project.
    # Scour is both subtle (channel-bed erosion) and hurt by inlet-photo label
    # noise (outlet can't be isolated). Kept disabled; revisit if outlet-tagged
    # photos or more scour examples become available.
    "scour": {
        "property": "Scour",
        "model_path": newest("scour_model_field_*.pt"),
        "loader": "torchscript",
        "input_size": 224,
        "classes": {0: "Little/No Scour", 1: "Scour Present"},
        "aggregation": "worst_case",
        "severity_order": ["Little/No Scour", "Scour Present"],
        "min_confidence": {"Scour Present": 0.50},
        "enabled": False,   # too weak to deploy (F1 0.34)
    },
}

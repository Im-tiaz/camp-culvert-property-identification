"""
============================================================================
CAMP Culvert ML Pipeline - MODEL CHECKER
============================================================================
Verifies every model in the registry: file exists, loads, and runs a dummy
prediction. Run this after adding any new model file.

USAGE:
    python check_models.py
============================================================================
"""

import os
import tempfile
import numpy as np
import torch
from PIL import Image

from model_registry import MODEL_REGISTRY
from run_inference import LOADERS


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # create one dummy image to test predictions (cross-platform temp dir)
    dummy_path = os.path.join(tempfile.gettempdir(), "_camp_dummy.jpg")
    Image.fromarray(
        (np.random.rand(480, 640, 3) * 255).astype("uint8")
    ).save(dummy_path)

    for key, cfg in MODEL_REGISTRY.items():
        status = []
        print(f"[{key}]  property={cfg['property']}  loader={cfg['loader']}")
        print(f"   file: {cfg['model_path']}")

        if not cfg.get("enabled", True):
            print("   -> DISABLED in registry (skipped)\n")
            continue

        if not os.path.exists(cfg["model_path"]):
            print("   -> MISSING: copy the .pt file into the models/ folder\n")
            continue

        try:
            model = LOADERS[cfg["loader"]](cfg, device)
            out = model.predict(dummy_path)
            print(f"   -> LOADS OK. Dummy prediction: {out}")
            if cfg["loader"].startswith("yolo"):
                print(f"   -> class names from weights: {model.classes}")
            del model
        except ImportError as e:
            print(f"   -> MISSING DEPENDENCY: {e}")
            print("      (YOLO models need: pip install ultralytics)")
        except Exception as e:
            print(f"   -> FAILED: {type(e).__name__}: {e}")
        print()

    os.remove(dummy_path)


if __name__ == "__main__":
    main()

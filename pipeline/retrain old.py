"""
============================================================================
CAMP Culvert ML Pipeline - RETRAIN
============================================================================
Retrains a silting or corrosion classifier using the architecture from
Hemanth's published paper (FMLDS 2025):

    EfficientNetV2-Small (ImageNet pretrained, FROZEN backbone)
      -> custom head: Linear -> BatchNorm -> ReLU -> Linear

This is the configuration that achieved 92.33% in the paper - NOT the
plain fine-tuned EfficientNet-B0 in the old train.ipynb (whose deployed
weights collapsed on real field images).

Key additions for real-world robustness:
  - class weighting              (fixes the imbalance that broke corrosion)
  - light data augmentation      (flip, rotation, color jitter)
  - early stopping on val loss   (prevents the overfitting Adam flagged)
  - TorchScript export           (drops into the pipeline registry directly)

USAGE:
    python retrain.py --data "C:\\path\\to\\silting"   --task silting
    python retrain.py --data "C:\\path\\to\\corrosion" --task corrosion

Optional:
    --epochs 50  --batch-size 32  --lr 1e-3  --unfreeze   (fine-tune backbone)

Output:
    models/{task}_model_v2_{date}.pt           (TorchScript, ready for pipeline)
    models/{task}_classmap_v2_{date}.json      (index -> folder name, for registry)
============================================================================
"""

import argparse
import json
import os
from collections import Counter
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224


# ---------------------------------------------------------------------------
# Model: EfficientNetV2-S backbone + paper's custom classification head
# ---------------------------------------------------------------------------
class CulvertClassifier(nn.Module):
    def __init__(self, num_classes, freeze_backbone=True):
        super().__init__()
        weights = EfficientNet_V2_S_Weights.DEFAULT
        self.backbone = efficientnet_v2_s(weights=weights)

        # number of features going into the original classifier
        in_features = self.backbone.classifier[1].in_features

        # freeze backbone feature extractor (transfer learning, as in paper)
        if freeze_backbone:
            for p in self.backbone.features.parameters():
                p.requires_grad = False

        # replace classifier with the paper's custom head:
        # FC -> BatchNorm -> ReLU -> FC
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def build_loaders(data_dir, batch_size):
    train_tf = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_name = "valid" if os.path.isdir(os.path.join(data_dir, "valid")) else "val"
    train_ds = ImageFolder(os.path.join(data_dir, "train"), train_tf)
    val_ds = ImageFolder(os.path.join(data_dir, val_name), eval_tf)
    test_dir = os.path.join(data_dir, "test")
    test_ds = ImageFolder(test_dir, eval_tf) if os.path.isdir(test_dir) else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = (DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
                   if test_ds else None)

    return train_ds, train_loader, val_loader, test_loader


def class_weights(train_ds, device):
    """Inverse-frequency weights to counter class imbalance."""
    counts = Counter(train_ds.targets)
    n = len(train_ds.targets)
    k = len(counts)
    w = [n / (k * counts[i]) for i in range(k)]
    return torch.tensor(w, dtype=torch.float, device=device)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    per_class_correct = Counter()
    per_class_total = Counter()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        for t, p in zip(y.tolist(), pred.tolist()):
            per_class_total[t] += 1
            if t == p:
                per_class_correct[t] += 1
    acc = correct / max(total, 1)
    per_class = {c: per_class_correct[c] / max(per_class_total[c], 1)
                 for c in sorted(per_class_total)}
    return acc, per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--task", required=True, choices=["silting", "corrosion"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--unfreeze", action="store_true",
                    help="also fine-tune the backbone (slower, needs more data)")
    ap.add_argument("--patience", type=int, default=7,
                    help="early-stopping patience on val loss")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if getattr(torch.backends, "mps", None)
              and torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    if device == "cpu":
        print("  (CPU training will be slow; a GPU is strongly recommended)")

    train_ds, train_loader, val_loader, test_loader = build_loaders(
        args.data, args.batch_size)
    num_classes = len(train_ds.classes)
    print(f"Task: {args.task} | classes (folder order): {train_ds.classes}")
    print(f"Train: {len(train_ds)} images")

    model = CulvertClassifier(num_classes, freeze_backbone=not args.unfreeze).to(device)
    weights = class_weights(train_ds, device)
    print(f"Class weights (imbalance correction): {weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=weights)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)

        # validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        val_acc, _ = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:3d} | train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | val_acc {val_acc*100:.1f}%")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (no val improvement).")
                break

    # restore best weights
    if best_state:
        model.load_state_dict(best_state)

    # final evaluation
    print("\n=== Final evaluation (best model) ===")
    val_acc, val_pc = evaluate(model, val_loader, device)
    print(f"Validation accuracy: {val_acc*100:.2f}%  per-class: "
          + ", ".join(f"{train_ds.classes[c]}={a*100:.0f}%" for c, a in val_pc.items()))
    if test_loader:
        test_acc, test_pc = evaluate(model, test_loader, device)
        print(f"Test accuracy:       {test_acc*100:.2f}%  per-class: "
              + ", ".join(f"{train_ds.classes[c]}={a*100:.0f}%" for c, a in test_pc.items()))

    # ---- export TorchScript + class map ----
    models_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "models")
    os.makedirs(models_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y_%m_%d")

    model.eval().cpu()
    example = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    scripted = torch.jit.trace(model, example)
    out_pt = os.path.join(models_dir, f"{args.task}_model_v2_{stamp}.pt")
    scripted.save(out_pt)

    classmap = {i: train_ds.classes[i] for i in range(num_classes)}
    out_json = os.path.join(models_dir, f"{args.task}_classmap_v2_{stamp}.json")
    with open(out_json, "w") as f:
        json.dump(classmap, f, indent=2)

    print(f"\nSaved model    -> {out_pt}")
    print(f"Saved classmap -> {out_json}")
    print(f"Class index -> folder name: {classmap}")
    print("\nNext: update model_registry.py to point at this new .pt and set")
    print("the 'classes' dict using the folder->meaning mapping you confirm.")


if __name__ == "__main__":
    main()

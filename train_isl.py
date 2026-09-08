"""
Training script for the ISL (Indian Sign Language) model.
Dataset: indian sign language/Indian/ — 35 classes (A-Z + 1-9)
~1,221 images per class — large balanced dataset, lighter augmentation needed.
GPU training via CUDA.
"""
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from model import build_model, NORMALIZE_MEAN, NORMALIZE_STD, INPUT_SIZE

# ─── Config ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
DATASET_DIR  = str(BASE_DIR / "indian sign language" / "Indian")
MODEL_SAVE   = str(BASE_DIR / "models" / "isl_model.pth")
NUM_EPOCHS   = int(os.environ.get("EPOCHS", 10))
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", 64))
LR           = 1e-3
VAL_SPLIT    = 0.15
NUM_WORKERS  = int(os.environ.get("NUM_WORKERS", 4))
# Frame-adjacency leakage control: images are consecutive video frames per class,
# so whole contiguous BLOCKS go to train or val -- never shuffled per image.
# Must match train_isl_augmented_spatial.py's BLOCK_SIZE.
BLOCK_SIZE   = int(os.environ.get("BLOCK_SIZE", 100))
SPLIT_SEED   = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Transforms ────────────────────────────────────────────────────────────────
# Lighter augmentation — dataset is already large enough
train_transforms = transforms.Compose([
    transforms.Resize((INPUT_SIZE + 16, INPUT_SIZE + 16)),
    transforms.RandomCrop(INPUT_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
])

val_transforms = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
])


def frame_number(filename):
    """Leading integer of the filename == temporal frame index.

    sorted() is lexicographic ('0','1','10','100'), which is NOT frame order.
    '217 copy.jpg' maps to 217 so it shares its frame's block.
    """
    m = re.match(r"(\d+)", os.path.splitext(filename)[0])
    return int(m.group(1)) if m else -1


class ISLFrameDataset(Dataset):
    """ImageFolder-equivalent built from an explicit file list."""

    def __init__(self, samples, class_to_idx, transform):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, target = self.samples[i]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return self.transform(img), target


def build_block_split():
    """Splits each class's frames into contiguous blocks; whole blocks go to
    train or val, so no near-duplicate neighbour straddles the boundary."""
    classes = sorted(d for d in os.listdir(DATASET_DIR)
                     if os.path.isdir(os.path.join(DATASET_DIR, d)))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    rng = np.random.default_rng(SPLIT_SEED)

    train_samples, val_samples = [], []
    for cls in classes:
        d = os.path.join(DATASET_DIR, cls)
        files = [f for f in os.listdir(d)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        files.sort(key=frame_number)          # true temporal order

        blocks = sorted({frame_number(f) // BLOCK_SIZE for f in files})
        n_val_blocks = max(1, int(round(len(blocks) * VAL_SPLIT)))
        val_blocks = set(rng.permutation(blocks)[:n_val_blocks].tolist())

        for f in files:
            item = (os.path.join(d, f), class_to_idx[cls])
            if frame_number(f) // BLOCK_SIZE in val_blocks:
                val_samples.append(item)
            else:
                train_samples.append(item)

    return train_samples, val_samples, classes, class_to_idx


def get_dataloaders():
    if not os.path.isdir(DATASET_DIR):
        raise SystemExit(f"[!] ISL dataset not found at: {DATASET_DIR}")

    train_samples, val_samples, class_names, class_to_idx = build_block_split()
    train_set = ISLFrameDataset(train_samples, class_to_idx, train_transforms)
    val_set   = ISLFrameDataset(val_samples,   class_to_idx, val_transforms)
    n_train, n_val = len(train_set), len(val_set)

    # Python 3.14 defaults to the forkserver start method, which is blocked in some
    # sandboxes (ConnectionResetError on worker spawn). Use fork where it exists.
    mp_ctx = {}
    if NUM_WORKERS > 0 and hasattr(os, "fork"):
        mp_ctx["multiprocessing_context"] = "fork"

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, **mp_ctx)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, **mp_ctx)

    print(f"[ISL] Block split: BLOCK_SIZE={BLOCK_SIZE}, {VAL_SPLIT:.0%} of blocks held out")
    print(f"[ISL] Classes ({len(class_names)}): {class_names}")
    print(f"[ISL] Train: {n_train} | Val: {n_val}")
    return train_loader, val_loader, class_names


def train():
    os.makedirs(os.path.dirname(MODEL_SAVE), exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  ISL Training  |  Device: {DEVICE}")
    print(f"{'='*60}\n")

    train_loader, val_loader, class_names = get_dataloaders()
    num_classes = len(class_names)

    model     = build_model(num_classes=num_classes, pretrained=True)
    model     = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR,
        steps_per_epoch=len(train_loader),
        epochs=NUM_EPOCHS
    )

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss    += loss.item() * images.size(0)
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += images.size(0)

            # Progress within epoch
            if (batch_idx + 1) % 50 == 0:
                partial_acc = train_correct / train_total * 100
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} | Acc: {partial_acc:.1f}%")

        # ── Validate ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss    = criterion(outputs, labels)
                val_loss    += loss.item() * images.size(0)
                preds        = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += images.size(0)

        t_loss = train_loss / train_total
        t_acc  = train_correct / train_total * 100
        v_loss = val_loss / val_total
        v_acc  = val_correct / val_total * 100
        elapsed = time.time() - t0

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        print(f"\nEpoch [{epoch:02d}/{NUM_EPOCHS}] | "
              f"Train Loss: {t_loss:.4f} | Train Acc: {t_acc:.1f}% | "
              f"Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.1f}% | "
              f"Time: {elapsed:.1f}s")

        # Save best model
        # NOTE: This dataset provides only one recording session per class
        # (single signer/background/lighting, ~1200 sequential frames).
        # Even a temporally-correct train/val split cannot separate
        # session-level shortcuts (background, lighting) from the actual
        # hand gesture. This model's reported accuracy is NOT a validated
        # generalization result -- see Module 1's landmark-based model
        # (97.30% on the same split) for the honest accuracy figure.
        #
        # Measured: a held-out frame sits 2.7-4.9 (mean abs pixel delta) from
        # its nearest same-class TRAIN frame, but 25-34 from any other class.
        # Widening the split to a maximal 600-frame temporal halving only
        # moves that to 4.1-10.6 -- no block size fixes it. A second recording
        # session (different signer/background) is what a real number needs.
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save({
                "epoch":             epoch,
                "model_state_dict":  model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc":           v_acc,
                "class_names":       class_names,
                "num_classes":       num_classes,
            }, MODEL_SAVE)
            print(f"  ✓ New best! Saved model (val_acc={v_acc:.1f}%)")

    print(f"\n[ISL] Training complete! Best val accuracy: {best_val_acc:.1f}%")
    print(f"[ISL] Model saved to: {MODEL_SAVE}\n")
    return history


if __name__ == "__main__":
    train()

"""
Training script for the ISL (Indian Sign Language) model.
Dataset: indian sign language/Indian/ — 35 classes (A-Z + 1-9)
~1,221 images per class — large balanced dataset, lighter augmentation needed.
GPU training via CUDA.
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import build_model, NORMALIZE_MEAN, NORMALIZE_STD, INPUT_SIZE

# ─── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR  = r"r:\Projects\Sign Language\indian sign language\Indian"
MODEL_SAVE   = r"r:\Projects\Sign Language\models\isl_model.pth"
NUM_EPOCHS   = 10
BATCH_SIZE   = 64   # Larger batch — more data
LR           = 1e-3
VAL_SPLIT    = 0.15
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


def get_dataloaders():
    full_dataset = datasets.ImageFolder(DATASET_DIR, transform=train_transforms)
    class_names  = full_dataset.classes
    n_total      = len(full_dataset)
    n_val        = int(n_total * VAL_SPLIT)
    n_train      = n_total - n_val

    train_set, val_set = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    val_set.dataset = datasets.ImageFolder(DATASET_DIR, transform=val_transforms)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

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

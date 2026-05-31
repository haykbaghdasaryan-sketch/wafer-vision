"""
WaferVision — Focal Loss + Mixup Training (Kaggle T4 GPU)
==========================================================
Copy-paste this cell into a Kaggle notebook with:
  - GPU: T4 (16 GB VRAM)
  - Internet enabled (for pip install)
  - Dataset: qingyi/wm811k-wafer-map attached

Target: improve macro KNN@5 from 90.8% to 92%+ by boosting Loc (85→90%) and Scratch (84→88%)
"""

# %% [markdown]
# # WaferVision — Focal Loss + Mixup Training
# **Goal:** Boost Loc and Scratch per-class KNN@5 using focal loss (γ=2.0) + mixup (α=0.4)

# %% Cell 1: Setup
# !pip install -q torch torchvision scikit-learn faiss-cpu tqdm numpy pandas umap-learn

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from pathlib import Path
from collections import Counter

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# %% Cell 2: Focal Loss Implementation
class FocalLoss(nn.Module):
    """Focal Loss: -alpha_t * (1-p_t)^gamma * log(p_t)"""

    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)

    def forward(self, logits, labels):
        probs = F.softmax(logits, dim=1)
        p_t = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - p_t) ** self.gamma
        log_p_t = torch.log(p_t + 1e-8)
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[labels]
            loss = -alpha_t * focal_weight * log_p_t
        else:
            loss = -focal_weight * log_p_t
        return loss.mean()


# %% Cell 3: Mixup
def mixup_batch(images, labels, alpha=0.4):
    """Apply mixup to a batch. Returns (mixed_images, labels_a, labels_b, lam)."""
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)  # ensure lam >= 0.5
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)
    mixed = lam * images + (1.0 - lam) * images[index]
    return mixed, labels, labels[index], lam


# %% Cell 4: Load and preprocess WM-811K
import pickle
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Adjust path based on your Kaggle dataset location
DATA_PATH = "/kaggle/input/wm811k-wafer-map/LSWMD.pkl"
if not os.path.exists(DATA_PATH):
    DATA_PATH = "data/raw/LSWMD.pkl"

print(f"Loading dataset from {DATA_PATH}...")
with open(DATA_PATH, "rb") as f:
    df = pickle.load(f)

# Filter labeled samples (exclude 'none' and unlabeled)
CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Extract labeled samples
labeled_mask = df["failureType"].apply(
    lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0 and x[0][0] in CLASSES
)
df_labeled = df[labeled_mask].copy()
df_labeled["label"] = df_labeled["failureType"].apply(lambda x: x[0][0])
df_labeled["label_idx"] = df_labeled["label"].map(CLASS_TO_IDX)

print(f"Labeled samples: {len(df_labeled)}")
print(f"Class distribution:\n{df_labeled['label'].value_counts()}")

# Lot-based split to prevent data leakage
df_labeled["lotName"] = df_labeled["lotName"].astype(str)
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
train_idx, temp_idx = next(gss.split(df_labeled, groups=df_labeled["lotName"]))

df_train = df_labeled.iloc[train_idx]
df_temp = df_labeled.iloc[temp_idx]

# Split temp into val/test
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
val_idx, test_idx = next(gss2.split(df_temp, groups=df_temp["lotName"]))
df_val = df_temp.iloc[val_idx]
df_test = df_temp.iloc[test_idx]

print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")


# %% Cell 5: Wafer map to tensor
def wafer_to_tensor(wafer_map, target_size=64):
    """Convert raw wafer map to 3-channel 64x64 tensor."""
    wm = np.array(wafer_map, dtype=np.float32)
    # 3-channel encoding: [defect, normal, edge]
    ch0 = (wm == 2).astype(np.float32)  # defect pixels
    ch1 = (wm == 1).astype(np.float32)  # normal die
    ch2 = (wm >= 1).astype(np.float32)  # wafer region (edge)
    tensor = np.stack([ch0, ch1, ch2], axis=0)
    tensor = torch.from_numpy(tensor).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(target_size, target_size), mode="bilinear", align_corners=False)
    return tensor.squeeze(0)


# Process all splits
def process_split(df_split):
    tensors, labels = [], []
    for _, row in df_split.iterrows():
        try:
            t = wafer_to_tensor(row["waferMap"])
            tensors.append(t)
            labels.append(row["label_idx"])
        except Exception:
            continue
    return torch.stack(tensors), torch.tensor(labels, dtype=torch.long)

print("Processing train split...")
train_data, train_labels = process_split(df_train)
print("Processing val split...")
val_data, val_labels = process_split(df_val)
print("Processing test split...")
test_data, test_labels = process_split(df_test)
print(f"Tensors: train={train_data.shape}, val={val_data.shape}, test={test_data.shape}")


# %% Cell 6: Dataset and DataLoader with balanced sampling
class WaferDataset(Dataset):
    def __init__(self, data, labels, augment=False):
        self.data = data
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.augment:
            # Random rotation (k*90)
            k = torch.randint(0, 4, (1,)).item()
            x = torch.rot90(x, k, [1, 2])
            # Random flip
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, [2])
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, [1])
            # Gaussian noise
            x = x + torch.randn_like(x) * 0.01
            x = x.clamp(0, 1)
        return x, self.labels[idx]

# Balanced sampling: oversample minority classes
class_counts = Counter(train_labels.numpy().tolist())
weights = [1.0 / class_counts[label.item()] for label in train_labels]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

BATCH_SIZE = 64
train_ds = WaferDataset(train_data, train_labels, augment=True)
val_ds = WaferDataset(val_data, val_labels, augment=False)
test_ds = WaferDataset(test_data, test_labels, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)


# %% Cell 7: ResNet50 backbone + classifier
import torchvision.models as models

backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
embedding_dim = backbone.fc.in_features
backbone.fc = nn.Identity()  # remove classification head
backbone = backbone.to(device)

classifier = nn.Linear(embedding_dim, len(CLASSES)).to(device)

# Compute focal loss alpha weights from class counts
counts_tensor = torch.tensor([class_counts.get(i, 1) for i in range(len(CLASSES))], dtype=torch.float32)
inv_freq = 1.0 / counts_tensor.clamp(min=1)
alpha_weights = (inv_freq / inv_freq.sum() * len(CLASSES)).to(device)

criterion = FocalLoss(gamma=2.0, alpha=alpha_weights)
print(f"Focal Loss gamma=2.0, alpha weights: {alpha_weights.cpu().numpy().round(2)}")


# %% Cell 8: Training loop with Focal + Mixup
EPOCHS = 60
LR = 5e-5
MIXUP_ALPHA = 0.4

optimizer = torch.optim.Adam(
    list(backbone.parameters()) + list(classifier.parameters()),
    lr=LR, weight_decay=1e-5
)

# Cosine annealing with warmup
warmup_epochs = 5
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - warmup_epochs, eta_min=1e-7)

best_val_loss = float("inf")
best_state = None

print(f"\n{'='*60}")
print(f"Training: Focal(γ=2.0) + Mixup(α={MIXUP_ALPHA}), {EPOCHS} epochs, lr={LR}")
print(f"{'='*60}\n")

for epoch in range(EPOCHS):
    # Warmup
    if epoch < warmup_epochs:
        lr_scale = (epoch + 1) / warmup_epochs
        for pg in optimizer.param_groups:
            pg["lr"] = LR * lr_scale

    backbone.train()
    classifier.train()
    train_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Mixup (50% of batches)
        if torch.rand(1).item() < 0.5:
            images, labels_a, labels_b, lam = mixup_batch(images, labels, alpha=MIXUP_ALPHA)
            embeddings = backbone(images)
            logits = classifier(embeddings)
            loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
            preds = logits.argmax(1)
            correct += (lam * (preds == labels_a).sum().item() + (1 - lam) * (preds == labels_b).sum().item())
        else:
            embeddings = backbone(images)
            logits = classifier(embeddings)
            loss = criterion(logits, labels)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        total += images.size(0)

    if epoch >= warmup_epochs:
        scheduler.step()

    # Validation
    backbone.eval()
    classifier.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = classifier(backbone(images))
            loss = criterion(logits, labels)
            val_loss += loss.item() * images.size(0)
            val_correct += (logits.argmax(1) == labels).sum().item()
            val_total += images.size(0)

    train_loss /= total
    val_loss /= val_total
    train_acc = correct / total
    val_acc = val_correct / val_total

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {
            "backbone": {k: v.cpu() for k, v in backbone.state_dict().items()},
            "classifier": {k: v.cpu() for k, v in classifier.state_dict().items()},
            "epoch": epoch,
        }

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

print(f"\nBest val loss: {best_val_loss:.4f} at epoch {best_state['epoch']+1}")


# %% Cell 9: Load best model and extract embeddings
backbone.load_state_dict({k: v.to(device) for k, v in best_state["backbone"].items()})
backbone.eval()

def extract_embeddings(loader):
    all_emb, all_lab = [], []
    with torch.no_grad():
        for images, labels in loader:
            emb = backbone(images.to(device))
            all_emb.append(emb.cpu())
            all_lab.append(labels)
    return torch.cat(all_emb).numpy(), torch.cat(all_lab).numpy()

print("Extracting embeddings...")
train_emb, train_lab = extract_embeddings(train_loader)
test_emb, test_lab = extract_embeddings(test_loader)
print(f"Train embeddings: {train_emb.shape}, Test embeddings: {test_emb.shape}")


# %% Cell 10: KNN Evaluation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# KNN@5
knn = KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=-1)
knn.fit(train_emb, train_lab)
pred_labels = knn.predict(test_emb)

print("\n" + "="*60)
print("KNN@5 Classification Report (Focal + Mixup)")
print("="*60)
report = classification_report(test_lab, pred_labels, target_names=CLASSES, digits=3)
print(report)

# Per-class KNN@5 accuracy
from sklearn.metrics import accuracy_score
print("\nPer-class KNN@5:")
for i, cls in enumerate(CLASSES):
    mask = test_lab == i
    if mask.sum() > 0:
        acc = accuracy_score(test_lab[mask], pred_labels[mask])
        print(f"  {cls:12s}: {acc*100:.1f}% ({mask.sum()} samples)")

# Macro average
from sklearn.metrics import recall_score
macro_knn5 = recall_score(test_lab, pred_labels, average="macro")
print(f"\n{'='*60}")
print(f"MACRO KNN@5: {macro_knn5*100:.1f}%")
print(f"{'='*60}")


# %% Cell 11: Save checkpoint
SAVE_PATH = "focal_mixup_best.pth"
torch.save(best_state, SAVE_PATH)
print(f"Saved checkpoint to {SAVE_PATH}")
print(f"\nFinal summary:")
print(f"  Model: ResNet50 + Focal(γ=2.0) + Mixup(α=0.4)")
print(f"  Best epoch: {best_state['epoch']+1}")
print(f"  Macro KNN@5: {macro_knn5*100:.1f}%")

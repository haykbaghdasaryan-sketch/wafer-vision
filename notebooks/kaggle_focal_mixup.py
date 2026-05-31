"""
WaferVision — Focal Loss + Mixup FULL POWER (Kaggle)
======================================================
Instructions:
1. Kaggle -> New Notebook
2. Add Input: search "wm811k-wafer-map" by qingyi
3. Settings: GPU T4 x2 (or P100), Internet ON
4. Paste this into ONE cell, Run All
5. ~3-5 hours depending on GPU

Uses ALL data, 150 epochs, full augmentation pipeline.
"""

# ============================================================
# SETUP
# ============================================================
import os, subprocess, sys

REPO = '/kaggle/working/wafer-vision'
if not os.path.exists(REPO):
    subprocess.run(['git', 'clone', '-b', 'feat/focal-mixup',
        'https://github.com/haykbaghdasaryan-sketch/wafer-vision.git', REPO], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
        'hydra-core', 'omegaconf', 'pydantic', 'structlog', 'tqdm',
        'grad-cam', 'umap-learn', 'psutil', 'faiss-cpu'], check=True)

sys.path.insert(0, REPO)
os.chdir(REPO)

import pickle, gc, time, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from collections import Counter, defaultdict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, recall_score

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # faster convolutions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | GPU: {torch.cuda.get_device_name(0) if device.type=='cuda' else 'N/A'}")
if device.type == 'cuda':
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

# ============================================================
# LOAD WM-811K
# ============================================================
class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('pandas.indexes'):
            module = module.replace('pandas.indexes', 'pandas.core.indexes')
        if module == 'pandas.core.internals' and name == 'BlockManager':
            module = 'pandas.core.internals.managers'
        return super().find_class(module, name)

PKL_PATHS = [
    "/kaggle/input/datasets/qingyi/wm811k-wafer-map/LSWMD.pkl",
    "/kaggle/input/wm811k-wafer-map/LSWMD.pkl",
    "/kaggle/input/wm811k/LSWMD.pkl",
]
pkl_path = next((p for p in PKL_PATHS if os.path.exists(p)), None)
assert pkl_path, f"Dataset not found! Add 'wm811k-wafer-map' as Input."

print(f"\n[1/6] Loading WM-811K from {pkl_path}...")
with open(pkl_path, "rb") as f:
    df = CompatUnpickler(f, encoding='latin1').load()
print(f"  Total records: {len(df):,}")

# ============================================================
# PARSE ALL LABELED SAMPLES + LOT-BASED SPLIT
# ============================================================
CLASSES = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Near-full','Random','Scratch']
class_to_idx = {c: i for i, c in enumerate(CLASSES)}

print("\n[2/6] Parsing ALL labeled defects + lot-based split...")
lot_indices = defaultdict(list)
for idx in range(len(df)):
    row = df.iloc[idx]
    wm = row.get("waferMap", None)
    if wm is None or not hasattr(wm, 'shape') or wm.ndim != 2: continue
    if wm.shape[0] < 5 or wm.shape[1] < 5: continue
    raw = row.get("failureType", None)
    for _ in range(5):
        if isinstance(raw, np.ndarray):
            if raw.size == 0: raw = None; break
            raw = raw.flat[0]
        elif isinstance(raw, list):
            if len(raw) == 0: raw = None; break
            raw = raw[0]
        else: break
    if raw is None: continue
    label = str(raw).strip()
    if label not in class_to_idx: continue
    lot = row.get("lotName", f"unk_{idx}")
    if lot is None or (isinstance(lot, float) and np.isnan(lot)): lot = f"unk_{idx}"
    lot_indices[lot].append((idx, class_to_idx[label]))

total = sum(len(v) for v in lot_indices.values())
print(f"  {total:,} defects across {len(lot_indices):,} lots")

# Lot-split: 70/15/15 (honest, no leakage)
np.random.seed(SEED)
lots = list(lot_indices.keys()); np.random.shuffle(lots)
n = len(lots)
train_lots = set(lots[:int(0.7*n)])
val_lots = set(lots[int(0.7*n):int(0.85*n)])
test_lots = set(lots[int(0.85*n):])

train_ic, val_ic, test_ic = [], [], []
for lot, items in lot_indices.items():
    for item in items:
        if lot in train_lots: train_ic.append(item)
        elif lot in val_lots: val_ic.append(item)
        else: test_ic.append(item)
del lot_indices; gc.collect()

print(f"  Raw split: train={len(train_ic):,} val={len(val_ic):,} test={len(test_ic):,}")

# ============================================================
# PROCESS ALL SAMPLES (no subsampling — full dataset)
# ============================================================
print("\n[3/6] Processing ALL samples to tensors (this takes a few minutes)...")

def make_tensors(ic_list, desc=""):
    t, l = [], []
    for i, (row_idx, ci) in enumerate(ic_list):
        wm = df.iloc[row_idx]["waferMap"]
        x = torch.from_numpy(wm.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        x = F.interpolate(x, size=(96, 96), mode='bilinear', align_corners=False)  # 96x96 for more detail
        x = x.squeeze(0) / 2.0
        x = x.repeat(3, 1, 1)
        t.append(x); l.append(ci)
        if (i+1) % 5000 == 0:
            print(f"    {desc}: {i+1}/{len(ic_list)}")
    return torch.stack(t), torch.tensor(l, dtype=torch.int64)

val_data, val_labels = make_tensors(val_ic, "val"); del val_ic
test_data, test_labels = make_tensors(test_ic, "test"); del test_ic

# Train: use ALL samples + oversample minority to max_class_count with augmentation
train_by_class = defaultdict(list)
for row_idx, ci in train_ic: train_by_class[ci].append(row_idx)
del train_ic

# Find the largest class count and oversample everything to that level
class_counts_raw = {ci: len(rows) for ci, rows in train_by_class.items()}
TARGET = max(class_counts_raw.values())  # match largest class
print(f"\n  Balancing train to {TARGET}/class (largest class size):")

train_t, train_l = [], []
for ci in range(8):
    rows = train_by_class[ci]; n_rows = len(rows)
    # First: add ALL original samples
    for r in rows:
        wm = df.iloc[r]["waferMap"]
        x = torch.from_numpy(wm.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        x = F.interpolate(x, size=(96, 96), mode='bilinear', align_corners=False)
        x = x.squeeze(0) / 2.0; x = x.repeat(3, 1, 1)
        train_t.append(x); train_l.append(ci)
    # Then: oversample with heavy augmentation to reach TARGET
    deficit = TARGET - n_rows
    if deficit > 0:
        for j in range(deficit):
            r = rows[j % n_rows]
            wm = df.iloc[r]["waferMap"]
            x = torch.from_numpy(wm.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            x = F.interpolate(x, size=(96, 96), mode='bilinear', align_corners=False)
            x = x.squeeze(0) / 2.0; x = x.repeat(3, 1, 1)
            # Heavy augmentation for oversampled
            k = torch.randint(0, 4, (1,)).item()
            x = torch.rot90(x, k, [1, 2])
            if torch.rand(1).item() > 0.5: x = torch.flip(x, [2])
            if torch.rand(1).item() > 0.5: x = torch.flip(x, [1])
            # Stronger noise for diversity
            x = (x + torch.randn_like(x) * 0.03).clamp(0, 1)
            train_t.append(x); train_l.append(ci)
    print(f"  {CLASSES[ci]:12s}: {n_rows:5d} -> {TARGET} ({deficit:+d} augmented)")

del df, train_by_class; gc.collect()
train_data = torch.stack(train_t); del train_t; gc.collect()
train_labels = torch.tensor(train_l, dtype=torch.int64); del train_l
print(f"\n  Final: Train={train_data.shape} Val={val_data.shape} Test={test_data.shape}")
print(f"  Total training samples: {len(train_data):,}")

# ============================================================
# TRAINING: 150 EPOCHS, FOCAL + MIXUP, FULL POWER
# ============================================================
print("\n[4/6] Training: Focal(γ=2.0) + Mixup(α=0.4), 150 epochs, full data...")

class FocalLoss(nn.Module):
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
            loss = -self.alpha[labels] * focal_weight * log_p_t
        else:
            loss = -focal_weight * log_p_t
        return loss.mean()

def mixup_batch(images, labels, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)
    index = torch.randperm(images.size(0), device=images.device)
    return lam * images + (1-lam) * images[index], labels, labels[index], lam

class WaferDS(Dataset):
    def __init__(self, data, labels, aug=False):
        self.data, self.labels, self.aug = data, labels, aug
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        x = self.data[i]
        if self.aug:
            # Heavy augmentation: rotation + flip + noise + random erasing
            x = torch.rot90(x, torch.randint(0, 4, (1,)).item(), [1, 2])
            if torch.rand(1).item() > 0.5: x = torch.flip(x, [2])
            if torch.rand(1).item() > 0.5: x = torch.flip(x, [1])
            # Gaussian noise
            x = x + torch.randn_like(x) * 0.015
            # Random erasing (10% chance, erase 5-15% of area)
            if torch.rand(1).item() < 0.1:
                _, h, w = x.shape
                eh, ew = int(h*np.random.uniform(0.05,0.15)), int(w*np.random.uniform(0.05,0.15))
                top, left = np.random.randint(0, h-eh), np.random.randint(0, w-ew)
                x[:, top:top+eh, left:left+ew] = 0
            x = x.clamp(0, 1)
        return x, self.labels[i]

# DataLoaders with larger batch size for full utilization
BATCH_SIZE = 128  # bigger batch for better GPU utilization
cc = Counter(train_labels.numpy().tolist())
sampler = WeightedRandomSampler([1.0/cc[l.item()] for l in train_labels], len(train_labels), replacement=True)
train_loader = DataLoader(WaferDS(train_data, train_labels, aug=True), batch_size=BATCH_SIZE,
                          sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
val_loader = DataLoader(WaferDS(val_data, val_labels), batch_size=256, num_workers=4, pin_memory=True)
test_loader = DataLoader(WaferDS(test_data, test_labels), batch_size=256, num_workers=4, pin_memory=True)

# Model: ResNet50 pretrained
backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
backbone.fc = nn.Identity()
backbone = backbone.to(device)
classifier = nn.Linear(2048, 8).to(device)

# Focal loss with class-balanced alpha
counts_t = torch.tensor([cc.get(i, 1) for i in range(8)], dtype=torch.float32)
inv_freq = 1.0 / counts_t.clamp(min=1)
alpha_w = (inv_freq / inv_freq.sum() * 8).to(device)
criterion = FocalLoss(gamma=2.0, alpha=alpha_w)
print(f"  Focal alpha: {alpha_w.cpu().numpy().round(2)}")

# Training config
EPOCHS = 150
LR = 3e-4  # higher LR for longer training with cosine decay
WARMUP = 10
MIXUP_ALPHA = 0.4
MIXUP_P = 0.5  # apply mixup 50% of batches

# AdamW with higher LR for long training
optimizer = torch.optim.AdamW(
    [{'params': backbone.parameters(), 'lr': LR * 0.1},  # backbone: lower LR (pretrained)
     {'params': classifier.parameters(), 'lr': LR}],      # head: full LR
    weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP, eta_min=1e-7)

best_val_loss = float("inf")
best_state = None
patience_counter = 0
PATIENCE = 30  # longer patience for longer training

t0 = time.time()
print(f"  Epochs: {EPOCHS} | LR: {LR} (backbone {LR*0.1}) | Batch: {BATCH_SIZE}")
print(f"  Mixup: α={MIXUP_ALPHA}, p={MIXUP_P} | Patience: {PATIENCE}")
print(f"  Train batches/epoch: {len(train_loader)}")
print()

for epoch in range(EPOCHS):
    # Warmup
    if epoch < WARMUP:
        lr_scale = (epoch + 1) / WARMUP
        for pg in optimizer.param_groups:
            pg["lr"] = pg["lr"] / lr_scale * ((epoch + 1) / WARMUP) if epoch > 0 else pg["lr"] * lr_scale

    backbone.train(); classifier.train()
    t_loss, correct, total = 0.0, 0, 0

    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)

        if torch.rand(1).item() < MIXUP_P:
            imgs, la, lb, lam = mixup_batch(imgs, lbls, MIXUP_ALPHA)
            logits = classifier(backbone(imgs))
            loss = lam * criterion(logits, la) + (1 - lam) * criterion(logits, lb)
            preds = logits.argmax(1)
            correct += lam * (preds == la).sum().item() + (1 - lam) * (preds == lb).sum().item()
        else:
            logits = classifier(backbone(imgs))
            loss = criterion(logits, lbls)
            correct += (logits.argmax(1) == lbls).sum().item()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
        optimizer.step()
        t_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)

    if epoch >= WARMUP:
        scheduler.step()

    # Validation
    backbone.eval(); classifier.eval()
    v_loss, v_correct, v_total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
            logits = classifier(backbone(imgs))
            l = criterion(logits, lbls)
            v_loss += l.item() * imgs.size(0)
            v_correct += (logits.argmax(1) == lbls).sum().item()
            v_total += imgs.size(0)
    v_loss /= v_total
    v_acc = v_correct / v_total

    # Best model tracking
    if v_loss < best_val_loss:
        best_val_loss = v_loss
        best_state = {
            "backbone": {k: v.cpu() for k, v in backbone.state_dict().items()},
            "classifier": {k: v.cpu() for k, v in classifier.state_dict().items()},
            "epoch": epoch,
            "val_loss": v_loss,
            "val_acc": v_acc,
        }
        patience_counter = 0
    else:
        patience_counter += 1

    # Logging
    if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == EPOCHS - 1:
        elapsed = time.time() - t0
        eta = elapsed / (epoch + 1) * (EPOCHS - epoch - 1)
        print(f"  Ep {epoch+1:3d}/{EPOCHS} | "
              f"t_loss={t_loss/total:.4f} t_acc={correct/total:.3f} | "
              f"v_loss={v_loss:.4f} v_acc={v_acc:.3f} | "
              f"lr={optimizer.param_groups[0]['lr']:.1e} | "
              f"patience={patience_counter}/{PATIENCE} | "
              f"ETA={eta/60:.0f}min")

    # Early stopping
    if patience_counter >= PATIENCE:
        print(f"\n  Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
        break

elapsed = time.time() - t0
print(f"\n  Training done: {elapsed/60:.1f} min ({elapsed/3600:.1f} hrs)")
print(f"  Best: epoch {best_state['epoch']+1}, val_loss={best_state['val_loss']:.4f}, val_acc={best_state['val_acc']:.3f}")

# ============================================================
# EVALUATE: KNN@5 (honest: train->index, test->query)
# ============================================================
print("\n[5/6] KNN@5 evaluation (honest: train=index, test=query)...")
backbone.load_state_dict({k: v.to(device) for k, v in best_state["backbone"].items()})
backbone.eval()

def get_emb(loader):
    E, L = [], []
    with torch.no_grad():
        for x, y in loader:
            E.append(backbone(x.to(device)).cpu())
            L.append(y)
    return torch.cat(E).numpy(), torch.cat(L).numpy()

train_emb, train_lab = get_emb(train_loader)
test_emb, test_lab = get_emb(test_loader)
print(f"  Embeddings: train={train_emb.shape}, test={test_emb.shape}")

# KNN with multiple K values
for K in [1, 3, 5, 10]:
    knn = KNeighborsClassifier(n_neighbors=K, metric="cosine", n_jobs=-1)
    knn.fit(train_emb, train_lab)
    pred = knn.predict(test_emb)
    macro = recall_score(test_lab, pred, average="macro")
    print(f"  KNN@{K:2d} macro: {macro*100:.1f}%")

# Detailed report for KNN@5
knn = KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=-1)
knn.fit(train_emb, train_lab)
pred = knn.predict(test_emb)

print("\n" + "=" * 65)
print("  KNN@5 Classification Report — Focal Loss + Mixup (FULL POWER)")
print("=" * 65)
print(classification_report(test_lab, pred, target_names=CLASSES, digits=3))

macro = recall_score(test_lab, pred, average="macro")
print(f"{'=' * 65}")
print(f"  >>> MACRO KNN@5: {macro*100:.1f}% <<<")
print(f"{'=' * 65}")

# Per-class breakdown
print("\n  Per-class KNN@5:")
from sklearn.metrics import accuracy_score
for i, cls in enumerate(CLASSES):
    mask = test_lab == i
    if mask.sum() > 0:
        acc = accuracy_score(test_lab[mask], pred[mask])
        print(f"    {cls:12s}: {acc*100:5.1f}% ({mask.sum():4d} test samples)")

# ============================================================
# SAVE + SUMMARY
# ============================================================
print(f"\n[6/6] Saving...")
torch.save(best_state, "focal_mixup_fullpower_best.pth")

print(f"\n{'=' * 65}")
print(f"  FINAL SUMMARY")
print(f"{'=' * 65}")
print(f"  Method:      ResNet50 + Focal(γ=2.0) + Mixup(α=0.4)")
print(f"  Data:        ALL {total:,} defects, balanced to {TARGET}/class")
print(f"  Resolution:  96x96 (3-channel)")
print(f"  Split:       Lot-based 70/15/15 (no leakage)")
print(f"  Epochs:      {best_state['epoch']+1} (of {EPOCHS} max)")
print(f"  LR:          {LR} (backbone {LR*0.1}) + cosine + {WARMUP} warmup")
print(f"  Batch:       {BATCH_SIZE}")
print(f"  Macro KNN@5: {macro*100:.1f}%")
print(f"  Time:        {elapsed/60:.1f} min on {torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}")
print(f"{'=' * 65}")
print(f"\n  Repo: https://github.com/haykbaghdasaryan-sketch/wafer-vision")
print(f"  Branch: feat/focal-mixup")

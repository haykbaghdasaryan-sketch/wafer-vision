"""
WaferVision — Focal Loss + Mixup, FULL POWER, 3 Backbones (Kaggle)
====================================================================
Instructions:
1. Kaggle -> New Notebook
2. Add Input: search "wm811k-wafer-map" by qingyi
3. Settings: GPU T4 x2 (or P100), Internet ON
4. Paste this into ONE cell, Run All
5. ~8-12 hours total (ResNet50 + EfficientNet-B0 + ViT-B/16)

Runs THREE experiments:
  A) ResNet50 + Focal(γ=2.0) + Mixup(α=0.4) — 150 epochs
  B) EfficientNet-B0 + Focal(γ=2.0) + Mixup(α=0.4) — 150 epochs
  C) ViT-B/16 + Focal(γ=2.0) + Mixup(α=0.4) — 150 epochs
All with FULL data, 96x96 (ViT auto-resizes to 224), lot-based honest eval.

GPU memory is freed between models, and ViT uses REAL gradient checkpointing
(torchvision has no built-in flag) + batch=32 to fit on a single T4.
If ViT still OOMs or the kernel restarts, run notebooks/kaggle_vit_only.py
to finish just the ViT experiment and print the final comparison table.
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
from sklearn.metrics import classification_report, recall_score, accuracy_score

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | GPU: {torch.cuda.get_device_name(0) if device.type=='cuda' else 'N/A'}")
if device.type == 'cuda':
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

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

print(f"\n[1/7] Loading WM-811K from {pkl_path}...")
with open(pkl_path, "rb") as f:
    df = CompatUnpickler(f, encoding='latin1').load()
print(f"  Total records: {len(df):,}")

# ============================================================
# PARSE + LOT-BASED SPLIT
# ============================================================
CLASSES = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Near-full','Random','Scratch']
class_to_idx = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

print("\n[2/7] Parsing ALL labeled defects + lot-based split...")
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

total_defects = sum(len(v) for v in lot_indices.values())
print(f"  {total_defects:,} defects across {len(lot_indices):,} lots")

# Lot-split: 70/15/15
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
print(f"  Split: train={len(train_ic):,} val={len(val_ic):,} test={len(test_ic):,}")

# ============================================================
# PROCESS ALL SAMPLES TO TENSORS (96x96)
# ============================================================
print("\n[3/7] Processing ALL samples to 96x96 tensors...")

def make_tensors(ic_list, desc=""):
    t, l = [], []
    for i, (row_idx, ci) in enumerate(ic_list):
        wm = df.iloc[row_idx]["waferMap"]
        x = torch.from_numpy(wm.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        x = F.interpolate(x, size=(96, 96), mode='bilinear', align_corners=False)
        x = x.squeeze(0) / 2.0
        x = x.repeat(3, 1, 1)
        t.append(x); l.append(ci)
        if (i+1) % 5000 == 0: print(f"    {desc}: {i+1}/{len(ic_list)}")
    return torch.stack(t), torch.tensor(l, dtype=torch.int64)

val_data, val_labels = make_tensors(val_ic, "val"); del val_ic
test_data, test_labels = make_tensors(test_ic, "test"); del test_ic

# Balanced train: oversample minority to match largest class
train_by_class = defaultdict(list)
for row_idx, ci in train_ic: train_by_class[ci].append(row_idx)
del train_ic

class_counts_raw = {ci: len(rows) for ci, rows in train_by_class.items()}
TARGET = max(class_counts_raw.values())
print(f"\n  Balancing to {TARGET}/class:")

train_t, train_l = [], []
for ci in range(NUM_CLASSES):
    rows = train_by_class[ci]; n_rows = len(rows)
    for r in rows:
        wm = df.iloc[r]["waferMap"]
        x = torch.from_numpy(wm.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        x = F.interpolate(x, size=(96, 96), mode='bilinear', align_corners=False)
        x = x.squeeze(0) / 2.0; x = x.repeat(3, 1, 1)
        train_t.append(x); train_l.append(ci)
    deficit = TARGET - n_rows
    if deficit > 0:
        for j in range(deficit):
            r = rows[j % n_rows]
            wm = df.iloc[r]["waferMap"]
            x = torch.from_numpy(wm.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            x = F.interpolate(x, size=(96, 96), mode='bilinear', align_corners=False)
            x = x.squeeze(0) / 2.0; x = x.repeat(3, 1, 1)
            k = torch.randint(0, 4, (1,)).item()
            x = torch.rot90(x, k, [1, 2])
            if torch.rand(1).item() > 0.5: x = torch.flip(x, [2])
            if torch.rand(1).item() > 0.5: x = torch.flip(x, [1])
            x = (x + torch.randn_like(x) * 0.03).clamp(0, 1)
            train_t.append(x); train_l.append(ci)
    print(f"    {CLASSES[ci]:12s}: {n_rows:5d} -> {TARGET}")

del df, train_by_class; gc.collect()
train_data = torch.stack(train_t); del train_t; gc.collect()
train_labels = torch.tensor(train_l, dtype=torch.int64); del train_l
print(f"\n  Train={train_data.shape} Val={val_data.shape} Test={test_data.shape}")

# ============================================================
# SHARED COMPONENTS
# ============================================================
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
            x = torch.rot90(x, torch.randint(0, 4, (1,)).item(), [1, 2])
            if torch.rand(1).item() > 0.5: x = torch.flip(x, [2])
            if torch.rand(1).item() > 0.5: x = torch.flip(x, [1])
            x = x + torch.randn_like(x) * 0.015
            if torch.rand(1).item() < 0.1:
                _, h, w = x.shape
                eh = int(h * np.random.uniform(0.05, 0.15))
                ew = int(w * np.random.uniform(0.05, 0.15))
                top, left = np.random.randint(0, h-eh), np.random.randint(0, w-ew)
                x[:, top:top+eh, left:left+ew] = 0
            x = x.clamp(0, 1)
        return x, self.labels[i]

# DataLoaders
BATCH_SIZE = 128
cc = Counter(train_labels.numpy().tolist())
sampler = WeightedRandomSampler([1.0/cc[l.item()] for l in train_labels], len(train_labels), replacement=True)
train_loader = DataLoader(WaferDS(train_data, train_labels, aug=True), batch_size=BATCH_SIZE,
                          sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
val_loader = DataLoader(WaferDS(val_data, val_labels), batch_size=256, num_workers=4, pin_memory=True)
test_loader = DataLoader(WaferDS(test_data, test_labels), batch_size=256, num_workers=4, pin_memory=True)

# IMPORTANT: separate loader for embedding extraction — no augmentation, no random sampling
train_eval_loader = DataLoader(WaferDS(train_data, train_labels, aug=False),
                               batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

# Focal loss alpha
counts_t = torch.tensor([cc.get(i, 1) for i in range(NUM_CLASSES)], dtype=torch.float32)
inv_freq = 1.0 / counts_t.clamp(min=1)
alpha_w = (inv_freq / inv_freq.sum() * NUM_CLASSES).to(device)

# ============================================================
# TRAINING FUNCTION (reusable for both backbones)
# ============================================================
def train_model(backbone, classifier, backbone_name, epochs=150, lr=3e-4, warmup=10, patience=30, batch_override=None):
    """Train one model with Focal+Mixup. Returns (best_state, elapsed, history)."""
    criterion = FocalLoss(gamma=2.0, alpha=alpha_w)
    MIXUP_ALPHA, MIXUP_P = 0.4, 0.5

    # Allow per-model batch size override (e.g. ViT needs smaller batch)
    effective_batch = batch_override if batch_override else BATCH_SIZE
    if batch_override:
        # Fresh weighted sampler (same balancing) + augmentation, smaller batch for VRAM
        _sampler = WeightedRandomSampler(
            [1.0 / cc[l.item()] for l in train_labels], len(train_labels), replacement=True)
        _train_loader = DataLoader(WaferDS(train_data, train_labels, aug=True), batch_size=effective_batch,
                                   sampler=_sampler, num_workers=4, pin_memory=True, drop_last=True)
    else:
        _train_loader = train_loader

    optimizer = torch.optim.AdamW([
        {'params': backbone.parameters(), 'lr': lr * 0.1},
        {'params': classifier.parameters(), 'lr': lr}
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup, eta_min=1e-7)

    best_val_loss, best_state, patience_counter = float("inf"), None, 0
    t0 = time.time()

    print(f"\n  [{backbone_name}] Epochs={epochs} LR={lr} (backbone={lr*0.1}) Batch={effective_batch}")

    for epoch in range(epochs):
        if epoch < warmup:
            scale = (epoch + 1) / warmup
            optimizer.param_groups[0]["lr"] = lr * 0.1 * scale
            optimizer.param_groups[1]["lr"] = lr * scale

        backbone.train(); classifier.train()
        t_loss, correct, total = 0.0, 0, 0

        for imgs, lbls in _train_loader:
            imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
            if torch.rand(1).item() < MIXUP_P:
                imgs, la, lb, lam = mixup_batch(imgs, lbls, MIXUP_ALPHA)
                logits = classifier(backbone(imgs))
                loss = lam * criterion(logits, la) + (1-lam) * criterion(logits, lb)
                preds = logits.argmax(1)
                correct += lam*(preds==la).sum().item() + (1-lam)*(preds==lb).sum().item()
            else:
                logits = classifier(backbone(imgs))
                loss = criterion(logits, lbls)
                correct += (logits.argmax(1)==lbls).sum().item()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()*imgs.size(0); total += imgs.size(0)

        if epoch >= warmup: scheduler.step()

        # Validation
        backbone.eval(); classifier.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
                logits = classifier(backbone(imgs))
                l = criterion(logits, lbls)
                v_loss += l.item()*imgs.size(0)
                v_correct += (logits.argmax(1)==lbls).sum().item()
                v_total += imgs.size(0)
        v_loss /= v_total; v_acc = v_correct / v_total

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = {
                "backbone": {k:v.cpu() for k,v in backbone.state_dict().items()},
                "classifier": {k:v.cpu() for k,v in classifier.state_dict().items()},
                "epoch": epoch, "val_loss": v_loss, "val_acc": v_acc}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch+1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - t0
            eta = elapsed/(epoch+1)*(epochs-epoch-1)
            print(f"    Ep {epoch+1:3d}/{epochs} | t_loss={t_loss/total:.4f} t_acc={correct/total:.3f} | "
                  f"v_loss={v_loss:.4f} v_acc={v_acc:.3f} | pat={patience_counter}/{patience} | ETA={eta/60:.0f}m")

        if patience_counter >= patience:
            print(f"    Early stop at epoch {epoch+1}")
            break

    elapsed = time.time() - t0
    print(f"    Done: {elapsed/60:.1f}min | Best epoch {best_state['epoch']+1} v_loss={best_state['val_loss']:.4f}")
    return best_state, elapsed

def evaluate_model(backbone, best_state, backbone_name):
    """Load best weights, extract embeddings, compute KNN@1,3,5,10."""
    backbone.load_state_dict({k:v.to(device) for k,v in best_state["backbone"].items()})
    backbone.eval()

    def get_emb(loader):
        E, L = [], []
        with torch.no_grad():
            for x, y in loader:
                E.append(backbone(x.to(device)).cpu()); L.append(y)
        return torch.cat(E).numpy(), torch.cat(L).numpy()

    train_emb, train_lab = get_emb(train_eval_loader)
    test_emb, test_lab = get_emb(test_loader)

    results = {}
    for K in [1, 3, 5, 10]:
        knn = KNeighborsClassifier(n_neighbors=K, metric="cosine", n_jobs=-1)
        knn.fit(train_emb, train_lab)
        pred = knn.predict(test_emb)
        macro = recall_score(test_lab, pred, average="macro")
        results[f"KNN@{K}"] = macro
        if K == 5:
            results["report"] = classification_report(test_lab, pred, target_names=CLASSES, digits=3)
            results["per_class"] = {}
            for i, cls in enumerate(CLASSES):
                mask = test_lab == i
                if mask.sum() > 0:
                    results["per_class"][cls] = (accuracy_score(test_lab[mask], pred[mask]), int(mask.sum()))

    print(f"\n{'='*65}")
    print(f"  {backbone_name} — KNN Results (Focal+Mixup, Honest Eval)")
    print(f"{'='*65}")
    for K in [1, 3, 5, 10]:
        print(f"    KNN@{K:2d}: {results[f'KNN@{K}']*100:.1f}%")
    print(f"\n  Detailed KNN@5:")
    print(results["report"])
    print(f"  Per-class KNN@5:")
    for cls, (acc, n) in results["per_class"].items():
        print(f"    {cls:12s}: {acc*100:5.1f}% ({n:4d} samples)")
    print(f"\n  >>> {backbone_name} MACRO KNN@5: {results['KNN@5']*100:.1f}% <<<")
    return results

# ============================================================
# EXPERIMENT A: ResNet50 + Focal + Mixup
# ============================================================
print("\n" + "="*65)
print("  [4/7] EXPERIMENT A: ResNet50 + Focal(γ=2.0) + Mixup(α=0.4)")
print("="*65)

backbone_r50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
backbone_r50.fc = nn.Identity()
backbone_r50 = backbone_r50.to(device)
classifier_r50 = nn.Linear(2048, NUM_CLASSES).to(device)

best_r50, time_r50 = train_model(backbone_r50, classifier_r50, "ResNet50")
results_r50 = evaluate_model(backbone_r50, best_r50, "ResNet50")

# Free GPU memory
del backbone_r50, classifier_r50
torch.cuda.empty_cache(); gc.collect()

# ============================================================
# EXPERIMENT B: EfficientNet-B0 + Focal + Mixup
# ============================================================
print("\n" + "="*65)
print("  [5/7] EXPERIMENT B: EfficientNet-B0 + Focal(γ=2.0) + Mixup(α=0.4)")
print("="*65)

backbone_efn = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
backbone_efn.classifier = nn.Identity()  # removes the final FC (1280 -> 1000)
backbone_efn = backbone_efn.to(device)
classifier_efn = nn.Linear(1280, NUM_CLASSES).to(device)

best_efn, time_efn = train_model(backbone_efn, classifier_efn, "EfficientNet-B0")
results_efn = evaluate_model(backbone_efn, best_efn, "EfficientNet-B0")

# Free GPU memory
del backbone_efn, classifier_efn
torch.cuda.empty_cache(); gc.collect()

# ============================================================
# EXPERIMENT C: ViT-B/16 + Focal + Mixup (memory-safe)
# ============================================================
print("\n" + "="*65)
print("  [6/7] EXPERIMENT C: ViT-B/16 + Focal(γ=2.0) + Mixup(α=0.4)")
print("="*65)

# Aggressively free GPU before ViT (it needs ~10GB for 224x224)
torch.cuda.empty_cache()
gc.collect()
print(f"  GPU free before ViT: {torch.cuda.memory_reserved(0)/1024**3:.1f}GB reserved, "
      f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0))/1024**3:.1f}GB free")

# ViT-B/16 with REAL gradient checkpointing (torchvision has no built-in flag)
from torch.utils.checkpoint import checkpoint_sequential

class ViTWrapper(nn.Module):
    def __init__(self, ckpt_segments=4):
        super().__init__()
        self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.model.heads = nn.Identity()
        self.ckpt_segments = ckpt_segments  # split 12 encoder layers into N checkpointed chunks
    def forward(self, x):
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        # Replicate torchvision ViT forward, but checkpoint the transformer layers
        x = self.model._process_input(x)
        n = x.shape[0]
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        enc = self.model.encoder
        x = x + enc.pos_embedding
        x = enc.dropout(x)
        if self.training and x.requires_grad:
            x = checkpoint_sequential(enc.layers, self.ckpt_segments, x, use_reentrant=False)
        else:
            x = enc.layers(x)
        x = enc.ln(x)
        return x[:, 0]  # class token

try:
    backbone_vit = ViTWrapper().to(device)
    classifier_vit = nn.Linear(768, NUM_CLASSES).to(device)
    # ViT: smaller batch (32) to fit in VRAM after gradient checkpointing
    best_vit, time_vit = train_model(backbone_vit, classifier_vit, "ViT-B/16",
                                      epochs=150, lr=1e-4, warmup=15, patience=30,
                                      batch_override=32)
    results_vit = evaluate_model(backbone_vit, best_vit, "ViT-B/16")
    vit_success = True
except torch.cuda.OutOfMemoryError:
    print("  ViT OOM even with gradient checkpointing — skipping (needs >15GB VRAM)")
    results_vit = {k: 0.0 for k in ['KNN@1','KNN@3','KNN@5','KNN@10']}
    results_vit['per_class'] = {c: (0.0, 0) for c in CLASSES}
    time_vit = 0
    vit_success = False

# ============================================================
# COMPARISON TABLE  (robust: only shows models that completed)
# ============================================================
# Collect into one dict so the summary never crashes on a missing/OOM model.
ALL_RESULTS = {
    "ResNet50":        {"res": results_r50, "time": time_r50, "ok": True},
    "EfficientNet-B0": {"res": results_efn, "time": time_efn, "ok": True},
    "ViT-B/16":        {"res": results_vit, "time": time_vit, "ok": vit_success},
}

print("\n\n" + "="*65)
print("  [7/7] FINAL COMPARISON — 3 Backbones × Focal+Mixup")
print("="*65)
print(f"\n  {'Method':<35} {'KNN@1':<8} {'KNN@3':<8} {'KNN@5':<8} {'KNN@10':<8} {'Time':<8}")
print(f"  {'-'*75}")
for name, d in ALL_RESULTS.items():
    r = d["res"]
    if d["ok"]:
        tstr = f"{d['time']/60:.0f}min"
        print(f"  {name + ' + Focal + Mixup':<35} "
              f"{r['KNN@1']*100:<8.1f} {r['KNN@3']*100:<8.1f} "
              f"{r['KNN@5']*100:<8.1f} {r['KNN@10']*100:<8.1f} {tstr:<8}")
    else:
        print(f"  {name + ' + Focal + Mixup':<35} {'OOM (needs >15GB VRAM)':<40}")

# Per-class comparison for Loc and Scratch (target classes)
print(f"\n  Target class improvement (KNN@5):")
print(f"  {'Class':<12} {'ResNet50':<12} {'EffNet-B0':<12} {'ViT-B/16':<12}")
print(f"  {'-'*48}")
for cls in CLASSES:
    r50_acc = results_r50["per_class"].get(cls, (0, 0))[0]
    efn_acc = results_efn["per_class"].get(cls, (0, 0))[0]
    vit_acc = results_vit["per_class"].get(cls, (0, 0))[0]
    marker = " ← TARGET" if cls in ["Loc", "Scratch"] else ""
    print(f"  {cls:<12} {r50_acc*100:<12.1f} {efn_acc*100:<12.1f} {vit_acc*100:<12.1f}{marker}")

# ============================================================
# SAVE
# ============================================================
print(f"\n  Saving checkpoints...")
torch.save(best_r50, "resnet50_focal_mixup_best.pth")
torch.save(best_efn, "efficientnet_b0_focal_mixup_best.pth")
if vit_success:
    torch.save(best_vit, "vit_b16_focal_mixup_best.pth")
else:
    time_vit = 0  # ensure defined for total_time

total_time = time_r50 + time_efn + time_vit
# Pick best only among models that actually completed
completed = [(name, d["res"]) for name, d in ALL_RESULTS.items() if d["ok"]]
best_model = max(completed, key=lambda x: x[1]["KNN@5"])

print(f"\n{'='*65}")
print(f"  FINAL SUMMARY")
print(f"{'='*65}")
print(f"  Best model:    {best_model[0]} (Macro KNN@5 = {best_model[1]['KNN@5']*100:.1f}%)")
print(f"  ResNet50:      {results_r50['KNN@5']*100:.1f}%")
print(f"  EfficientNet:  {results_efn['KNN@5']*100:.1f}%")
print(f"  ViT-B/16:      {results_vit['KNN@5']*100:.1f}%" if vit_success else "  ViT-B/16:      OOM (skipped)")
print(f"  Total time:    {total_time/60:.0f} min ({total_time/3600:.1f} hrs)")
print(f"  GPU:           {torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}")
print(f"{'='*65}")
print(f"\n  Copy this table to docs/EXPERIMENTS.md and push!")
print(f"\n  Repo: https://github.com/haykbaghdasaryan-sketch/wafer-vision")
print(f"  Branch: feat/focal-mixup")

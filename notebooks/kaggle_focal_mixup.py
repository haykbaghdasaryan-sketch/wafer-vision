"""
WaferVision — Focal Loss + Mixup (Kaggle One-Click)
=====================================================
Instructions for your friend:
1. Kaggle -> New Notebook
2. Add Input: search "wm811k-wafer-map" by qingyi
3. Settings: GPU T4, Internet ON
4. Paste this entire file into ONE cell, hit Run All
5. Wait ~25 min, get results

That's it.
"""

# ============================================================
# CELL 1: SETUP (clone repo + install)
# ============================================================
import os, subprocess, sys

REPO = '/kaggle/working/wafer-vision'
if not os.path.exists(REPO):
    subprocess.run(['git', 'clone',
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | GPU: {torch.cuda.get_device_name(0) if device.type=='cuda' else 'N/A'}")

# ============================================================
# CELL 2: LOAD WM-811K (handles old pandas pickle format)
# ============================================================
class CompatUnpickler(pickle.Unpickler):
    """Fixes 'No module named pandas.indexes' for old WM-811K pickle."""
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
assert pkl_path, f"Dataset not found! Tried: {PKL_PATHS}. Add 'wm811k-wafer-map' dataset as Input."

print(f"\n[1/5] Loading WM-811K from {pkl_path}...")
with open(pkl_path, "rb") as f:
    df = CompatUnpickler(f, encoding='latin1').load()
print(f"  Total records: {len(df):,}")

# ============================================================
# CELL 3: PARSE + LOT-BASED SPLIT + BALANCED TRAIN
# ============================================================
CLASSES = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Near-full','Random','Scratch']
class_to_idx = {c: i for i, c in enumerate(CLASSES)}

print("\n[2/5] Parsing labels + lot-based split...")
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

def make_tensors(ic_list):
    t, l = [], []
    for row_idx, ci in ic_list:
        wm = df.iloc[row_idx]["waferMap"]
        x = torch.from_numpy(wm.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        x = F.interpolate(x, size=(64,64), mode='bilinear', align_corners=False)
        x = x.squeeze(0) / 2.0; x = x.repeat(3,1,1)
        t.append(x); l.append(ci)
    return torch.stack(t), torch.tensor(l, dtype=torch.int64)

val_data, val_labels = make_tensors(val_ic); del val_ic
test_data, test_labels = make_tensors(test_ic); del test_ic

# Balanced train: 4K/class (oversample minority with augmentation)
TARGET = 4000
train_by_class = defaultdict(list)
for row_idx, ci in train_ic: train_by_class[ci].append(row_idx)
del train_ic

train_t, train_l = [], []
for ci in range(8):
    rows = train_by_class[ci]; n_rows = len(rows)
    for r in rows[:TARGET]:
        wm = df.iloc[r]["waferMap"]
        x = torch.from_numpy(wm.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        x = F.interpolate(x, size=(64,64), mode='bilinear', align_corners=False)
        x = x.squeeze(0) / 2.0; x = x.repeat(3,1,1)
        train_t.append(x); train_l.append(ci)
    if n_rows < TARGET:
        for _ in range(TARGET - n_rows):
            r = rows[np.random.randint(0, n_rows)]
            wm = df.iloc[r]["waferMap"]
            x = torch.from_numpy(wm.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            x = F.interpolate(x, size=(64,64), mode='bilinear', align_corners=False)
            x = x.squeeze(0) / 2.0; x = x.repeat(3,1,1)
            k = torch.randint(0,4,(1,)).item()
            x = torch.rot90(x, k, [1,2])
            if torch.rand(1).item()>0.5: x = torch.flip(x,[2])
            x = (x + torch.randn_like(x)*0.02).clamp(0,1)
            train_t.append(x); train_l.append(ci)
    print(f"  {CLASSES[ci]:12s}: {n_rows:5d} -> {TARGET}")

del df, train_by_class; gc.collect()
train_data = torch.stack(train_t); del train_t; gc.collect()
train_labels = torch.tensor(train_l, dtype=torch.int64); del train_l
print(f"\n  Train: {train_data.shape}  Val: {val_data.shape}  Test: {test_data.shape}")

# ============================================================
# CELL 4: TRAIN — Focal Loss (γ=2) + Mixup (α=0.4), 60 epochs
# ============================================================
print("\n[3/5] Training: Focal(γ=2.0) + Mixup(α=0.4), 60 epochs...")

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
            x = torch.rot90(x, torch.randint(0,4,(1,)).item(), [1,2])
            if torch.rand(1).item()>0.5: x = torch.flip(x,[2])
            if torch.rand(1).item()>0.5: x = torch.flip(x,[1])
            x = (x + torch.randn_like(x)*0.01).clamp(0,1)
        return x, self.labels[i]

cc = Counter(train_labels.numpy().tolist())
sampler = WeightedRandomSampler([1.0/cc[l.item()] for l in train_labels], len(train_labels), replacement=True)
train_loader = DataLoader(WaferDS(train_data,train_labels,aug=True), batch_size=64, sampler=sampler, num_workers=2, pin_memory=True)
val_loader = DataLoader(WaferDS(val_data,val_labels), batch_size=128, num_workers=2, pin_memory=True)
test_loader = DataLoader(WaferDS(test_data,test_labels), batch_size=128, num_workers=2, pin_memory=True)

# Model
backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
backbone.fc = nn.Identity()
backbone = backbone.to(device)
classifier = nn.Linear(2048, 8).to(device)

counts_t = torch.tensor([cc.get(i,1) for i in range(8)], dtype=torch.float32)
alpha_w = ((1.0/counts_t.clamp(min=1)) / (1.0/counts_t.clamp(min=1)).sum() * 8).to(device)
criterion = FocalLoss(gamma=2.0, alpha=alpha_w)

EPOCHS, LR, MIXUP_ALPHA = 60, 5e-5, 0.4
optimizer = torch.optim.Adam(list(backbone.parameters())+list(classifier.parameters()), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=55, eta_min=1e-7)
best_val_loss, best_state = float("inf"), None
t0 = time.time()

for epoch in range(EPOCHS):
    if epoch < 5:
        for pg in optimizer.param_groups: pg["lr"] = LR * (epoch+1)/5
    backbone.train(); classifier.train()
    t_loss, correct, total = 0.0, 0, 0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        if torch.rand(1).item() < 0.5:
            imgs, la, lb, lam = mixup_batch(imgs, lbls, MIXUP_ALPHA)
            logits = classifier(backbone(imgs))
            loss = lam*criterion(logits,la) + (1-lam)*criterion(logits,lb)
            correct += lam*(logits.argmax(1)==la).sum().item() + (1-lam)*(logits.argmax(1)==lb).sum().item()
        else:
            logits = classifier(backbone(imgs))
            loss = criterion(logits, lbls)
            correct += (logits.argmax(1)==lbls).sum().item()
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
        optimizer.step()
        t_loss += loss.item()*imgs.size(0); total += imgs.size(0)
    if epoch >= 5: scheduler.step()

    backbone.eval(); classifier.eval()
    v_loss, v_total = 0.0, 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            l = criterion(classifier(backbone(imgs)), lbls)
            v_loss += l.item()*imgs.size(0); v_total += imgs.size(0)
    v_loss /= v_total
    if v_loss < best_val_loss:
        best_val_loss = v_loss
        best_state = {"backbone":{k:v.cpu() for k,v in backbone.state_dict().items()},
                      "classifier":{k:v.cpu() for k,v in classifier.state_dict().items()},"epoch":epoch}
    if (epoch+1)%10==0 or epoch==0:
        print(f"  Ep {epoch+1:2d}/{EPOCHS} | t_loss={t_loss/total:.4f} acc={correct/total:.3f} | v_loss={v_loss:.4f} | lr={optimizer.param_groups[0]['lr']:.1e}")

elapsed = time.time() - t0
print(f"\n  Training done in {elapsed/60:.1f} min | Best epoch: {best_state['epoch']+1}")

# ============================================================
# CELL 5: EVALUATE — KNN@5 (honest: train->index, test->query)
# ============================================================
print("\n[4/5] KNN@5 evaluation...")
backbone.load_state_dict({k:v.to(device) for k,v in best_state["backbone"].items()})
backbone.eval()

def get_emb(loader):
    E, L = [], []
    with torch.no_grad():
        for x,y in loader:
            E.append(backbone(x.to(device)).cpu()); L.append(y)
    return torch.cat(E).numpy(), torch.cat(L).numpy()

train_emb, train_lab = get_emb(train_loader)
test_emb, test_lab = get_emb(test_loader)

knn = KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=-1)
knn.fit(train_emb, train_lab)
pred = knn.predict(test_emb)

print("\n" + "="*60)
print("  KNN@5 Results — Focal Loss + Mixup (Honest Eval)")
print("="*60)
print(classification_report(test_lab, pred, target_names=CLASSES, digits=3))

macro = recall_score(test_lab, pred, average="macro")
print(f"{'='*60}")
print(f"  >>> MACRO KNN@5: {macro*100:.1f}% <<<")
print(f"{'='*60}")

# ============================================================
# DONE
# ============================================================
print(f"\n[5/5] Summary:")
print(f"  Method: ResNet50 + Focal(γ=2.0) + Mixup(α=0.4)")
print(f"  Split: Lot-based 70/15/15 (no leakage)")
print(f"  Train: 4K/class balanced | Epochs: 60 | LR: {LR}")
print(f"  Best epoch: {best_state['epoch']+1}")
print(f"  Macro KNN@5: {macro*100:.1f}%")
print(f"  Time: {elapsed/60:.1f} min on {torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}")
print(f"\n  Repo: https://github.com/haykbaghdasaryan-sketch/wafer-vision")
print(f"  Branch: feat/focal-mixup")

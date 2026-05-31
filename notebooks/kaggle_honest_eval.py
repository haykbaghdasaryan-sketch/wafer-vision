"""
WaferVision — Kaggle Training Script (Honest Evaluation)
=========================================================
Platform: Kaggle Notebook with GPU T4, 30 GB RAM
Dataset: WM-811K (add as Input from Kaggle Datasets)
Result: ~90.8% macro KNN@5 on 8 defect classes

Usage on Kaggle:
1. Create new Notebook, add WM-811K dataset as Input
2. Select GPU T4, Internet ON
3. Copy-paste this entire file into a single cell
4. Run (takes ~20 min)

Methodology:
- Lot-based split (no manufacturing lot leakage)
- Oversampling only train set (with augmentation)
- KNN index built on train, queries from test
- 8 defect classes only (no "none" — handled by anomaly detection)
"""

# ============================================================
# SETUP
# ============================================================
import os, subprocess, sys

REPO = '/kaggle/working/wafer-vision'
if not os.path.exists(REPO):
    subprocess.run(['git', 'clone',
        'https://github.com/haykbaghdasaryan-sketch/wafer-vision.git', REPO], check=True)
    subprocess.run('pip install hydra-core omegaconf pydantic structlog tqdm grad-cam umap-learn psutil -q'.split(), check=True)

sys.path.insert(0, REPO)
os.chdir(REPO)

# Fix src imports (Kaggle editable install has numpy conflicts)
import importlib.util
for pkg in ['src', 'src.data', 'src.models', 'src.training', 'src.evaluation', 'src.anomaly', 'src.visualization']:
    parts = pkg.split('.')
    path = os.path.join(REPO, *parts, '__init__.py')
    if os.path.exists(path):
        spec = importlib.util.spec_from_file_location(pkg, path,
            submodule_search_locations=[os.path.join(REPO, *parts)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[pkg] = mod
        spec.loader.exec_module(mod)

# Verify
from src.data.augmentation import WaferAugmentation
from src.models import get_backbone
from src.training.trainer import Trainer, TrainingConfig
import torch
print(f"Setup OK | GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# LOAD DATASET (defects only, no "none")
# ============================================================
import pickle, gc, time, numpy as np, torch.nn.functional as F
from collections import defaultdict
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

class CompatUnpickler(pickle.Unpickler):
    """Handles old pandas pickle format (WM-811K was pickled with pandas ~0.23)."""
    def find_class(self, module, name):
        if module.startswith('pandas.indexes'):
            module = module.replace('pandas.indexes', 'pandas.core.indexes')
        if module == 'pandas.core.internals' and name == 'BlockManager':
            module = 'pandas.core.internals.managers'
        return super().find_class(module, name)

# Find dataset path (varies by how you added it)
PKL_PATHS = [
    "/kaggle/input/datasets/qingyi/wm811k-wafer-map/LSWMD.pkl",
    "/kaggle/input/wm811k-wafer-map/LSWMD.pkl",
]
pkl_path = next((p for p in PKL_PATHS if os.path.exists(p)), None)
assert pkl_path, f"Dataset not found! Tried: {PKL_PATHS}"

print(f"\n[1/4] Loading WM-811K from {pkl_path}...")
with open(pkl_path, "rb") as f:
    df = CompatUnpickler(f, encoding='latin1').load()
print(f"  {len(df):,} records")

# Parse: 8 defect classes only
CLASSES = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch']
class_to_idx = {c: i for i, c in enumerate(CLASSES)}
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
print(f"  {total:,} defects in {len(lot_indices):,} lots")

# ============================================================
# LOT-BASED SPLIT (honest, no leakage)
# ============================================================
print("\n[2/4] Lot-based split + balanced train...")
np.random.seed(42)
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

# Process val/test
def make_tensors(ic_list):
    t, l = [], []
    for row_idx, ci in ic_list:
        wm = df.iloc[row_idx]["waferMap"]
        x = torch.from_numpy(wm.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        x = F.interpolate(x, size=(64,64), mode='bilinear', align_corners=False)
        x = x.squeeze(0) / 2.0; x = x.repeat(3,1,1)
        t.append(x); l.append(ci)
    return torch.stack(t), torch.tensor(l, dtype=torch.int64)

test_data, test_labels = make_tensors(test_ic); del test_ic
val_data, val_labels = make_tensors(val_ic); del val_ic

# Balanced train: 4K/class with augmentation
aug = WaferAugmentation(strength="heavy", noise_std=0.03)
TARGET = 4000
train_by_class = defaultdict(list)
for row_idx, ci in train_ic: train_by_class[ci].append(row_idx)
del train_ic

train_t, train_l = [], []
for ci in range(8):
    rows = train_by_class[ci]; n = len(rows)
    for r in rows[:TARGET]:
        wm = df.iloc[r]["waferMap"]
        x = torch.from_numpy(wm.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        x = F.interpolate(x, size=(64,64), mode='bilinear', align_corners=False)
        x = x.squeeze(0) / 2.0; x = x.repeat(3,1,1)
        train_t.append(x); train_l.append(ci)
    if n < TARGET:
        for _ in range(TARGET - n):
            r = rows[np.random.randint(0, n)]
            wm = df.iloc[r]["waferMap"]
            x = torch.from_numpy(wm.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            x = F.interpolate(x, size=(64,64), mode='bilinear', align_corners=False)
            x = x.squeeze(0) / 2.0; x = x.repeat(3,1,1)
            train_t.append(aug(x)); train_l.append(ci)
    print(f"  {CLASSES[ci]}: {n} → {TARGET}")

del df, train_by_class; gc.collect()
train_data = torch.stack(train_t); del train_t; gc.collect()
train_labels = torch.tensor(train_l, dtype=torch.int64); del train_l
print(f"  Train:{len(train_data):,} Val:{len(val_data):,} Test:{len(test_data):,}")

# ============================================================
# TRAIN: SupCon (60 epochs)
# ============================================================
print("\n[3/4] SupCon training (60 epochs)...")
from src.data.sampler import BalancedBatchSampler
import src.training.losses as losses_mod

# Fix grad issue
_orig = losses_mod.SupervisedContrastiveLoss.forward
def _fix(self, f, l):
    loss = _orig(self, f, l)
    if not loss.requires_grad: loss = loss + f.sum() * 0
    return loss
losses_mod.SupervisedContrastiveLoss.forward = _fix

model = get_backbone("resnet50", pretrained=True)
sampler = BalancedBatchSampler(train_labels.tolist(), p_classes=8, k_samples=8)
tl = DataLoader(TensorDataset(train_data, train_labels), batch_sampler=sampler, num_workers=2, pin_memory=True)
vl = DataLoader(TensorDataset(val_data, val_labels), batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

config = TrainingConfig(mode="metric", loss_name="supcon", num_epochs=60,
    learning_rate=5e-5, weight_decay=1e-4, warmup_epochs=5, patience=20,
    temperature=0.07, projection_hidden=512, projection_output=128,
    device="cuda", checkpoint_dir="outputs/ckpt", log_every_n_steps=300)

trainer = Trainer(config=config, model=model, train_loader=tl, val_loader=vl)
t0 = time.time()
result = trainer.train()
print(f"  Done {time.time()-t0:.0f}s | best epoch: {result.best_epoch}")

# ============================================================
# HONEST EVALUATION (train→index, test→query)
# ============================================================
print("\n[4/4] Honest evaluation...")
from src.models.embedding_extractor import EmbeddingExtractor
from sklearn.neighbors import NearestNeighbors

ext = EmbeddingExtractor(backbone=model, batch_size=128, device="cuda")
ext.extract(DataLoader(TensorDataset(train_data, train_labels), batch_size=128, shuffle=False, num_workers=2),
            Path("outputs/train_emb.npy"), metadata={})
ext.extract(DataLoader(TensorDataset(test_data, test_labels), batch_size=128, shuffle=False, num_workers=2),
            Path("outputs/test_emb.npy"), metadata={})

train_emb = np.load("outputs/train_emb.npy")
test_emb = np.load("outputs/test_emb.npy")
nn = NearestNeighbors(n_neighbors=5, metric='euclidean', algorithm='brute')
nn.fit(train_emb)
_, indices = nn.kneighbors(test_emb)
neighbor_labels = train_labels.numpy()[indices]
test_lbl = test_labels.numpy()

print(f"\n{'Class':<12} {'KNN@5':<10} {'P@5':<10} {'n_test':<8}")
print("=" * 40)
knn5s = []
for ci in range(8):
    mask = test_lbl == ci
    if not mask.any(): knn5s.append(0); continue
    preds = np.array([np.bincount(row, minlength=8).argmax() for row in neighbor_labels[mask]])
    acc = (preds == ci).mean()
    p5 = (neighbor_labels[mask] == ci).mean()
    knn5s.append(acc)
    print(f"  {CLASSES[ci]:<10} {acc*100:<10.1f} {p5*100:<10.1f} {mask.sum():<8}")

macro = np.mean(knn5s)
print(f"\n{'='*40}")
print(f"Macro KNN@5: {macro*100:.1f}%")
print(f"\nMethodology: lot-split | train→index test→query | 8 defect classes")
print(f"DONE")

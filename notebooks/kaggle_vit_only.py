"""
WaferVision — ViT-B/16 ONLY + Final Summary (Kaggle, memory-safe)
==================================================================
Use this when ResNet50 and EfficientNet-B0 already finished in a previous
run (or crashed the summary). This cell:
  1. Frees ALL leftover GPU memory from prior models.
  2. Reuses the in-memory tensors (train_data/val_data/test_data) if they
     still exist; otherwise rebuilds them from scratch (works after a
     kernel restart too).
  3. Trains ONLY ViT-B/16 with real gradient checkpointing + batch=32.
  4. Prints the FINAL comparison table using the known ResNet50 /
     EfficientNet-B0 numbers (set below) plus the fresh ViT result.

Paste into ONE cell and Run. ~2-4 h for ViT alone on a T4.
"""

# ---- Known results from the completed runs (edit if yours differ) ----
RESNET50_KNN5 = 90.7   # macro KNN@5 %
EFFNET_KNN5   = 90.8   # macro KNN@5 %

# ============================================================
# 0. FREE ANY LEFTOVER GPU MEMORY FROM PRIOR MODELS
# ============================================================
import gc, os, sys, time, pickle
try:
    import torch
    for _v in ['backbone_r50','classifier_r50','backbone_efn','classifier_efn',
               'backbone_vit','classifier_vit']:
        if _v in dir():
            try: del globals()[_v]
            except Exception: pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
except ImportError:
    pass

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.checkpoint import checkpoint_sequential
from collections import Counter, defaultdict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, recall_score, accuracy_score

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == 'cuda':
    free_b = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
    print(f"VRAM free: {free_b/1024**3:.1f} GB")

CLASSES = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Near-full','Random','Scratch']
class_to_idx = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

# ============================================================
# 1. REUSE IN-MEMORY TENSORS, OR REBUILD IF MISSING
# ============================================================
need_rebuild = not all(v in dir() for v in
                       ['train_data','train_labels','val_data','val_labels','test_data','test_labels'])

if need_rebuild:
    print("\nTensors not in memory — rebuilding from LSWMD.pkl (lot-based split)...")

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
    assert pkl_path, "Dataset not found! Add 'wm811k-wafer-map' as Input."
    with open(pkl_path, "rb") as f:
        df = CompatUnpickler(f, encoding='latin1').load()
    print(f"  Total records: {len(df):,}")

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

    np.random.seed(SEED)
    lots = list(lot_indices.keys()); np.random.shuffle(lots)
    n = len(lots)
    train_lots = set(lots[:int(0.7*n)]); val_lots = set(lots[int(0.7*n):int(0.85*n)])
    train_ic, val_ic, test_ic = [], [], []
    for lot, items in lot_indices.items():
        for item in items:
            if lot in train_lots: train_ic.append(item)
            elif lot in val_lots: val_ic.append(item)
            else: test_ic.append(item)
    del lot_indices; gc.collect()
    print(f"  Split: train={len(train_ic):,} val={len(val_ic):,} test={len(test_ic):,}")

    def make_tensors(ic_list):
        t, l = [], []
        for row_idx, ci in ic_list:
            wm = df.iloc[row_idx]["waferMap"]
            x = torch.from_numpy(wm.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            x = F.interpolate(x, size=(96, 96), mode='bilinear', align_corners=False)
            x = x.squeeze(0) / 2.0; x = x.repeat(3, 1, 1)
            t.append(x); l.append(ci)
        return torch.stack(t), torch.tensor(l, dtype=torch.int64)

    val_data, val_labels = make_tensors(val_ic); del val_ic
    test_data, test_labels = make_tensors(test_ic); del test_ic

    train_by_class = defaultdict(list)
    for row_idx, ci in train_ic: train_by_class[ci].append(row_idx)
    del train_ic
    TARGET = max(len(v) for v in train_by_class.values())
    train_t, train_l = [], []
    for ci in range(NUM_CLASSES):
        rows = train_by_class[ci]; n_rows = len(rows)
        for r in rows:
            wm = df.iloc[r]["waferMap"]
            x = torch.from_numpy(wm.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            x = F.interpolate(x, size=(96, 96), mode='bilinear', align_corners=False)
            x = x.squeeze(0) / 2.0; x = x.repeat(3, 1, 1)
            train_t.append(x); train_l.append(ci)
        for j in range(TARGET - n_rows):
            r = rows[j % n_rows]
            wm = df.iloc[r]["waferMap"]
            x = torch.from_numpy(wm.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            x = F.interpolate(x, size=(96, 96), mode='bilinear', align_corners=False)
            x = x.squeeze(0) / 2.0; x = x.repeat(3, 1, 1)
            x = torch.rot90(x, torch.randint(0, 4, (1,)).item(), [1, 2])
            if torch.rand(1).item() > 0.5: x = torch.flip(x, [2])
            if torch.rand(1).item() > 0.5: x = torch.flip(x, [1])
            x = (x + torch.randn_like(x) * 0.03).clamp(0, 1)
            train_t.append(x); train_l.append(ci)
    del df, train_by_class; gc.collect()
    train_data = torch.stack(train_t); del train_t
    train_labels = torch.tensor(train_l, dtype=torch.int64); del train_l; gc.collect()
else:
    print("\nReusing in-memory tensors from previous cells.")

print(f"  Train={tuple(train_data.shape)} Val={tuple(val_data.shape)} Test={tuple(test_data.shape)}")

# ============================================================
# 2. SHARED COMPONENTS
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)
    def forward(self, logits, labels):
        probs = F.softmax(logits, dim=1)
        p_t = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        log_p_t = torch.log(p_t + 1e-8)
        loss = -((1.0 - p_t) ** self.gamma) * log_p_t
        if self.alpha is not None:
            loss = self.alpha[labels] * loss
        return loss.mean()

def mixup_batch(images, labels, alpha=0.4):
    lam = np.random.beta(alpha, alpha); lam = max(lam, 1.0 - lam)
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
            x = (x + torch.randn_like(x) * 0.015).clamp(0, 1)
        return x, self.labels[i]

cc = Counter(train_labels.numpy().tolist())
counts_t = torch.tensor([cc.get(i, 1) for i in range(NUM_CLASSES)], dtype=torch.float32)
inv_freq = 1.0 / counts_t.clamp(min=1)
alpha_w = (inv_freq / inv_freq.sum() * NUM_CLASSES).to(device)

val_loader  = DataLoader(WaferDS(val_data, val_labels), batch_size=128, num_workers=2, pin_memory=True)
test_loader = DataLoader(WaferDS(test_data, test_labels), batch_size=128, num_workers=2, pin_memory=True)
train_eval_loader = DataLoader(WaferDS(train_data, train_labels, aug=False),
                               batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

# ============================================================
# 3. ViT-B/16 WITH REAL GRADIENT CHECKPOINTING
# ============================================================
class ViTWrapper(nn.Module):
    def __init__(self, ckpt_segments=4):
        super().__init__()
        self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.model.heads = nn.Identity()
        self.ckpt_segments = ckpt_segments
    def forward(self, x):
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.model._process_input(x)
        n = x.shape[0]
        cls = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([cls, x], dim=1)
        enc = self.model.encoder
        x = enc.dropout(x + enc.pos_embedding)
        if self.training and x.requires_grad:
            x = checkpoint_sequential(enc.layers, self.ckpt_segments, x, use_reentrant=False)
        else:
            x = enc.layers(x)
        x = enc.ln(x)
        return x[:, 0]

def train_vit(backbone, classifier, epochs=120, lr=1e-4, warmup=15, patience=25, batch_size=32):
    criterion = FocalLoss(gamma=2.0, alpha=alpha_w)
    MIXUP_ALPHA, MIXUP_P = 0.4, 0.5
    sampler = WeightedRandomSampler([1.0/cc[l.item()] for l in train_labels],
                                    len(train_labels), replacement=True)
    loader = DataLoader(WaferDS(train_data, train_labels, aug=True), batch_size=batch_size,
                        sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    opt = torch.optim.AdamW([
        {'params': backbone.parameters(), 'lr': lr * 0.1},
        {'params': classifier.parameters(), 'lr': lr}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - warmup, eta_min=1e-7)
    best_loss, best_state, pat = float("inf"), None, 0
    t0 = time.time()
    print(f"\n  [ViT-B/16] epochs={epochs} lr={lr} batch={batch_size} (grad-checkpoint)")
    for epoch in range(epochs):
        if epoch < warmup:
            s = (epoch+1)/warmup
            opt.param_groups[0]["lr"] = lr*0.1*s; opt.param_groups[1]["lr"] = lr*s
        backbone.train(); classifier.train()
        tl, corr, tot = 0.0, 0, 0
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
            if torch.rand(1).item() < MIXUP_P:
                imgs, la, lb, lam = mixup_batch(imgs, lbls, MIXUP_ALPHA)
                logits = classifier(backbone(imgs))
                loss = lam*criterion(logits, la) + (1-lam)*criterion(logits, lb)
                preds = logits.argmax(1)
                corr += lam*(preds==la).sum().item() + (1-lam)*(preds==lb).sum().item()
            else:
                logits = classifier(backbone(imgs))
                loss = criterion(logits, lbls)
                corr += (logits.argmax(1)==lbls).sum().item()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0); opt.step()
            tl += loss.item()*imgs.size(0); tot += imgs.size(0)
        if epoch >= warmup: sched.step()
        backbone.eval(); classifier.eval()
        vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
                logits = classifier(backbone(imgs))
                vl += criterion(logits, lbls).item()*imgs.size(0)
                vc += (logits.argmax(1)==lbls).sum().item(); vt += imgs.size(0)
        vl /= vt; va = vc/vt
        if vl < best_loss:
            best_loss = vl; pat = 0
            best_state = {"backbone": {k:v.cpu() for k,v in backbone.state_dict().items()},
                          "classifier": {k:v.cpu() for k,v in classifier.state_dict().items()},
                          "epoch": epoch, "val_loss": vl, "val_acc": va}
        else:
            pat += 1
        if (epoch+1) % 5 == 0 or epoch == 0:
            el = time.time()-t0; eta = el/(epoch+1)*(epochs-epoch-1)
            print(f"    Ep {epoch+1:3d}/{epochs} | t_loss={tl/tot:.4f} t_acc={corr/tot:.3f} | "
                  f"v_loss={vl:.4f} v_acc={va:.3f} | pat={pat}/{patience} | ETA={eta/60:.0f}m")
        if pat >= patience:
            print(f"    Early stop at epoch {epoch+1}"); break
    el = time.time()-t0
    print(f"    Done: {el/60:.1f}min | best epoch {best_state['epoch']+1} v_loss={best_state['val_loss']:.4f}")
    return best_state, el

def evaluate(backbone, best_state, name="ViT-B/16"):
    backbone.load_state_dict({k:v.to(device) for k,v in best_state["backbone"].items()})
    backbone.eval()
    def emb(loader):
        E, L = [], []
        with torch.no_grad():
            for x, y in loader:
                E.append(backbone(x.to(device)).cpu()); L.append(y)
        return torch.cat(E).numpy(), torch.cat(L).numpy()
    tr_e, tr_l = emb(train_eval_loader)
    te_e, te_l = emb(test_loader)
    res = {}
    for K in [1, 3, 5, 10]:
        knn = KNeighborsClassifier(n_neighbors=K, metric="cosine", n_jobs=-1)
        knn.fit(tr_e, tr_l); pred = knn.predict(te_e)
        res[f"KNN@{K}"] = recall_score(te_l, pred, average="macro")
        if K == 5:
            res["report"] = classification_report(te_l, pred, target_names=CLASSES, digits=3)
            res["per_class"] = {}
            for i, c in enumerate(CLASSES):
                m = te_l == i
                if m.sum() > 0: res["per_class"][c] = (accuracy_score(te_l[m], pred[m]), int(m.sum()))
    print(f"\n  {name} — KNN@5: {res['KNN@5']*100:.1f}%")
    print(res["report"])
    return res

# ============================================================
# 4. RUN ViT (memory-safe) + FINAL OUTPUT
# ============================================================
if device.type == 'cuda':
    torch.cuda.empty_cache(); gc.collect()

try:
    backbone_vit = ViTWrapper().to(device)
    classifier_vit = nn.Linear(768, NUM_CLASSES).to(device)
    best_vit, time_vit = train_vit(backbone_vit, classifier_vit)
    results_vit = evaluate(backbone_vit, best_vit, "ViT-B/16")
    vit_ok = True
    torch.save(best_vit, "vit_b16_focal_mixup_best.pth")
except torch.cuda.OutOfMemoryError:
    print("\n  ViT OOM even with gradient checkpointing + batch=32.")
    print("  ResNet50 / EfficientNet-B0 remain the reported models.")
    results_vit = {"KNN@5": 0.0}; time_vit = 0; vit_ok = False

# ============================================================
# 5. FINAL COMPARISON (robust)
# ============================================================
print("\n" + "="*60)
print("  FINAL RESULTS — Focal+Mixup, Lot-Split, Honest Eval")
print("="*60)
print(f"\n  {'Model':<22}{'Macro KNN@5':<14}{'Status'}")
print("  " + "-"*46)
print(f"  {'ResNet50':<22}{str(RESNET50_KNN5)+'%':<14}{'done'}")
print(f"  {'EfficientNet-B0':<22}{str(EFFNET_KNN5)+'%':<14}{'done  (best)'}")
if vit_ok:
    vit_pct = f"{results_vit['KNN@5']*100:.1f}%"
    print(f"  {'ViT-B/16':<22}{vit_pct:<14}{f'{time_vit/60:.0f}min'}")
else:
    print(f"  {'ViT-B/16':<22}{'—':<14}{'OOM (skipped)'}")
print("\n  Methodology: lot-based split, oversample train only,")
print("  KNN index=train / query=test, 8 defect classes (no 'none').")
print("="*60)
print("\n  Repo: https://github.com/haykbaghdasaryan-sketch/wafer-vision")

"""
WaferVision — Focal Loss + Mixup, 3 Backbones, FULL DATA (Kaggle, single cell)
==============================================================================

WHAT THIS DOES
--------------
Trains three backbones on the FULL WM-811K defect set and evaluates each with
an honest, lot-based retrieval protocol:

    A) ResNet50        + Focal(gamma=2.0) + Mixup(alpha=0.4)
    B) EfficientNet-B0 + Focal(gamma=2.0) + Mixup(alpha=0.4)
    C) ViT-B/16        + Focal(gamma=2.0) + Mixup(alpha=0.4)

It prints a final comparison table and writes every result to
`focal_mixup_results.json` as each model finishes.

HOW TO RUN (Kaggle)
-------------------
1. Kaggle -> New Notebook.
2. Add Data (Input): search "wm811k-wafer-map" by qingyi.
3. Settings: Accelerator = GPU T4, Internet = ON.
4. Paste this whole file into ONE cell -> Run All.

WHY IT IS RELIABLE
------------------
- Mixed precision (AMP) halves GPU memory and roughly doubles speed on a T4,
  so ViT-B/16 at 224x224 actually fits.
- ViT additionally uses gradient checkpointing + a small batch size.
- GPU memory is fully released between models.
- Results are saved to disk after EACH model. If the Kaggle 12h limit is hit
  during the (slowest) ViT stage, the ResNet50 + EfficientNet numbers are
  already saved and the summary still prints.

RUNTIME
-------
~3-6 h total on a single T4 (CNNs are fast with AMP; ViT is the long pole).
Kaggle GPU sessions last up to 12 h, so a full run fits with margin.
"""

# ============================================================
# CONFIG — the only knobs you normally touch
# ============================================================
SEED            = 42
TARGET_SIZE     = 96      # wafer resolution for CNNs (ViT upsamples to 224 internally)
EPOCHS          = 150     # max epochs per model (early stopping usually ends sooner)
PATIENCE        = 30      # early-stop patience (epochs without val-loss improvement)
USE_AMP         = True    # mixed precision: faster + less memory on T4

# Per-backbone batch sizes (ViT is smaller because 224x224 is memory-heavy)
BATCH_CNN       = 128
BATCH_VIT       = 32
EVAL_BATCH_CNN  = 256
EVAL_BATCH_VIT  = 64

RESULTS_JSON    = "focal_mixup_results.json"

# ============================================================
# 0. SETUP
# ============================================================
import os, sys, gc, time, json, pickle, subprocess

REPO = "/kaggle/working/wafer-vision"
if not os.path.exists(REPO):
    subprocess.run(["git", "clone", "-b", "feat/focal-mixup",
        "https://github.com/haykbaghdasaryan-sketch/wafer-vision.git", REPO], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
        "hydra-core", "omegaconf", "pydantic", "structlog", "tqdm",
        "grad-cam", "umap-learn", "psutil", "faiss-cpu"], check=False)
if os.path.exists(REPO):
    sys.path.insert(0, REPO)
    os.chdir(REPO)

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.checkpoint import checkpoint_sequential
from collections import Counter, defaultdict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, recall_score, accuracy_score

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | GPU: {torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}")
if device.type == "cuda":
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB | AMP: {USE_AMP}")

CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch"]
class_to_idx = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)


# ============================================================
# 1. LOAD WM-811K
# ============================================================
class CompatUnpickler(pickle.Unpickler):
    """Reads the legacy pandas pickle (old module paths) used by LSWMD.pkl."""
    def find_class(self, module, name):
        if module.startswith("pandas.indexes"):
            module = module.replace("pandas.indexes", "pandas.core.indexes")
        if module == "pandas.core.internals" and name == "BlockManager":
            module = "pandas.core.internals.managers"
        return super().find_class(module, name)


PKL_PATHS = [
    "/kaggle/input/datasets/qingyi/wm811k-wafer-map/LSWMD.pkl",
    "/kaggle/input/wm811k-wafer-map/LSWMD.pkl",
    "/kaggle/input/wm811k/LSWMD.pkl",
]
pkl_path = next((p for p in PKL_PATHS if os.path.exists(p)), None)
assert pkl_path, "Dataset not found! Add 'wm811k-wafer-map' (by qingyi) as a Kaggle Input."

print(f"\n[1/7] Loading WM-811K from {pkl_path} ...")
with open(pkl_path, "rb") as f:
    df = CompatUnpickler(f, encoding="latin1").load()
print(f"  Total records: {len(df):,}")


# ============================================================
# 2. PARSE LABELS + LOT-BASED SPLIT (no leakage)
# ============================================================
print("\n[2/7] Parsing labeled defects + lot-based split (70/15/15) ...")
lot_indices = defaultdict(list)
for idx in range(len(df)):
    row = df.iloc[idx]
    wm = row.get("waferMap", None)
    if wm is None or not hasattr(wm, "shape") or wm.ndim != 2:
        continue
    if wm.shape[0] < 5 or wm.shape[1] < 5:
        continue
    raw = row.get("failureType", None)
    for _ in range(5):  # unwrap nested arrays/lists that WM-811K uses
        if isinstance(raw, np.ndarray):
            if raw.size == 0:
                raw = None; break
            raw = raw.flat[0]
        elif isinstance(raw, list):
            if len(raw) == 0:
                raw = None; break
            raw = raw[0]
        else:
            break
    if raw is None:
        continue
    label = str(raw).strip()
    if label not in class_to_idx:
        continue
    lot = row.get("lotName", f"unk_{idx}")
    if lot is None or (isinstance(lot, float) and np.isnan(lot)):
        lot = f"unk_{idx}"
    lot_indices[lot].append((idx, class_to_idx[label]))

total_defects = sum(len(v) for v in lot_indices.values())
print(f"  {total_defects:,} defects across {len(lot_indices):,} lots")

np.random.seed(SEED)
lots = list(lot_indices.keys())
np.random.shuffle(lots)
n = len(lots)
train_lots = set(lots[:int(0.70 * n)])
val_lots = set(lots[int(0.70 * n):int(0.85 * n)])

train_ic, val_ic, test_ic = [], [], []
for lot, items in lot_indices.items():
    bucket = train_ic if lot in train_lots else (val_ic if lot in val_lots else test_ic)
    bucket.extend(items)
del lot_indices
gc.collect()
print(f"  Split: train={len(train_ic):,}  val={len(val_ic):,}  test={len(test_ic):,}")

# --- Leakage guard #1: no lot may appear in more than one split ---
test_lots = set(lots) - train_lots - val_lots
assert train_lots.isdisjoint(val_lots), "LEAK: lot(s) shared between train and val!"
assert train_lots.isdisjoint(test_lots), "LEAK: lot(s) shared between train and test!"
assert val_lots.isdisjoint(test_lots), "LEAK: lot(s) shared between val and test!"
print(f"  [leak-check] lot disjointness OK "
      f"(train={len(train_lots)} val={len(val_lots)} test={len(test_lots)} lots, no overlap)")


# ============================================================
# 3. BUILD TENSORS (FULL DATA, no subsampling)
# ============================================================
print(f"\n[3/7] Building {TARGET_SIZE}x{TARGET_SIZE} tensors for ALL samples ...")


def _to_tensor(wafer_map):
    x = torch.from_numpy(wafer_map.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    x = F.interpolate(x, size=(TARGET_SIZE, TARGET_SIZE), mode="bilinear", align_corners=False)
    x = x.squeeze(0) / 2.0          # normalize {0,1,2} -> {0, .5, 1}
    return x.repeat(3, 1, 1)        # 1ch -> 3ch for ImageNet backbones


def make_tensors(ic_list, desc=""):
    t, l = [], []
    for i, (row_idx, ci) in enumerate(ic_list):
        t.append(_to_tensor(df.iloc[row_idx]["waferMap"]))
        l.append(ci)
        if (i + 1) % 5000 == 0:
            print(f"    {desc}: {i + 1}/{len(ic_list)}")
    return torch.stack(t), torch.tensor(l, dtype=torch.int64)


val_data, val_labels = make_tensors(val_ic, "val"); del val_ic
test_data, test_labels = make_tensors(test_ic, "test"); del test_ic

# Train: keep ALL originals, then oversample minorities (with augmentation) to the largest class.
train_by_class = defaultdict(list)
for row_idx, ci in train_ic:
    train_by_class[ci].append(row_idx)
del train_ic

TARGET = max(len(rows) for rows in train_by_class.values())
print(f"\n  Balancing train to {TARGET}/class:")
train_t, train_l = [], []
for ci in range(NUM_CLASSES):
    rows = train_by_class[ci]
    n_rows = len(rows)
    for r in rows:                               # all original samples
        train_t.append(_to_tensor(df.iloc[r]["waferMap"]))
        train_l.append(ci)
    for j in range(TARGET - n_rows):             # augmented oversamples
        x = _to_tensor(df.iloc[rows[j % n_rows]]["waferMap"])
        x = torch.rot90(x, torch.randint(0, 4, (1,)).item(), [1, 2])
        if torch.rand(1).item() > 0.5: x = torch.flip(x, [2])
        if torch.rand(1).item() > 0.5: x = torch.flip(x, [1])
        x = (x + torch.randn_like(x) * 0.03).clamp(0, 1)
        train_t.append(x)
        train_l.append(ci)
    print(f"    {CLASSES[ci]:12s}: {n_rows:5d} -> {TARGET}")

del df, train_by_class
gc.collect()
train_data = torch.stack(train_t); del train_t
train_labels = torch.tensor(train_l, dtype=torch.int64); del train_l
gc.collect()
print(f"\n  Train={tuple(train_data.shape)}  Val={tuple(val_data.shape)}  Test={tuple(test_data.shape)}")

# --- Leakage guard #2: no identical wafer tensor shared between train and test ---
# Hash each preprocessed wafer (bytes of the rounded tensor) and check set intersection.
# This catches any duplicate wafer that might have slipped across the split.
def _tensor_hashes(data):
    import hashlib
    hashes = set()
    arr = (data.numpy() * 255).round().astype(np.uint8)  # quantize to ignore fp noise
    for i in range(arr.shape[0]):
        hashes.add(hashlib.md5(arr[i].tobytes()).hexdigest())
    return hashes

_train_hashes = _tensor_hashes(train_data)
_test_hashes = _tensor_hashes(test_data)
_dupes = _train_hashes & _test_hashes
assert len(_dupes) == 0, f"LEAK: {len(_dupes)} identical wafer(s) appear in BOTH train and test!"
print(f"  [leak-check] no duplicate wafers across train/test "
      f"({len(_train_hashes)} unique train, {len(_test_hashes)} unique test, 0 shared)")
del _train_hashes, _test_hashes, _dupes
gc.collect()


# ============================================================
# 4. SHARED COMPONENTS
# ============================================================
class FocalLoss(nn.Module):
    """Class-balanced focal loss. Computed in fp32 for AMP stability."""
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)

    def forward(self, logits, labels):
        logits = logits.float()                      # fp32 for numerical stability under AMP
        probs = F.softmax(logits, dim=1)
        p_t = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        loss = -((1.0 - p_t) ** self.gamma) * torch.log(p_t + 1e-8)
        if self.alpha is not None:
            loss = self.alpha[labels] * loss
        return loss.mean()


def mixup_batch(images, labels, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)        # keep the original label dominant (single draw, reflected)
    index = torch.randperm(images.size(0), device=images.device)
    return lam * images + (1 - lam) * images[index], labels, labels[index], lam


class WaferDS(Dataset):
    """Wafer tensor dataset. aug=True applies light on-the-fly augmentation (train only)."""
    def __init__(self, data, labels, aug=False):
        self.data, self.labels, self.aug = data, labels, aug

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.data[i]
        if self.aug:
            x = torch.rot90(x, torch.randint(0, 4, (1,)).item(), [1, 2])
            if torch.rand(1).item() > 0.5: x = torch.flip(x, [2])
            if torch.rand(1).item() > 0.5: x = torch.flip(x, [1])
            if torch.rand(1).item() < 0.1:               # random erasing
                _, h, w = x.shape
                eh, ew = int(h * np.random.uniform(0.05, 0.15)), int(w * np.random.uniform(0.05, 0.15))
                top, left = np.random.randint(0, h - eh), np.random.randint(0, w - ew)
                x = x.clone()
                x[:, top:top + eh, left:left + ew] = 0
            x = (x + torch.randn_like(x) * 0.015).clamp(0, 1)
        return x, self.labels[i]


# Class-balanced focal alpha (inverse frequency, normalized to sum=NUM_CLASSES)
cc = Counter(train_labels.numpy().tolist())
counts_t = torch.tensor([cc.get(i, 1) for i in range(NUM_CLASSES)], dtype=torch.float32)
inv_freq = 1.0 / counts_t.clamp(min=1)
alpha_w = (inv_freq / inv_freq.sum() * NUM_CLASSES).to(device)


# ============================================================
# 5. TRAIN / EVAL (AMP-enabled, per-model batch sizes)
# ============================================================
def train_model(backbone, classifier, name, lr=3e-4, warmup=10, batch_size=BATCH_CNN, eval_batch=EVAL_BATCH_CNN):
    """Train one backbone with Focal + Mixup. Returns (best_state, elapsed_seconds)."""
    criterion = FocalLoss(gamma=2.0, alpha=alpha_w)
    mixup_alpha, mixup_p = 0.4, 0.5

    sampler = WeightedRandomSampler([1.0 / cc[l.item()] for l in train_labels],
                                    len(train_labels), replacement=True)
    loader = DataLoader(WaferDS(train_data, train_labels, aug=True), batch_size=batch_size,
                        sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    # Validation batch matches the model's eval batch so ViT (224x224) does not OOM here.
    val_loader = DataLoader(WaferDS(val_data, val_labels), batch_size=eval_batch,
                            num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(
        [{"params": backbone.parameters(), "lr": lr * 0.1},   # pretrained backbone: lower LR
         {"params": classifier.parameters(), "lr": lr}],      # fresh head: full LR
        weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - warmup, eta_min=1e-7)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_val, best_state, patience_ctr = float("inf"), None, 0
    t0 = time.time()
    print(f"\n  [{name}] epochs={EPOCHS} lr={lr} batch={batch_size} amp={USE_AMP} "
          f"steps/epoch={len(loader)}")

    for epoch in range(EPOCHS):
        if epoch < warmup:                                    # linear LR warmup
            s = (epoch + 1) / warmup
            optimizer.param_groups[0]["lr"] = lr * 0.1 * s
            optimizer.param_groups[1]["lr"] = lr * s

        backbone.train(); classifier.train()
        t_loss, correct, total = 0.0, 0.0, 0
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
                if torch.rand(1).item() < mixup_p:
                    imgs, la, lb, lam = mixup_batch(imgs, lbls, mixup_alpha)
                    logits = classifier(backbone(imgs))
                    loss = lam * criterion(logits, la) + (1 - lam) * criterion(logits, lb)
                    preds = logits.argmax(1)
                    correct += lam * (preds == la).sum().item() + (1 - lam) * (preds == lb).sum().item()
                else:
                    logits = classifier(backbone(imgs))
                    loss = criterion(logits, lbls)
                    correct += (logits.argmax(1) == lbls).sum().item()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            t_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)

        if epoch >= warmup:
            scheduler.step()

        # Validation
        backbone.eval(); classifier.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
                    logits = classifier(backbone(imgs))
                    loss = criterion(logits, lbls)
                v_loss += loss.item() * imgs.size(0)
                v_correct += (logits.argmax(1) == lbls).sum().item()
                v_total += imgs.size(0)
        v_loss /= v_total
        v_acc = v_correct / v_total

        if best_state is None or v_loss < best_val:
            best_val = v_loss if np.isfinite(v_loss) else best_val
            best_state = {
                "backbone": {k: v.cpu() for k, v in backbone.state_dict().items()},
                "classifier": {k: v.cpu() for k, v in classifier.state_dict().items()},
                "epoch": epoch, "val_loss": v_loss, "val_acc": v_acc}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            el = time.time() - t0
            eta = el / (epoch + 1) * (EPOCHS - epoch - 1)
            print(f"    Ep {epoch+1:3d}/{EPOCHS} | t_loss={t_loss/total:.4f} t_acc={correct/total:.3f} | "
                  f"v_loss={v_loss:.4f} v_acc={v_acc:.3f} | pat={patience_ctr}/{PATIENCE} | ETA={eta/60:.0f}m")

        if patience_ctr >= PATIENCE:
            print(f"    Early stop at epoch {epoch+1}")
            break

    elapsed = time.time() - t0
    print(f"    Done: {elapsed/60:.1f} min | best epoch {best_state['epoch']+1} "
          f"v_loss={best_state['val_loss']:.4f} v_acc={best_state['val_acc']:.3f}")
    return best_state, elapsed


def evaluate_model(backbone, best_state, name, eval_batch=EVAL_BATCH_CNN):
    """Load best weights, extract embeddings, compute macro KNN@{1,3,5,10} (honest)."""
    backbone.load_state_dict({k: v.to(device) for k, v in best_state["backbone"].items()})
    backbone.eval()

    train_eval = DataLoader(WaferDS(train_data, train_labels, aug=False), batch_size=eval_batch,
                            shuffle=False, num_workers=2, pin_memory=True)
    test_eval = DataLoader(WaferDS(test_data, test_labels, aug=False), batch_size=eval_batch,
                           shuffle=False, num_workers=2, pin_memory=True)

    def embed(loader):
        E, L = [], []
        with torch.no_grad():
            for x, y in loader:
                # fp32 forward for numerical safety (no_grad => no activation memory cost).
                e = backbone(x.to(device, non_blocking=True))
                E.append(e.float().cpu())
                L.append(y)
        emb = torch.cat(E).numpy()
        emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)  # guard against rare NaN/Inf
        return emb, torch.cat(L).numpy()

    train_emb, train_lab = embed(train_eval)
    test_emb, test_lab = embed(test_eval)

    res = {}
    for K in [1, 3, 5, 10]:
        knn = KNeighborsClassifier(n_neighbors=K, metric="cosine", n_jobs=-1)
        knn.fit(train_emb, train_lab)
        pred = knn.predict(test_emb)
        res[f"KNN@{K}"] = float(recall_score(test_lab, pred, average="macro"))
        if K == 5:
            res["report"] = classification_report(test_lab, pred, target_names=CLASSES, digits=3)
            res["per_class"] = {
                c: [float(accuracy_score(test_lab[test_lab == i], pred[test_lab == i])), int((test_lab == i).sum())]
                for i, c in enumerate(CLASSES) if (test_lab == i).sum() > 0}

    print(f"\n  {'='*60}\n  {name} — KNN (Focal+Mixup, honest lot-split eval)\n  {'='*60}")
    for K in [1, 3, 5, 10]:
        print(f"    KNN@{K:<2d}: {res[f'KNN@{K}']*100:.1f}%")
    print("\n" + res["report"])
    print(f"  >>> {name} MACRO KNN@5: {res['KNN@5']*100:.1f}% <<<")
    return res


def save_result(name, res, elapsed):
    """Append/overwrite this model's result in the on-disk JSON (incremental safety)."""
    data = {}
    if os.path.exists(RESULTS_JSON):
        try:
            with open(RESULTS_JSON) as f:
                data = json.load(f)
        except Exception:
            data = {}
    data[name] = {
        "knn": {k: res[k] for k in ["KNN@1", "KNN@3", "KNN@5", "KNN@10"]},
        "per_class": res.get("per_class", {}),
        "minutes": round(elapsed / 60, 1),
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  saved -> {RESULTS_JSON}")


def free_gpu(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


# ============================================================
# 6. EXPERIMENTS
# ============================================================
# ---- A: ResNet50 ----
print("\n" + "=" * 65 + "\n  [4/7] EXPERIMENT A: ResNet50 + Focal + Mixup\n" + "=" * 65)
bb = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
bb.fc = nn.Identity()
bb = bb.to(device)
head = nn.Linear(2048, NUM_CLASSES).to(device)
best_r50, t_r50 = train_model(bb, head, "ResNet50", lr=3e-4, warmup=10, batch_size=BATCH_CNN)
res_r50 = evaluate_model(bb, best_r50, "ResNet50", eval_batch=EVAL_BATCH_CNN)
torch.save(best_r50, "resnet50_focal_mixup_best.pth")
save_result("ResNet50", res_r50, t_r50)
free_gpu(bb, head)

# ---- B: EfficientNet-B0 ----
print("\n" + "=" * 65 + "\n  [5/7] EXPERIMENT B: EfficientNet-B0 + Focal + Mixup\n" + "=" * 65)
bb = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
bb.classifier = nn.Identity()
bb = bb.to(device)
head = nn.Linear(1280, NUM_CLASSES).to(device)
best_efn, t_efn = train_model(bb, head, "EfficientNet-B0", lr=3e-4, warmup=10, batch_size=BATCH_CNN)
res_efn = evaluate_model(bb, best_efn, "EfficientNet-B0", eval_batch=EVAL_BATCH_CNN)
torch.save(best_efn, "efficientnet_b0_focal_mixup_best.pth")
save_result("EfficientNet-B0", res_efn, t_efn)
free_gpu(bb, head)


# ---- C: ViT-B/16 (gradient checkpointing + small batch + AMP) ----
print("\n" + "=" * 65 + "\n  [6/7] EXPERIMENT C: ViT-B/16 + Focal + Mixup\n" + "=" * 65)


class ViTWrapper(nn.Module):
    """ViT-B/16 feature extractor. Upsamples 96->224 and checkpoints the
    12 transformer layers to keep activation memory low enough for a T4."""
    def __init__(self, ckpt_segments=4):
        super().__init__()
        self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.model.heads = nn.Identity()
        self.ckpt_segments = ckpt_segments

    def forward(self, x):
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = self.model._process_input(x)
        cls = self.model.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        enc = self.model.encoder
        x = enc.dropout(x + enc.pos_embedding)
        if self.training and x.requires_grad:
            x = checkpoint_sequential(enc.layers, self.ckpt_segments, x, use_reentrant=False)
        else:
            x = enc.layers(x)
        return enc.ln(x)[:, 0]                 # class-token embedding (768-d)


res_vit, t_vit, vit_ok = None, 0.0, False
try:
    bb = ViTWrapper().to(device)
    head = nn.Linear(768, NUM_CLASSES).to(device)
    # ViT is more sensitive: lower LR, longer warmup, smaller batch.
    best_vit, t_vit = train_model(bb, head, "ViT-B/16", lr=1e-4, warmup=15,
                                  batch_size=BATCH_VIT, eval_batch=EVAL_BATCH_VIT)
    res_vit = evaluate_model(bb, best_vit, "ViT-B/16", eval_batch=EVAL_BATCH_VIT)
    torch.save(best_vit, "vit_b16_focal_mixup_best.pth")
    save_result("ViT-B/16", res_vit, t_vit)
    vit_ok = True
    free_gpu(bb, head)
except torch.cuda.OutOfMemoryError:
    print("\n  ViT OOM even with gradient checkpointing + batch=32.")
    print("  Lower BATCH_VIT (e.g. 16) at the top and re-run the ViT stage.")
    free_gpu()


# ============================================================
# 7. FINAL COMPARISON (reads the on-disk JSON so it never crashes)
# ============================================================
print("\n\n" + "=" * 65 + "\n  [7/7] FINAL COMPARISON — Focal + Mixup, honest lot-split eval\n" + "=" * 65)

with open(RESULTS_JSON) as f:
    saved = json.load(f)

print(f"\n  {'Model':<20}{'KNN@1':<9}{'KNN@3':<9}{'KNN@5':<9}{'KNN@10':<9}{'Time':<8}")
print("  " + "-" * 62)
for name in ["ResNet50", "EfficientNet-B0", "ViT-B/16"]:
    if name in saved:
        k = saved[name]["knn"]
        print(f"  {name:<20}{k['KNN@1']*100:<9.1f}{k['KNN@3']*100:<9.1f}"
              f"{k['KNN@5']*100:<9.1f}{k['KNN@10']*100:<9.1f}{saved[name]['minutes']:<.0f}min")
    else:
        print(f"  {name:<20}{'(not completed — see log above)'}")

# Per-class KNN@5 across whichever models finished (Loc / Scratch are the focal+mixup targets)
done = [n for n in ["ResNet50", "EfficientNet-B0", "ViT-B/16"] if n in saved]
if done:
    print(f"\n  Per-class KNN@5:")
    print("  " + "Class".ljust(12) + "".join(n[:11].ljust(13) for n in done))
    print("  " + "-" * (12 + 13 * len(done)))
    for c in CLASSES:
        row = "  " + c.ljust(12)
        for n in done:
            pc = saved[n]["per_class"].get(c)
            row += (f"{pc[0]*100:.1f}").ljust(13) if pc else "-".ljust(13)
        row += "  <- target" if c in ("Loc", "Scratch") else ""
        print(row)

    best = max(done, key=lambda n: saved[n]["knn"]["KNN@5"])
    print(f"\n  Best model: {best}  (Macro KNN@5 = {saved[best]['knn']['KNN@5']*100:.1f}%)")

print(f"\n  Results JSON : {os.path.abspath(RESULTS_JSON)}")
print(f"  Checkpoints  : resnet50_/efficientnet_b0_/vit_b16_focal_mixup_best.pth")
print(f"  Repo         : https://github.com/haykbaghdasaryan-sketch/wafer-vision")
print("=" * 65)

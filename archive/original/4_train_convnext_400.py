# train_convnextb_320_mix_ema.py
import os, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from safetensors.torch import load_file
import timm, torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
from timm.models import load_checkpoint  # ★

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from torchvision.datasets import ImageFolder

# ---- SPEED-UP（建议开启，4090D 很合适）----
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ========== Dataset & Loader ==========
class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            return sample, target
        except Exception:
            return None

def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.tensor(ys)

def get_loaders(root, img_size=320, batch_size=64):  # ✅ 320 / bs=32（OOM 就降 24/16）
    mean, std = [0.485,0.456,0.406], [0.229,0.224,0.225]
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = SafeImageFolder(os.path.join(root, "train"), train_tf)
    val_ds   = SafeImageFolder(os.path.join(root, "val"),   val_tf)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=12, pin_memory=True, persistent_workers=True, prefetch_factor=2,
        drop_last=True,  # ★★★ 关键：丢弃最后一个不足 batch_size 的 batch，避免奇数
        collate_fn=safe_collate
    )
    val_loader   = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=12, pin_memory=True, persistent_workers=True, prefetch_factor=2,
        collate_fn=safe_collate
    )
    num_classes = len(train_ds.classes)
    return train_loader, val_loader, num_classes, train_ds.classes

# ========== AMP ==========
from torch.cuda.amp import autocast, GradScaler

# ========== 轻量 EMA ==========
class ModelEMA:
    def __init__(self, model, decay=0.9998, device=None):
        self.ema = deepcopy(model)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        if device is not None:
            self.ema.to(device)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k in msd:
                v.copy_(v * d + msd[k] * (1.0 - d))

# ========== Eval & Train ==========
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    skipped = 0
    pbar = tqdm(loader, desc="Validating", leave=False, total=len(loader))
    for batch in pbar:
        if batch is None:
            skipped += 1
            continue
        x, y = batch
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        n += x.size(0)
        if n > 0:
            pbar.set_postfix({"val_loss": f"{loss_sum/n:.4f}",
                              "val_acc":  f"{correct/n:.4f}",
                              "skip":     skipped})
    return (loss_sum / max(n,1)), (correct / max(n,1)), skipped

def train_one_epoch(model, loader, optimizer, scaler, criterion, device, mixup_fn=None, ema:ModelEMA=None):
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    skipped = 0
    pbar = tqdm(loader, desc="Training", leave=False, total=len(loader))
    for batch in pbar:
        if batch is None:
            skipped += 1
            continue
        x, y = batch
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        if mixup_fn is not None:
            x, y = mixup_fn(x, y)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=(device=="cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)

        if mixup_fn is None:
            loss_sum += loss.item() * x.size(0)
            correct  += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
            if n > 0:
                pbar.set_postfix({"train_loss": f"{loss_sum/n:.4f}",
                                  "train_acc":  f"{correct/n:.4f}",
                                  "skip":       skipped})
        else:
            pbar.set_postfix({"train_loss": f"{loss.item():.4f}", "skip": skipped})

    if mixup_fn is None:
        return (loss_sum / max(n,1)), (correct / max(n,1)), skipped
    else:
        return (loss.item()), 0.0, skipped

# ========== Main ==========
def main():
    data_root = "/home/ma-user/work/data/webinat5000_split"
    out_dir   = "/home/ma-user/work/exp/webinat5000_convnext_in1k"   # ✅ 新目录
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    train_loader, val_loader, num_classes, class_names = get_loaders(
        data_root, img_size=320, batch_size=64
    )
    print("num_classes =", num_classes)

    with open(os.path.join(out_dir, "classes.txt"), "w", encoding="utf-8") as f:
        for i, name in enumerate(class_names):
            f.write(f"{i}\t{name}\n")


    # ✅ ConvNeXt-Base，严格 in1k 预训练
    model = timm.create_model(
        "convnext_base.fb_in1k",
        pretrained=False,
        num_classes=1000,
        drop_path_rate=0.4,       # ConvNeXt 也吃 droppath
    ).to(device)

    ckpt_path = "/home/ma-user/work/data/weights/convnext_base.fb_in1k.safetensors"  # 改成你的实际路径
    load_checkpoint(model, ckpt_path, strict=False)  # ★ strict=False 允许跳过分类头
    print("Loaded local checkpoint:", ckpt_path)

    model.reset_classifier(num_classes)

# 3) ☆ 关键：把新头一起搬回 GPU，并恢复 channels_last
    model = model.to(device)
    model.to(memory_format=torch.channels_last)



    # channels_last 对 ConvNeXt 也有效
    model.to(memory_format=torch.channels_last)

    # ✅ ConvNeXt 对较大学习率更稳（配合 wd）
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)

    # Mixup/CutMix + 软标签（训练） / CE（验证）
    from timm.data import Mixup
    from timm.loss import SoftTargetCrossEntropy
    mixup_fn = Mixup(
        mixup_alpha=0.2, cutmix_alpha=0.2, prob=1.0, switch_prob=0.5,
        label_smoothing=0.0, num_classes=num_classes
    )
    train_criterion = SoftTargetCrossEntropy()
    val_criterion   = nn.CrossEntropyLoss()

    scaler = GradScaler(enabled=(device=="cuda"))
    ema = ModelEMA(model, decay=0.9999, device=device)

    best_acc, best_path = 0.0, os.path.join(out_dir, "best.pth")
    epochs = 100
    warmup = LinearLR(optimizer, start_factor=0.2, total_iters=3)
    cosine = CosineAnnealingLR(optimizer, T_max=epochs-3, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[3])

    for epoch in range(1, epochs+1):
        t0 = time.time()
        tr_loss, tr_acc, tr_skip = train_one_epoch(
            model, train_loader, optimizer, scaler, train_criterion, device,
            mixup_fn=mixup_fn, ema=ema
        )

        val_loss, val_acc, va_skip = evaluate(ema.ema, val_loader, val_criterion, device)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        print(f"Epoch {epoch:02d}/{epochs} | lr {lr:.2e} | "
              f"train loss {tr_loss:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} (skip {va_skip}) | {dt:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(ema.ema.state_dict(), best_path)
            print(f"  ✅ New best acc={best_acc:.4f} -> {best_path}")

    print(f"Done. Best val acc={best_acc:.4f}")

if __name__ == "__main__":
    main()

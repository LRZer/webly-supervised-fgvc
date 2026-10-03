"""Portable ConvNeXt training derived from the recovered two task scripts."""
import argparse
import csv
import json
import random
from copy import deepcopy
from pathlib import Path
import time

import numpy as np
from PIL import ImageFile
import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.models import load_checkpoint
import torch
from torch import nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

ImageFile.LOAD_TRUNCATED_IMAGES = True


class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except (OSError, ValueError):
            return None


def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.tensor(ys)


class ModelEMA:
    def __init__(self, model, decay):
        self.ema = deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        current = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.is_floating_point():
                v.mul_(self.decay).add_(current[k], alpha=1 - self.decay)
            else:
                v.copy_(current[k])


def loaders(args, cfg):
    size = cfg['img_size']
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(size * 1.14)), transforms.CenterCrop(size),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    train = SafeImageFolder(args.data_root / 'train', train_tf)
    val = SafeImageFolder(args.data_root / 'val', val_tf, allow_empty=True)
    if train.class_to_idx != val.class_to_idx:
        raise ValueError('Train and validation class mappings must match')
    if len(train.classes) != cfg['expected_classes']:
        raise ValueError(f"Expected {cfg['expected_classes']} classes, found {len(train.classes)}")
    kw = dict(batch_size=cfg['batch_size'], num_workers=cfg['workers'],
              pin_memory=args.device == 'cuda', persistent_workers=bool(cfg['workers']),
              prefetch_factor=2 if cfg['workers'] else None, collate_fn=safe_collate)
    return (DataLoader(train, shuffle=True, drop_last=True, **kw),
            DataLoader(val, shuffle=False, **kw), train.classes)


@torch.no_grad()
def evaluate(model, loader, device, max_batches=0):
    model.eval()
    loss_sum = correct = n = 0
    for step, batch in enumerate(loader):
        if max_batches and step >= max_batches:
            break
        if batch is None:
            continue
        x, y = (v.to(device) for v in batch)
        logits = model(x)
        loss_sum += nn.functional.cross_entropy(logits, y).item() * len(x)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(x)
    if n == 0:
        raise RuntimeError('Validation has no readable images')
    return loss_sum / n, correct / n, n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    weights = ap.add_mutually_exclusive_group(required=True)
    weights.add_argument('--pretrained-checkpoint', type=Path, help='Local ImageNet-1K pretrained weights')
    weights.add_argument('--init-weights', type=Path, help='Existing task weights; starts a new optimizer and schedule')
    weights.add_argument('--random-init', action='store_true', help='Synthetic smoke tests only')
    for name, typ in [('epochs', int), ('batch-size', int), ('workers', int), ('img-size', int),
                      ('warmup-epochs', int), ('expected-classes', int), ('lr', float), ('seed', int)]:
        ap.add_argument('--' + name, type=typ, default=None)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--max-batches', type=int, default=0, help='Nonzero limits each epoch for smoke testing')
    ap.add_argument('--disable-mixup', action='store_true')
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    for k in cfg:
        value = getattr(args, k, None)
        if value is not None:
            cfg[k] = value
    if cfg['epochs'] < 1 or not 0 <= cfg['warmup_epochs'] < cfg['epochs']:
        raise ValueError('Require epochs > warmup_epochs >= 0')
    if cfg['batch_size'] < 1 or cfg['workers'] < 0 or cfg['img_size'] < 32 or args.max_batches < 0:
        raise ValueError('Invalid loader, image size, or batch limit')
    if cfg['lr'] <= 0 or not 0 <= cfg['ema_decay'] < 1:
        raise ValueError('Invalid learning rate or EMA decay')
    for p in [args.pretrained_checkpoint, args.init_weights]:
        if p is not None and not p.is_file():
            raise FileNotFoundError(p)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError('Use a new output directory for each training run')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    if args.device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    train_dl, val_dl, classes = loaders(args, cfg)
    model = timm.create_model(cfg['model'], pretrained=False, num_classes=1000,
                              drop_path_rate=cfg['drop_path_rate'])
    if args.pretrained_checkpoint:
        load_checkpoint(model, str(args.pretrained_checkpoint), strict=True)
    model.reset_classifier(len(classes))
    if args.init_weights:
        model.load_state_dict(torch.load(args.init_weights, map_location='cpu', weights_only=True), strict=True)
    model.to(args.device, memory_format=torch.channels_last)
    ema = ModelEMA(model, cfg['ema_decay'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    warm = cfg['warmup_epochs']
    cosine = CosineAnnealingLR(optimizer, T_max=cfg['epochs'] - warm, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [LinearLR(optimizer, start_factor=0.2, total_iters=warm), cosine], [warm]) if warm else cosine
    mixup = None if args.disable_mixup else Mixup(
        mixup_alpha=cfg['mixup_alpha'], cutmix_alpha=cfg['cutmix_alpha'], prob=1.0,
        switch_prob=0.5, label_smoothing=0.0, num_classes=len(classes))
    criterion = SoftTargetCrossEntropy() if mixup else nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=args.device == 'cuda')
    resolved = dict(cfg, data_root=str(args.data_root), device=args.device,
                    weight_source=str(args.pretrained_checkpoint or args.init_weights or 'random_init'),
                    max_batches=args.max_batches, mixup_enabled=bool(mixup))
    (args.out_dir / 'run_config.json').write_text(json.dumps(resolved, indent=2) + '\n', encoding='utf-8')
    (args.out_dir / 'classes.txt').write_text(''.join(f'{i}\t{n}\n' for i, n in enumerate(classes)), encoding='utf-8')
    best = -1.0
    with (args.out_dir / 'history.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'lr', 'train_loss', 'val_loss', 'val_accuracy', 'train_images', 'val_images', 'seconds'])
        for epoch in range(1, cfg['epochs'] + 1):
            t0 = time.time(); model.train(); loss_sum = n = 0
            epoch_lr = optimizer.param_groups[0]['lr']
            for step, batch in enumerate(train_dl):
                if args.max_batches and step >= args.max_batches:
                    break
                if batch is None:
                    continue
                x, y = (v.to(args.device) for v in batch)
                if mixup:
                    # timm batch-mode Mixup needs even batches after decode failures.
                    if len(x) % 2:
                        x, y = x[:-1], y[:-1]
                    if len(x) == 0:
                        continue
                    x, y = mixup(x, y)
                x = x.contiguous(memory_format=torch.channels_last)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=args.device, enabled=args.device == 'cuda'):
                    loss = criterion(model(x), y)
                if not torch.isfinite(loss):
                    raise RuntimeError('Training loss is not finite')
                scale = scaler.get_scale()
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                if scaler.get_scale() >= scale:
                    ema.update(model)
                loss_sum += loss.item() * len(x); n += len(x)
            if n == 0:
                raise RuntimeError('No training batch processed; check batch size and readable images')
            vl, va, vn = evaluate(ema.ema, val_dl, args.device, args.max_batches)
            scheduler.step()
            writer.writerow([epoch, epoch_lr, loss_sum/n, vl, va, n, vn, time.time()-t0]); f.flush()
            torch.save(ema.ema.state_dict(), args.out_dir / 'last.pth')
            torch.save(model.state_dict(), args.out_dir / 'last_raw.pth')
            if va > best:
                best = va; torch.save(ema.ema.state_dict(), args.out_dir / 'best.pth')
            print(f'Epoch {epoch}/{cfg["epochs"]}: train_loss={loss_sum/n:.4f}, val_accuracy={va:.4f}, best={best:.4f}', flush=True)


if __name__ == '__main__':
    main()

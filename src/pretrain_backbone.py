"""Preentrena el backbone propio en clasificación de imágenes (transfer learning).

Solo el backbone se transfiere; las cabezas de detección y segmentación se
entrenan desde cero en PENGWIN. Se usa Intel Image Classification (carpetas
por clase) convertido a escala de grises, y el backbone propio
(FundidoraBackbone), no una arquitectura ajena.

    python src/pretrain_backbone.py --data /ruta/seg_train --out outputs/s10/backbone_intel_w32.pth \
        --base-channels 32 --epochs 8 --cbam 1

Luego:  python src/train_s10.py ... --backbone-init outputs/s10/backbone_intel_w32.pth
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from week9_detection import FundidoraBackbone


class BackboneClassifier(nn.Module):
    def __init__(self, num_classes: int, in_channels: int, base_channels: int, cbam: int):
        super().__init__()
        self.backbone = FundidoraBackbone(in_channels=in_channels, base_channels=base_channels,
                                          use_cbam=cbam > 0, cbam_stages=3 if cbam == 3 else 1)
        self.fc = nn.Linear(self.backbone.out_channels, num_classes)

    def forward(self, x):
        f = self.backbone(x)
        return self.fc(F.adaptive_avg_pool2d(f, 1).flatten(1))


def build_loaders(root: Path, size: int, batch: int, in_channels: int, workers: int):
    from torchvision import datasets, transforms
    tf_train = transforms.Compose([
        transforms.Grayscale(1), transforms.RandomResizedCrop(size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.5], [0.25])])
    ds = datasets.ImageFolder(str(root), transform=tf_train)
    n_val = max(1, int(0.1 * len(ds)))
    g = torch.Generator().manual_seed(0)
    train, val = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val], generator=g)
    mk = lambda d, sh: torch.utils.data.DataLoader(d, batch_size=batch, shuffle=sh, num_workers=workers,
                                                   drop_last=sh)
    return mk(train, True), mk(val, False), len(ds.classes)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--cbam", type=int, default=1, choices=[0, 1, 3])
    p.add_argument("--in-channels", type=int, default=1)
    p.add_argument("--size", type=int, default=160)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val, n_cls = build_loaders(args.data, args.size, args.batch_size, args.in_channels, args.workers)
    model = BackboneClassifier(n_cls, args.in_channels, args.base_channels, args.cbam).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train))
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y in train:
            x = x.to(device).repeat(1, args.in_channels, 1, 1)
            loss = F.cross_entropy(model(x), y.to(device), label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        ok = tot = 0
        with torch.no_grad():
            for x, y in val:
                pred = model(x.to(device).repeat(1, args.in_channels, 1, 1)).argmax(1).cpu()
                ok += int((pred == y).sum())
                tot += len(y)
        print(f"epoch {ep:02d} loss={loss.item():.3f} val_acc={ok / tot:.3f} ({time.time() - t0:.0f}s)", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone": model.backbone.state_dict(), "base_channels": args.base_channels,
                "cbam": args.cbam, "in_channels": args.in_channels, "val_acc": ok / tot}, args.out)
    print(f"Guardado {args.out}")


if __name__ == "__main__":
    main()

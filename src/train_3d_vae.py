from __future__ import annotations
import argparse, os, csv, math
from datetime import datetime, UTC
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import VoxelNPZDataset
from model_3d import VAE3D, kl_divergence
from utils import ensure_dir, save_json

def compute_iou(pred_logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    p = (torch.sigmoid(pred_logits) > threshold).float()
    t = (target > 0.5).float()
    inter = (p * t).sum(dim=(1,2,3,4))
    union = ((p + t) > 0).float().sum(dim=(1,2,3,4)).clamp_min(1.0)
    return float((inter / union).mean().item())

def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.sigmoid(logits)
    t = (target > 0.5).float()
    num = 2.0 * (p * t).sum(dim=(1,2,3,4)) + eps
    den = (p + t).sum(dim=(1,2,3,4)) + eps
    dice = 1.0 - (num / den)
    return dice.mean()

@torch.no_grad()
def save_recon_grid(model, batch, out_path: str):
    model.eval()
    x = batch[:8]
    logits, _, _ = model(x)
    probs = torch.sigmoid(logits)

    def proj(v):
        v = (v > 0.5).float()
        xy = v.max(dim=4).values.squeeze(1)
        return xy.cpu().numpy()

    gt = proj(x)
    rc = proj(probs)
    B = gt.shape[0]
    fig, axes = plt.subplots(2, B, figsize=(B*2, 4))
    for i in range(B):
        axes[0, i].imshow(gt[i], cmap="gray"); axes[0, i].axis("off")
        axes[1, i].imshow(rc[i], cmap="gray"); axes[1, i].axis("off")
    axes[0,0].set_title("GT"); axes[1,0].set_title("Recon")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument("--base_ch", type=int, default=48)
    p.add_argument("--kl_weight", type=float, default=5e-4)
    p.add_argument("--kl_warmup_epochs", type=int, default=30, help="linearly warm up kl_weight over first N epochs")
    p.add_argument("--dice_weight", type=float, default=0.5, help="extra dice loss weight on top of BCE")
    p.add_argument("--overfit_n", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--amp", action="store_true", help="mixed precision on CUDA")
    p.add_argument("--grad_clip", type=float, default=1.0)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, f"run_{timestamp}")
    ensure_dir(run_dir)
    save_json(vars(args), os.path.join(run_dir, "config.json"))

    train_ds = VoxelNPZDataset(os.path.join(args.data_root, "train.npz"), resolution=args.resolution, augment=True, seed=args.seed)
    val_ds = VoxelNPZDataset(os.path.join(args.data_root, "val.npz"), resolution=args.resolution, augment=False, seed=args.seed)
    if args.overfit_n and args.overfit_n > 0:
        train_ds.voxels = train_ds.voxels[:args.overfit_n]
        val_ds.voxels = val_ds.voxels[:min(len(val_ds), args.overfit_n)]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=args.pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=args.pin_memory)

    device = torch.device(args.device)
    model = VAE3D(resolution=args.resolution, latent_dim=args.latent_dim, base_ch=args.base_ch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # estimate pos_weight for sparse occupancy from a few batches
    with torch.no_grad():
        xs = []
        for i, x in enumerate(train_loader):
            xs.append(x)
            if i >= 6: break
        x0 = torch.cat(xs, dim=0)
        pos = float((x0 > 0.5).sum().item())
        neg = float((x0 <= 0.5).sum().item())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Estimated pos_weight={float(pos_weight.item()):.2f} (pos={pos:.0f}, neg={neg:.0f})")

    use_amp = bool(args.amp) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    metrics_path = os.path.join(run_dir, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch","lr","kl_w","train_loss","train_bce","train_dice","train_kl","val_loss","val_iou"])

    best_val = float("inf")
    hist = []

    for epoch in range(1, args.epochs+1):
        model.train()
        tl=tb=td=tk=0.0; nb=0

        # KL warmup
        if args.kl_warmup_epochs and args.kl_warmup_epochs > 0:
            kl_w = args.kl_weight * min(1.0, epoch / float(args.kl_warmup_epochs))
        else:
            kl_w = args.kl_weight

        for x in train_loader:
            x = x.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, mu, logvar = model(x)
                b = bce(logits, x)
                d = dice_loss_from_logits(logits, x)
                kl = kl_divergence(mu, logvar)
                loss = b + args.dice_weight * d + kl_w * kl

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            tl += float(loss.item()); tb += float(b.item()); td += float(d.item()); tk += float(kl.item()); nb += 1

        sched.step()

        tl/=max(nb,1); tb/=max(nb,1); td/=max(nb,1); tk/=max(nb,1)
        lr_cur = float(opt.param_groups[0]["lr"])

        model.eval()
        vl=vi=0.0; vb=0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits, mu, logvar = model(x)
                    b = bce(logits, x)
                    d = dice_loss_from_logits(logits, x)
                    kl = kl_divergence(mu, logvar)
                    loss = b + args.dice_weight * d + kl_w * kl
                vl += float(loss.item())
                vi += compute_iou(logits, x)
                vb += 1
        vl/=max(vb,1); vi/=max(vb,1)

        print(f"Epoch {epoch:03d} | lr {lr_cur:.2e} | kl_w {kl_w:.2e} | train {tl:.4f} (bce {tb:.4f}, dice {td:.4f}, kl {tk:.4f}) | val {vl:.4f} | IoU {vi:.3f}")

        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, lr_cur, kl_w, tl, tb, td, tk, vl, vi])

        hist.append((epoch, tl, vl, vi))

        # recon preview
        try:
            batch0 = next(iter(val_loader)).to(device)
            save_recon_grid(model, batch0, os.path.join(run_dir, f"epoch_{epoch:03d}_recon.png"))
        except Exception:
            pass

        ckpt = {"model": model.state_dict(), "config": vars(args), "pos_weight": float(pos_weight.item())}
        torch.save(ckpt, os.path.join(run_dir, "last.pt"))
        if vl < best_val:
            best_val = vl
            torch.save(ckpt, os.path.join(run_dir, "best.pt"))

    # plots
    ep=[h[0] for h in hist]; trl=[h[1] for h in hist]; vll=[h[2] for h in hist]; iou=[h[3] for h in hist]
    fig=plt.figure(); plt.plot(ep,trl,label="train_loss"); plt.plot(ep,vll,label="val_loss"); plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss"); plt.tight_layout()
    plt.savefig(os.path.join(run_dir,"loss_curve.png"),dpi=150); plt.close(fig)
    fig=plt.figure(); plt.plot(ep,iou,label="val_iou"); plt.legend(); plt.xlabel("epoch"); plt.ylabel("IoU"); plt.tight_layout()
    plt.savefig(os.path.join(run_dir,"iou_curve.png"),dpi=150); plt.close(fig)

    print("Done. Run directory:", os.path.abspath(run_dir))

if __name__ == "__main__":
    main()

from __future__ import annotations
import argparse, json, os
import torch
from torch.utils.data import DataLoader
from dataset import VoxelNPZDataset
from model_3d import VAE3D, kl_divergence

def compute_iou(pred_logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    p = (torch.sigmoid(pred_logits) > threshold).float()
    t = (target > 0.5).float()
    inter = (p * t).sum(dim=(1,2,3,4))
    union = ((p + t) > 0).float().sum(dim=(1,2,3,4)).clamp_min(1.0)
    return float((inter / union).mean().item())

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--split", choices=["train","val","test"], default="test")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("config", {})
    pos_weight = float(ckpt.get("pos_weight", 1.0))
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    resolution = int(cfg.get("resolution", 64))
    latent_dim = int(cfg.get("latent_dim", 128))
    base_ch = int(cfg.get("base_ch", 48))
    kl_weight = float(cfg.get("kl_weight", 5e-4))

    ds = VoxelNPZDataset(os.path.join(args.data_root, f"{args.split}.npz"), resolution=resolution, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = VAE3D(resolution=resolution, latent_dim=latent_dim, base_ch=base_ch)
    model.load_state_dict(ckpt["model"]); model.eval()

    total_loss=total_recon=total_kl=total_iou=0.0; n=0
    for x in loader:
        logits, mu, logvar = model(x)
        recon = bce(logits, x)
        kl = kl_divergence(mu, logvar)
        loss = recon + kl_weight * kl
        iou = compute_iou(logits, x, threshold=args.threshold)
        bs = x.shape[0]
        total_loss += float(loss.item())*bs
        total_recon += float(recon.item())*bs
        total_kl += float(kl.item())*bs
        total_iou += float(iou)*bs
        n += bs

    out = {
        "split": args.split,
        "n": n,
        "loss": total_loss/max(n,1),
        "recon_bce": total_recon/max(n,1),
        "kl": total_kl/max(n,1),
        "iou": total_iou/max(n,1),
        "pos_weight": pos_weight,
        "threshold": args.threshold,
        "resolution": resolution,
        "latent_dim": latent_dim,
        "base_ch": base_ch,
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()

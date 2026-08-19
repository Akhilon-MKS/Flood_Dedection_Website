"""
train.py
--------
Trains a U-Net (ResNet34 encoder) on Sen1Floods11 for flood segmentation.

Usage:
    python train.py --data_dir /path/to/sen1floods11 --epochs 15

For a fast hackathon run, reduce --epochs and/or use --max_samples to
train on a subset first, just to confirm the pipeline works end-to-end
before committing to a longer run.
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import segmentation_models_pytorch as smp
from tqdm import tqdm

from dataset import Sen1Floods11Dataset


def dice_loss(pred, target, eps=1e-7):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def combined_loss(pred, target):
    bce = nn.BCEWithLogitsLoss()(pred, target)
    dl = dice_loss(pred, target)
    return bce + dl


def iou_score(pred, target, threshold=0.5, eps=1e-7):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = ((pred + target) > 0).float().sum(dim=(1, 2, 3))
    return ((intersection + eps) / (union + eps)).mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                         help="folder containing S1Hand/ and LabelHand/")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None,
                         help="limit dataset size for a quick smoke-test run")
    parser.add_argument("--train_split", type=str, default=None)
    parser.add_argument("--val_split", type=str, default=None)
    parser.add_argument("--out", type=str, default="models/flood_unet.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = Sen1Floods11Dataset(
        args.data_dir, split_list=args.train_split, augment=True, img_size=args.img_size
    )
    val_ds = Sen1Floods11Dataset(
        args.data_dir, split_list=args.val_split, augment=False, img_size=args.img_size
    ) if args.val_split else None

    if args.max_samples:
        train_ds = Subset(train_ds, range(min(args.max_samples, len(train_ds))))

    if val_ds is None:
        n = len(train_ds)
        n_val = max(1, int(0.15 * n))
        n_train = n - n_val
        train_ds, val_ds = torch.utils.data.random_split(train_ds, [n_train, n_val])
    else:
        n_train = len(train_ds)
        n_val = len(val_ds)
    print(f"Train samples: {n_train} | Val samples: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # U-Net with pretrained ResNet34 encoder, adapted for 2-channel SAR input
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=2,      # VV, VH
        classes=1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    best_iou = 0.0
    output_dir = os.path.dirname(args.out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for sar, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]"):
            sar, mask = sar.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = model(sar)
            loss = combined_loss(pred, mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_iou = 0.0
        with torch.no_grad():
            for sar, mask in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [val]"):
                sar, mask = sar.to(device), mask.to(device)
                pred = model(sar)
                val_iou += iou_score(pred, mask)

        val_iou /= max(1, len(val_loader))
        scheduler.step(val_iou)
        print(f"Epoch {epoch+1}: train_loss={train_loss/len(train_loader):.4f}  val_IoU={val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), args.out)
            print(f"  -> saved new best model (IoU={val_iou:.4f}) to {args.out}")

    print(f"Training complete. Best val IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()

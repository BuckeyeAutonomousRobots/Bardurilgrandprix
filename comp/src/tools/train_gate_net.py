"""Train GateNet segmentation on synthetic and/or MonoRaceGate data."""

from __future__ import annotations

import argparse
import os
import sys

try:
    import torch
    from torch.utils.data import ConcatDataset, DataLoader, Dataset
except ImportError as exc:
    print(f"ERROR: {exc}")
    sys.exit(1)

from src.perception.gatenet.gate_net import GateNet, gate_net_loss
from src.perception.gatenet.monorace_dataset import (
    GateNetTorchDataset,
    LiveCaptureFolderDataset,
    MonoRaceGateFolderDataset,
    SyntheticGateDataset,
    default_monorace_gate_root,
)


def _device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _FolderTorchDataset(Dataset):
    def __init__(self, folder_ds: MonoRaceGateFolderDataset | LiveCaptureFolderDataset):
        self.ds = folder_ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        x, y = self.ds[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MonoRace GateNet")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--samples", type=int, default=3000, help="Synthetic training samples")
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--output", default="models/gate_net.pth")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--monorace-data",
        default="",
        help="Optional MonoRaceGate data folder with downloaded images",
    )
    parser.add_argument(
        "--live-data",
        default="",
        help="Folder with frame_XXXXX.jpg + frame_XXXXX_mask.png from live sim capture",
    )
    args = parser.parse_args()

    device = _device(args.device)
    print(f"[GateNet] device={device}")

    datasets = [GateNetTorchDataset(SyntheticGateDataset(size=args.size, length=args.samples))]

    mono_root = args.monorace_data or str(default_monorace_gate_root())
    if os.path.isdir(mono_root):
        real = MonoRaceGateFolderDataset(mono_root, size=args.size)
        if len(real) > 0:
            datasets.append(_FolderTorchDataset(real))
            print(f"[GateNet] Added {len(real)} real MonoRaceGate samples")
        else:
            print(f"[GateNet] No images in {mono_root}; synthetic only")

    if args.live_data and os.path.isdir(args.live_data):
        live = LiveCaptureFolderDataset(args.live_data, size=args.size)
        if len(live) > 0:
            datasets.append(_FolderTorchDataset(live))
            print(f"[GateNet] Added {len(live)} live sim capture samples")
        else:
            print(f"[GateNet] No paired frames in {args.live_data}")

    train_set = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = GateNet(factor=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[10, 25, 35], gamma=0.316)

    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            preds = model(imgs)
            loss = gate_net_loss(preds, masks)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item())
        sched.step()
        avg = running / max(len(loader), 1)

        marker = ""
        if avg < best:
            best = avg
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "loss": avg,
                    "input_size": args.size,
                },
                args.output,
            )
            marker = " *saved*"

        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch + 1:3d}/{args.epochs} loss={avg:.4f}{marker}")

    print(f"[GateNet] best loss={best:.4f} saved to {args.output}")


if __name__ == "__main__":
    main()

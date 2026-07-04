import io
import os
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

ROOT = Path("cifar100")
URLS = {
    "train": "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/train-00000-of-00001.parquet",
    "test": "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/test-00000-of-00001.parquet",
}


def download(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 1024 * 1024:
        return
    subprocess.run(["curl", "-L", "--fail", "-C", "-", "-o", str(dst), url], check=True)


def convert(split: str) -> None:
    out = ROOT / f"{split}.pt"
    if out.exists():
        data = torch.load(out, map_location="cpu", weights_only=True)
        img_shape = tuple(data["images"].shape)
        lab_shape = tuple(data["labels"].shape)
        print(f"{out}: exists images={img_shape} labels={lab_shape}")
        return
    parquet = ROOT / f"{split}.parquet"
    download(URLS[split], parquet)
    table = pq.read_table(parquet)
    names = set(table.column_names)
    label_col = "fine_label" if "fine_label" in names else "label"
    img_col = "img" if "img" in names else "image"
    imgs = table.column(img_col).to_pylist()
    labels = table.column(label_col).to_pylist()
    arr = np.empty((len(labels), 32, 32, 3), dtype=np.uint8)
    for i, rec in enumerate(imgs):
        raw = rec["bytes"] if isinstance(rec, dict) else rec
        with Image.open(io.BytesIO(raw)) as im:
            arr[i] = np.asarray(im.convert("RGB"))
    y = torch.tensor(labels, dtype=torch.long)
    torch.save({"images": torch.from_numpy(arr), "labels": y, "classes": list(range(100))}, out)
    counts = torch.bincount(y, minlength=100)
    print(f"{out}: wrote images={tuple(arr.shape)} labels={tuple(y.shape)} min_count={int(counts.min())} max_count={int(counts.max())}")


if __name__ == "__main__":
    ROOT.mkdir(exist_ok=True)
    raw_splits = os.getenv("C100_PREP_SPLITS", "train,test")
    splits = [split.strip() for split in raw_splits.split(",") if split.strip()]
    unknown = sorted(set(splits) - set(URLS))
    if unknown:
        raise ValueError(f"unknown C100_PREP_SPLITS entries: {unknown}")
    for split in splits:
        convert(split)
    data = torch.load(ROOT / "train.pt", map_location="cpu", weights_only=True)
    x = data["images"].float().div(255)
    print("mean", [round(v, 6) for v in x.mean((0, 1, 2)).tolist()])
    print("std", [round(v, 6) for v in x.std((0, 1, 2)).tolist()])

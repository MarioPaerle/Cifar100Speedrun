
"""CIFAR-100 single-A100 speedrun baseline.

BENCHMARK CONTRACT
- Validation is frozen. Do not optimize, tune, branch, adapt, augment, or compile
  against the validation path as a benchmark improvement. Validation is an
  untimed pass/fail gate only.
- Compilation, data staging, warmup, logging, and measurement boundaries are
  benchmark infrastructure. Do not optimize them for record claims. They may
  only be changed to fix correctness/portability bugs while preserving semantics.
- The only admissible optimization surfaces are model architecture, optimizer,
  and training hyperparameters inside the timed training loop.
"""

import csv
import hashlib
import json
import math
import os
import random
import subprocess
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


torch.backends.cudnn.benchmark = True
MEAN = torch.tensor((0.5071, 0.4867, 0.4408), dtype=torch.float16, device="cuda").view(1, 3, 1, 1)
STD = torch.tensor((0.2675, 0.2565, 0.2761), dtype=torch.float16, device="cuda").view(1, 3, 1, 1)


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def load_split(name):
    data = torch.load(Path("cifar100") / f"{name}.pt", map_location="cuda", weights_only=True)
    images = data["images"].to(torch.float16).div_(255.0).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
    labels = data["labels"].long()
    return images, labels


@torch.no_grad()
def make_train_dev_split(images, labels, dev_per_class=50, split_seed=20260703):
    generator = torch.Generator(device=labels.device)
    generator.manual_seed(split_seed)
    train_parts, dev_parts = [], []
    for cls in range(100):
        cls_idx = torch.nonzero(labels == cls, as_tuple=False).flatten()
        order = torch.randperm(len(cls_idx), generator=generator, device=labels.device)
        cls_idx = cls_idx[order]
        dev_parts.append(cls_idx[:dev_per_class])
        train_parts.append(cls_idx[dev_per_class:])
    train_idx = torch.cat(train_parts)
    dev_idx = torch.cat(dev_parts)
    train_idx = train_idx[torch.randperm(len(train_idx), generator=generator, device=labels.device)]
    dev_idx = dev_idx[torch.randperm(len(dev_idx), generator=generator, device=labels.device)]
    return images[train_idx], labels[train_idx], images[dev_idx], labels[dev_idx]


@torch.no_grad()
def normalize(x):
    return (x - MEAN) / STD


@torch.no_grad()
def random_crop_flip(x, pad=4):
    b, c, h, w = x.shape
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    ys = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    xs = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    yy = torch.arange(h, device=x.device).view(1, 1, h, 1) + ys.view(b, 1, 1, 1)
    xx = torch.arange(w, device=x.device).view(1, 1, 1, w) + xs.view(b, 1, 1, 1)
    bb = torch.arange(b, device=x.device).view(b, 1, 1, 1)
    cc = torch.arange(c, device=x.device).view(1, c, 1, 1)
    out = x[bb, cc, yy, xx]
    flip = (torch.rand(b, device=x.device) < 0.5).view(b, 1, 1, 1)
    return torch.where(flip, out.flip(-1), out).contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def random_cutout(x, size):
    if size <= 0:
        return x
    b, _, h, w = x.shape
    size = min(size, h, w)
    ys = torch.randint(0, h - size + 1, (b,), device=x.device)
    xs = torch.randint(0, w - size + 1, (b,), device=x.device)
    yy = torch.arange(h, device=x.device).view(1, h, 1)
    xx = torch.arange(w, device=x.device).view(1, 1, w)
    mask = (
        (yy >= ys.view(b, 1, 1))
        & (yy < (ys + size).view(b, 1, 1))
        & (xx >= xs.view(b, 1, 1))
        & (xx < (xs + size).view(b, 1, 1))
    )
    return x.masked_fill(mask.view(b, 1, h, w), 0.0).contiguous(memory_format=torch.channels_last)


class Block(nn.Module):
    def __init__(self, channels_in, channels_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels_out)
        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels_out)
        self.skip = nn.Identity() if channels_in == channels_out and stride == 1 else nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 1, stride=stride, bias=False),
            nn.BatchNorm2d(channels_out),
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


class SimpleResNet(nn.Module):
    def __init__(self, widths=(64, 128, 256), blocks=(2, 2, 2), num_classes=100):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        layers = []
        channels = widths[0]
        for stage, (width, n_blocks) in enumerate(zip(widths, blocks)):
            for i in range(n_blocks):
                stride = 2 if stage > 0 and i == 0 else 1
                layers.append(Block(channels, width, stride))
                channels = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(channels, num_classes, bias=False)
        self.to(memory_format=torch.channels_last)

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)


@torch.no_grad()
def zeropower_newton_schulz(g, steps=5):
    shape = g.shape
    x = g.reshape(g.shape[0], -1).float()
    if x.norm() == 0:
        return torch.zeros_like(g)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        xx = x @ x.T
        x = a * x + (b * xx + c * xx @ xx) @ x
    if transposed:
        x = x.T
    return x.reshape(shape).to(g.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if wd:
                    p.mul_(1 - lr * wd)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(p.grad)
                update = zeropower_newton_schulz(buf, steps=ns_steps)
                fan_out = update.shape[0]
                fan_in = max(1, update.numel() // fan_out)
                scale = math.sqrt(max(1.0, fan_out / fan_in))
                p.add_(update, alpha=-lr * scale)


def batches(images, labels, batch_size):
    order = torch.randperm(len(images), device=images.device)
    usable = len(order) // batch_size * batch_size
    order = order[:usable]
    for i in range(0, usable, batch_size):
        idx = order[i:i + batch_size]
        yield images[idx], labels[idx]


# Frozen validation gate. This function is outside the timed score and is not
# an optimization surface: no TTA, TTT, ensembling, confidence branches, BN
# adaptation, validation-label feedback, or benchmark-specific compilation games.
@torch.no_grad()
def evaluate(model, images, labels, batch_size=1000):
    model.eval()
    total = 0
    correct = 0
    for i in range(0, len(images), batch_size):
        x = normalize(images[i:i + batch_size])
        y = labels[i:i + batch_size]
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += len(y)
    return correct / total


def reset_model(model):
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()


def git_sha():
    return git_output(["git", "rev-parse", "HEAD"])


def git_output(args, default="unknown"):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return default


def safe_env_snapshot():
    prefixes = ("C100_", "CUDA", "HF_", "HUGGINGFACE_", "NVIDIA", "SLURM_", "TORCH", "TRITON", "UV_", "XDG_")
    names = {
        "HF_DATASETS_CACHE",
        "HF_HOME",
        "HOSTNAME",
        "OMP_NUM_THREADS",
        "PIP_CACHE_DIR",
        "PYTHONHASHSEED",
        "PYTHONPATH",
        "TRANSFORMERS_CACHE",
        "VIRTUAL_ENV",
    }
    sensitive = ("KEY", "PASSWORD", "SECRET", "TOKEN")
    snapshot = {}
    for key, value in sorted(os.environ.items()):
        if key in names or key.startswith(prefixes):
            snapshot[key] = "<redacted>" if any(marker in key for marker in sensitive) else value
    return snapshot


def parse_int_tuple_env(name, default, expected_len):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return tuple(default)
    try:
        values = tuple(int(part.strip()) for part in raw.split(","))
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated integers, got {raw!r}") from exc
    if len(values) != expected_len:
        raise ValueError(f"{name} must contain {expected_len} integers, got {len(values)} from {raw!r}")
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} values must be positive, got {raw!r}")
    return values


def parse_lr_schedule_env():
    lr_schedule = os.getenv("C100_LR_SCHEDULE", "cosine").strip().lower()
    if lr_schedule not in ("cosine", "onecycle"):
        raise ValueError(f"C100_LR_SCHEDULE must be 'cosine' or 'onecycle', got {lr_schedule!r}")
    onecycle_pct_up = float(os.getenv("C100_ONECYCLE_PCT_UP", "0.30"))
    if not math.isfinite(onecycle_pct_up) or not 0.0 < onecycle_pct_up < 1.0:
        raise ValueError(f"C100_ONECYCLE_PCT_UP must be finite and in (0, 1), got {onecycle_pct_up!r}")
    onecycle_div_factor = float(os.getenv("C100_ONECYCLE_DIV_FACTOR", "10.0"))
    if not math.isfinite(onecycle_div_factor) or onecycle_div_factor <= 0.0:
        raise ValueError(f"C100_ONECYCLE_DIV_FACTOR must be finite and positive, got {onecycle_div_factor!r}")
    return lr_schedule, onecycle_pct_up, onecycle_div_factor


def parse_bounded_int_env(name, default, min_value, max_value):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    value = int(raw)
    if not min_value <= value <= max_value:
        raise ValueError(f"{name} must be in [{min_value}, {max_value}], got {value}")
    return value


def parse_bounded_float_env(name, default, min_value, max_value, include_max=True):
    raw = os.getenv(name)
    value = default if raw is None or raw.strip() == "" else float(raw)
    upper_ok = value <= max_value if include_max else value < max_value
    if not math.isfinite(value) or value < min_value or not upper_ok:
        upper_bracket = "]" if include_max else ")"
        raise ValueError(f"{name} must be finite and in [{min_value}, {max_value}{upper_bracket}, got {value!r}")
    return value


def lr_multiplier(lr_schedule, progress, onecycle_pct_up, onecycle_div_factor):
    if lr_schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    if progress < onecycle_pct_up:
        warmup_progress = progress / onecycle_pct_up
        return (1.0 / onecycle_div_factor) + (1.0 - (1.0 / onecycle_div_factor)) * warmup_progress
    decay_progress = (progress - onecycle_pct_up) / (1.0 - onecycle_pct_up)
    return 0.5 * (1.0 + math.cos(math.pi * decay_progress))


def gpu_metadata():
    metadata = {
        "torch_device_name": torch.cuda.get_device_name(),
    }
    try:
        query = "index,uuid,name,memory.total,driver_version"
        lines = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()
        metadata["nvidia_smi"] = [line.strip() for line in lines if line.strip()]
    except Exception:
        metadata["nvidia_smi"] = []
    return metadata


def write_repro_metadata(output_dir):
    repo_root = git_output(["git", "rev-parse", "--show-toplevel"])
    status = git_output(["git", "status", "--porcelain"], default="")
    diff = git_output(["git", "diff", "--binary", "HEAD"], default="")
    diff_path = None
    diff_sha256 = None
    if diff:
        diff_path = "git_diff.patch"
        patch_path = output_dir / diff_path
        patch_path.write_text(diff + ("\n" if not diff.endswith("\n") else ""))
        diff_sha256 = hashlib.sha256(diff.encode()).hexdigest()
    metadata = {
        "git": {
            "repo_root": repo_root,
            "repo_url": git_output(["git", "remote", "get-url", "origin"]),
            "branch": git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "commit_sha": git_sha(),
            "dirty": bool(status.strip()),
            "status_porcelain": status.splitlines(),
            "diff_path": diff_path,
            "diff_sha256": diff_sha256,
        },
        "environment": safe_env_snapshot(),
        "gpu": gpu_metadata(),
    }
    write_json(output_dir / "repro_metadata.json", metadata)
    return metadata


def train_once(
    run_name,
    seed,
    model,
    train_images,
    train_labels,
    test_images,
    test_labels,
    epochs,
    batch_size,
    target,
    label_smoothing,
    cutout_size,
    lr_schedule,
    onecycle_pct_up,
    onecycle_div_factor,
    ns_steps,
    muon_momentum,
    muon_weight_decay,
    sgd_momentum,
    evaluate_validation=True,
):
    seed_all(seed)
    reset_model(model)
    muon_params = [p for p in model.parameters() if p.ndim >= 2]
    other_params = [p for p in model.parameters() if p.ndim < 2]
    muon = Muon(
        muon_params,
        lr=float(os.getenv("C100_MUON_LR", "0.035")),
        momentum=muon_momentum,
        weight_decay=muon_weight_decay,
        ns_steps=ns_steps,
    )
    sgd = torch.optim.SGD(other_params, lr=float(os.getenv("C100_BIAS_LR", "0.02")), momentum=sgd_momentum, nesterov=True)
    steps_per_epoch = len(train_images) // batch_size
    total_steps = max(1, int(math.ceil(epochs * steps_per_epoch)))
    step = 0
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    # Timed training begins here. Only architecture, optimizer, and training
    # hyperparameters inside this region are valid speedrun optimization surfaces.
    starter.record()
    model.train()
    while step < total_steps:
        for x, y in batches(train_images, train_labels, batch_size):
            x = normalize(random_crop_flip(x))
            if cutout_size > 0:
                x = random_cutout(x, cutout_size)
            logits = model(x)
            loss = F.cross_entropy(logits.float(), y, label_smoothing=label_smoothing)
            loss.backward()
            progress = step / total_steps
            lr_mult = lr_multiplier(lr_schedule, progress, onecycle_pct_up, onecycle_div_factor)
            muon.param_groups[0]["lr"] = float(os.getenv("C100_MUON_LR", "0.035")) * lr_mult
            sgd.param_groups[0]["lr"] = float(os.getenv("C100_BIAS_LR", "0.02")) * lr_mult
            muon.step(); sgd.step()
            muon.zero_grad(set_to_none=True); sgd.zero_grad(set_to_none=True)
            step += 1
            if step >= total_steps:
                break
    # Timed training ends here. Validation remains an untimed correctness gate.
    ender.record(); torch.cuda.synchronize()
    time_seconds = starter.elapsed_time(ender) * 1e-3
    train_acc = evaluate(model, train_images[:10000], train_labels[:10000])
    if evaluate_validation:
        val_acc = evaluate(model, test_images, test_labels)
        hit = float(val_acc >= target)
        val_text = f"{val_acc:0.4f}"
        hit_text = f"{hit:0.4f}"
    else:
        val_acc = None
        hit = None
        val_text = "skipped"
        hit_text = "skipped"
    print(f"|  {str(run_name).rjust(6)}  |   eval  |     {train_acc:0.4f}  |   {val_text:>8}  |     {hit_text:>8}  |      {time_seconds:0.4f}  |", flush=True)
    return {
        "run": run_name,
        "seed": seed,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "target_hit": hit,
        "time_seconds": time_seconds,
        "epochs": epochs,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "validation_evaluated": evaluate_validation,
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_metrics(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run", "seed", "train_acc", "val_acc", "target_hit", "time_seconds", "epochs", "batch_size", "steps_per_epoch", "total_steps"]
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row[field] for field in fields})


def main():
    runs = int(os.getenv("C100_RUNS", "30"))
    epochs = float(os.getenv("C100_EPOCHS", "16"))
    batch_size = int(os.getenv("C100_BATCH", "1024"))
    target = float(os.getenv("C100_TARGET", "0.70"))
    seed_base = int(os.getenv("C100_SEED_BASE", "880000"))
    sleep_cycles = int(os.getenv("C100_SLEEP_CYCLES", "1000000000"))
    validation_source = os.getenv("C100_VALIDATION_SOURCE", "official")
    dev_per_class = int(os.getenv("C100_DEV_PER_CLASS", "50"))
    dev_split_seed = int(os.getenv("C100_DEV_SPLIT_SEED", "20260703"))
    widths = parse_int_tuple_env("C100_WIDTHS", (64, 128, 256), 3)
    blocks = parse_int_tuple_env("C100_BLOCKS", (2, 2, 2), 3)
    label_smoothing = float(os.getenv("C100_LABEL_SMOOTHING", "0.05"))
    if not math.isfinite(label_smoothing) or not 0.0 <= label_smoothing <= 1.0:
        raise ValueError(f"C100_LABEL_SMOOTHING must be finite and in [0, 1], got {label_smoothing!r}")
    cutout_size = int(os.getenv("C100_CUTOUT_SIZE", "0"))
    if cutout_size < 0:
        raise ValueError(f"C100_CUTOUT_SIZE must be non-negative, got {cutout_size}")
    lr_schedule, onecycle_pct_up, onecycle_div_factor = parse_lr_schedule_env()
    ns_steps = parse_bounded_int_env("C100_NS_STEPS", 5, 1, 8)
    muon_momentum = parse_bounded_float_env("C100_MUON_MOMENTUM", 0.95, 0.0, 1.0, include_max=False)
    muon_weight_decay = parse_bounded_float_env("C100_MUON_WEIGHT_DECAY", 2e-4, 0.0, 0.01)
    sgd_momentum = parse_bounded_float_env("C100_SGD_MOMENTUM", 0.9, 0.0, 1.0, include_max=False)
    output_dir_raw = os.getenv("C100_OUTPUT_DIR", "")
    output_dir = Path(output_dir_raw) if output_dir_raw else None
    train_images, train_labels = load_split("train")
    if validation_source == "official":
        eval_images, eval_labels = load_split("test")
    elif validation_source == "train_dev":
        train_images, train_labels, eval_images, eval_labels = make_train_dev_split(train_images, train_labels, dev_per_class, dev_split_seed)
    else:
        raise ValueError(f"unknown C100_VALIDATION_SOURCE={validation_source!r}")
    compile_enabled = os.getenv("C100_COMPILE", "1") != "0"
    compile_mode = os.getenv("C100_COMPILE_MODE", "default")
    compile_mode_label = compile_mode if compile_enabled else "off"
    model = SimpleResNet(widths=widths, blocks=blocks).cuda().to(torch.float16).to(memory_format=torch.channels_last)
    # Compile is infrastructure, not a record surface. It is paid in warmup and
    # must not be tuned as a benchmark trick; use it only to make the fixed
    # training implementation run normally on the target stack.
    if compile_enabled:
        if compile_mode in ("", "default", "none"):
            model.compile()
        else:
            model.compile(mode=compile_mode)
    config = {
        "model": "simple_resnet_muon",
        "runs": runs,
        "epochs": epochs,
        "batch_size": batch_size,
        "target": target,
        "seed_base": seed_base,
        "sleep_cycles": sleep_cycles,
        "validation_source": validation_source,
        "dev_per_class": dev_per_class if validation_source == "train_dev" else None,
        "dev_split_seed": dev_split_seed if validation_source == "train_dev" else None,
        "train_examples": len(train_images),
        "eval_examples": len(eval_images),
        "compile": compile_enabled,
        "compile_mode": compile_mode_label,
        "widths": list(widths),
        "blocks": list(blocks),
        "muon_lr": float(os.getenv("C100_MUON_LR", "0.035")),
        "bias_lr": float(os.getenv("C100_BIAS_LR", "0.02")),
        "ns_steps": ns_steps,
        "muon_momentum": muon_momentum,
        "muon_weight_decay": muon_weight_decay,
        "sgd_momentum": sgd_momentum,
        "label_smoothing": label_smoothing,
        "cutout_size": cutout_size,
        "lr_schedule": lr_schedule,
        "onecycle_pct_up": onecycle_pct_up,
        "onecycle_div_factor": onecycle_div_factor,
        "no_tta": True,
        "git_sha": git_sha(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(),
    }
    if output_dir is not None:
        write_json(output_dir / "config.json", config)
        write_repro_metadata(output_dir)
    print(f"config model=simple_resnet_muon runs={runs} epochs={epochs} batch={batch_size} target={target} validation_source={validation_source} compile={int(compile_enabled)} compile_mode={compile_mode_label} label_smoothing={label_smoothing} cutout_size={cutout_size} lr_schedule={lr_schedule} onecycle_pct_up={onecycle_pct_up} onecycle_div_factor={onecycle_div_factor} ns_steps={ns_steps} muon_momentum={muon_momentum} muon_weight_decay={muon_weight_decay} sgd_momentum={sgd_momentum} no_tta=1")
    print("---------------------------------------------------------------------------------")
    print("|  run     |  epoch  |  train_acc  |  val_acc  |  target_hit   |  time_seconds  |")
    print("---------------------------------------------------------------------------------")
    warmup = train_once(
        "warmup",
        seed_base - 1,
        model,
        train_images,
        train_labels,
        eval_images,
        eval_labels,
        min(1.0, epochs),
        batch_size,
        target,
        label_smoothing,
        cutout_size,
        lr_schedule,
        onecycle_pct_up,
        onecycle_div_factor,
        ns_steps,
        muon_momentum,
        muon_weight_decay,
        sgd_momentum,
        evaluate_validation=False,
    )
    if output_dir is not None:
        write_json(output_dir / "warmup.json", warmup)
    vals, times = [], []
    for run in range(runs):
        torch.cuda.empty_cache(); torch.cuda.synchronize()
        if sleep_cycles > 0:
            torch.cuda._sleep(sleep_cycles)
        row = train_once(
            run + 1,
            seed_base + run,
            model,
            train_images,
            train_labels,
            eval_images,
            eval_labels,
            epochs,
            batch_size,
            target,
            label_smoothing,
            cutout_size,
            lr_schedule,
            onecycle_pct_up,
            onecycle_div_factor,
            ns_steps,
            muon_momentum,
            muon_weight_decay,
            sgd_momentum,
        )
        if output_dir is not None:
            append_metrics(output_dir / "metrics.csv", row)
        val = row["val_acc"]
        sec = row["time_seconds"]
        vals.append(val); times.append(sec)
        print(f"Mean val accuracy after {run + 1} runs: {sum(vals) / len(vals):.6f} | Mean time: {sum(times) / len(times):.6f}s", end="\r", flush=True)
    print()
    v = torch.tensor(vals); t = torch.tensor(times)
    print("Val accuracies: Mean: %.6f    Std: %.6f    Min: %.6f    Max: %.6f" % (v.mean(), v.std(unbiased=False), v.min(), v.max()))
    print("Times (s):      Mean: %.6f    Std: %.6f    Min: %.6f    Max: %.6f" % (t.mean(), t.std(unbiased=False), t.min(), t.max()))
    print("Target %.4f hit count: %d/%d" % (target, int((v >= target).sum().item()), runs))
    if output_dir is not None:
        summary = {
            "runs": runs,
            "val_acc_mean": float(v.mean().item()),
            "val_acc_std": float(v.std(unbiased=False).item()),
            "val_acc_min": float(v.min().item()),
            "val_acc_max": float(v.max().item()),
            "time_seconds_mean": float(t.mean().item()),
            "time_seconds_std": float(t.std(unbiased=False).item()),
            "time_seconds_min": float(t.min().item()),
            "time_seconds_max": float(t.max().item()),
            "target": target,
            "target_hit_count": int((v >= target).sum().item()),
        }
        write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()

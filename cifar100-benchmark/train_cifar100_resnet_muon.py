
import math
import os
import random
import time
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
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
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
                update = zeropower_newton_schulz(buf)
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


def train_once(run_name, seed, model, train_images, train_labels, test_images, test_labels, epochs, batch_size, target):
    seed_all(seed)
    reset_model(model)
    muon_params = [p for p in model.parameters() if p.ndim >= 2]
    other_params = [p for p in model.parameters() if p.ndim < 2]
    muon = Muon(muon_params, lr=float(os.getenv("C100_MUON_LR", "0.035")), momentum=0.95, weight_decay=2e-4)
    sgd = torch.optim.SGD(other_params, lr=float(os.getenv("C100_BIAS_LR", "0.02")), momentum=0.9, nesterov=True)
    steps_per_epoch = len(train_images) // batch_size
    total_steps = max(1, int(math.ceil(epochs * steps_per_epoch)))
    step = 0
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    starter.record()
    model.train()
    while step < total_steps:
        for x, y in batches(train_images, train_labels, batch_size):
            x = normalize(random_crop_flip(x))
            logits = model(x)
            loss = F.cross_entropy(logits.float(), y, label_smoothing=0.05)
            loss.backward()
            progress = step / total_steps
            lr_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
            muon.param_groups[0]["lr"] = float(os.getenv("C100_MUON_LR", "0.035")) * lr_mult
            sgd.param_groups[0]["lr"] = float(os.getenv("C100_BIAS_LR", "0.02")) * lr_mult
            muon.step(); sgd.step()
            muon.zero_grad(set_to_none=True); sgd.zero_grad(set_to_none=True)
            step += 1
            if step >= total_steps:
                break
    ender.record(); torch.cuda.synchronize()
    time_seconds = starter.elapsed_time(ender) * 1e-3
    val_acc = evaluate(model, test_images, test_labels)
    train_acc = evaluate(model, train_images[:10000], train_labels[:10000])
    hit = float(val_acc >= target)
    print(f"|  {str(run_name).rjust(6)}  |   eval  |     {train_acc:0.4f}  |   {val_acc:0.4f}  |       {hit:0.4f}  |      {time_seconds:0.4f}  |", flush=True)
    return val_acc, time_seconds


def main():
    runs = int(os.getenv("C100_RUNS", "30"))
    epochs = float(os.getenv("C100_EPOCHS", "16"))
    batch_size = int(os.getenv("C100_BATCH", "1024"))
    target = float(os.getenv("C100_TARGET", "0.70"))
    seed_base = int(os.getenv("C100_SEED_BASE", "880000"))
    sleep_cycles = int(os.getenv("C100_SLEEP_CYCLES", "1000000000"))
    train_images, train_labels = load_split("train")
    test_images, test_labels = load_split("test")
    compile_enabled = os.getenv("C100_COMPILE", "1") != "0"
    compile_mode = os.getenv("C100_COMPILE_MODE", "reduce-overhead")
    model = SimpleResNet().cuda().to(torch.float16).to(memory_format=torch.channels_last)
    if compile_enabled:
        model.compile(mode=compile_mode)
    print(f"config model=simple_resnet_muon runs={runs} epochs={epochs} batch={batch_size} target={target} compile={int(compile_enabled)} compile_mode={compile_mode if compile_enabled else off} no_tta=1")
    print("---------------------------------------------------------------------------------")
    print("|  run     |  epoch  |  train_acc  |  val_acc  |  target_hit   |  time_seconds  |")
    print("---------------------------------------------------------------------------------")
    train_once("warmup", seed_base - 1, model, train_images, train_labels, test_images, test_labels, min(1.0, epochs), batch_size, target)
    vals, times = [], []
    for run in range(runs):
        torch.cuda.empty_cache(); torch.cuda.synchronize()
        if sleep_cycles > 0:
            torch.cuda._sleep(sleep_cycles)
        val, sec = train_once(run + 1, seed_base + run, model, train_images, train_labels, test_images, test_labels, epochs, batch_size, target)
        vals.append(val); times.append(sec)
        print(f"Mean val accuracy after {run + 1} runs: {sum(vals) / len(vals):.6f} | Mean time: {sum(times) / len(times):.6f}s", end="\r", flush=True)
    print()
    v = torch.tensor(vals); t = torch.tensor(times)
    print("Val accuracies: Mean: %.6f    Std: %.6f    Min: %.6f    Max: %.6f" % (v.mean(), v.std(unbiased=False), v.min(), v.max()))
    print("Times (s):      Mean: %.6f    Std: %.6f    Min: %.6f    Max: %.6f" % (t.mean(), t.std(unbiased=False), t.min(), t.max()))
    print("Target %.4f hit count: %d/%d" % (target, int((v >= target).sum().item()), runs))


if __name__ == "__main__":
    main()

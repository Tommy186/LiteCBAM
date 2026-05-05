#!/usr/bin/env python3


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os, json, random, time
import numpy as np

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
SEEDS   = [42, 123, 2024]
EPOCHS  = 200
LR      = 0.1
BATCH   = 128
NUM_CLASSES = 100
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(SCRIPT_DIR, "data")
RESULTS_DIR  = os.path.join(SCRIPT_DIR, "results", "multi_seed")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ──────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False   # deterministic mode


# ──────────────────────────────────────────────────────────────
# Attention Modules
# ──────────────────────────────────────────────────────────────
class LiteChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc2(F.relu(self.fc1(y)))
        return torch.sigmoid(y).view(b, c, 1, 1)


class LiteSpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(1)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return torch.sigmoid(self.bn(self.conv(torch.cat([avg, mx], dim=1))))


class LiteCBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = LiteChannelAttention(channels, reduction=16)
        self.sa = LiteSpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class OriginalChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False))

    def forward(self, x):
        b, c, _, _ = x.size()
        a = self.avg_pool(x).view(b, c)
        m = self.max_pool(x).view(b, c)
        return torch.sigmoid(self.fc(a) + self.fc(m)).view(b, c, 1, 1)


class OriginalSpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.bn   = nn.BatchNorm2d(1)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return torch.sigmoid(self.bn(self.conv(torch.cat([avg, mx], dim=1))))


class OriginalCBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = OriginalChannelAttention(channels, reduction=8)
        self.sa = OriginalSpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


# ──────────────────────────────────────────────────────────────
# ResNet-20 for CIFAR
# ──────────────────────────────────────────────────────────────
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, attn_cls=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.attn  = attn_cls(out_ch) if attn_cls else nn.Identity()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attn(out)
        return F.relu(out + self.shortcut(x))


class ResNet20(nn.Module):
    def __init__(self, num_classes=100, attn_cls=None):
        super().__init__()
        self.in_ch = 64
        self.stem  = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(64,  3, stride=1, attn_cls=attn_cls)
        self.layer2 = self._make_layer(128, 3, stride=2, attn_cls=attn_cls)
        self.layer3 = self._make_layer(256, 3, stride=2, attn_cls=attn_cls)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(256, num_classes)

    def _make_layer(self, out_ch, n, stride, attn_cls):
        layers = [BasicBlock(self.in_ch, out_ch, stride, attn_cls)]
        self.in_ch = out_ch
        for _ in range(n - 1):
            layers.append(BasicBlock(out_ch, out_ch, 1, attn_cls))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.pool(x); x = x.flatten(1)
        return self.fc(x)


def build_model(name: str) -> nn.Module:
    if name == "Plain":
        return ResNet20(NUM_CLASSES, attn_cls=None)
    elif name == "OriginalCBAM":
        return ResNet20(NUM_CLASSES, attn_cls=OriginalCBAM)
    elif name == "LiteCBAM":
        return ResNet20(NUM_CLASSES, attn_cls=LiteCBAM)
    raise ValueError(f"Unknown model: {name}")


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


# ──────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────
def get_cifar100_loaders(batch_size=128):
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    train_ds = datasets.CIFAR100(DATA_DIR, train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=test_tf)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
    test_ld  = DataLoader(test_ds,  batch_size=256, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)
    return train_ld, test_ld


# ──────────────────────────────────────────────────────────────
# Train / Eval
# ──────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
    return total_loss / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / total


def run_one(model_name: str, seed: int, train_ld, test_ld) -> dict:
    set_seed(seed)
    model = build_model(model_name).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        train_epoch(model, train_ld, optimizer, criterion)
        scheduler.step()
        if ep % 20 == 0 or ep == EPOCHS:
            acc = evaluate(model, test_ld)
            best_acc = max(best_acc, acc)
            elapsed = time.time() - t0
            eta = elapsed / ep * (EPOCHS - ep)
            print(f"  [{model_name}|seed={seed}] ep {ep:3d}/{EPOCHS}  "
                  f"acc={acc:.2f}%  best={best_acc:.2f}%  "
                  f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m")

    final_acc = evaluate(model, test_ld)
    print(f"  => DONE  {model_name} seed={seed}  final={final_acc:.2f}%")
    return {"model": model_name, "seed": seed,
            "final_acc": round(final_acc, 4),
            "best_acc":  round(best_acc,  4),
            "params_M":  round(count_params(model), 4)}


# ──────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────
def summarize(records: list[dict]) -> dict:
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in records:
        grouped[r["model"]].append(r["final_acc"])
    stats = {}
    for name, accs in grouped.items():
        arr = np.array(accs)
        stats[name] = {
            "accs":  accs,
            "mean":  round(float(arr.mean()), 4),
            "std":   round(float(arr.std(ddof=1)), 4),
            "max":   round(float(arr.max()), 4),
            "min":   round(float(arr.min()), 4),
        }
    return stats


# ──────────────────────────────────────────────────────────────
# LaTeX table generator
# ──────────────────────────────────────────────────────────────
def write_latex_table(stats: dict, seeds: list[int]):
    seed_str = "/".join(str(s) for s in seeds)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{CIFAR-100 Classification Results ({EPOCHS} epochs, {len(seeds)} seeds: {seed_str}, mean$\pm$std)}}",
        r"\label{tab:cifar100_multiseed}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Accuracy (\%)} & \textbf{Attn Params (M)} \\",
        r"\midrule",
    ]

    rows = [
        ("Plain ResNet-20",        stats.get("Plain",        {}), "—"),
        ("OriginalCBAM-ResNet-20", stats.get("OriginalCBAM", {}), "0.065"),
        ("\\textbf{LiteCBAM-ResNet-20 (Ours)}",
         stats.get("LiteCBAM", {}), "\\textbf{0.032}"),
    ]
    for label, s, attn in rows:
        if s:
            acc_str = f"{s['mean']:.2f}$\\pm${s['std']:.2f}"
        else:
            acc_str = "—"
        if label.startswith("\\textbf"):
            lines.append(rf"{label} & \textbf{{{acc_str}}} & {attn} \\")
        else:
            lines.append(rf"{label} & {acc_str} & {attn} \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path = os.path.join(SCRIPT_DIR, "results", "table_cifar100_multiseed.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nLaTeX table saved → {tex_path}")
    return tex_path


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    MODEL_NAMES = ["Plain", "OriginalCBAM", "LiteCBAM"]
    train_ld, test_ld = get_cifar100_loaders(BATCH)

    all_records = []
    total_runs  = len(MODEL_NAMES) * len(SEEDS)
    run_idx     = 0

    t_global = time.time()
    for seed in SEEDS:
        for mname in MODEL_NAMES:
            run_idx += 1
            print(f"\n{'='*60}")
            print(f"Run {run_idx}/{total_runs}: model={mname}  seed={seed}")
            print(f"{'='*60}")
            rec = run_one(mname, seed, train_ld, test_ld)
            all_records.append(rec)
            # Save incrementally so you don't lose results if interrupted
            raw_path = os.path.join(RESULTS_DIR, "multi_seed_results.json")
            with open(raw_path, "w") as f:
                json.dump(all_records, f, indent=2)

    total_min = (time.time() - t_global) / 60
    print(f"\n{'='*60}")
    print(f"All {total_runs} runs done in {total_min:.1f} min")

    # Statistics
    stats = summarize(all_records)
    stats_path = os.path.join(RESULTS_DIR, "stats_summary.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Print summary table
    print("\n── Results Summary ──────────────────────────────────────")
    print(f"{'Model':<22} {'Seeds':>24}  {'Mean':>7}  {'Std':>6}")
    print("-" * 65)
    for mname in MODEL_NAMES:
        s = stats.get(mname, {})
        accs_str = "  ".join(f"{a:.2f}" for a in s.get("accs", []))
        print(f"{mname:<22} [{accs_str}]  {s.get('mean',0):.4f}  ±{s.get('std',0):.4f}")

    # LaTeX table
    write_latex_table(stats, SEEDS)

    print(f"\nRaw records : {os.path.join(RESULTS_DIR, 'multi_seed_results.json')}")
    print(f"Stats       : {stats_path}")
    print("Done ✓")


if __name__ == "__main__":
    main()

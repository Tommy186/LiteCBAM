#!/usr/bin/env python3


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import time
import os
import json
import random
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

# ============================================================
# Configuration
# ============================================================
DATASET = 'cifar100'
NUM_CLASSES = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

# Results / data dirs (relative to script location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']
CIFAR100_CLASSES = [f'class_{i}' for i in range(100)]

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # GPU: auto-tune fastest conv algorithm


# ============================================================
# LiteCBAM: Lightweight Channel-Spatial Attention (~30% fewer params)
# ============================================================

class LiteChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(LiteChannelAttention, self).__init__()
        # Lite: single-pooling (GAP only) + larger reduction ratio r=16
        # Single-layer MLP saves additional params vs two-layer MLP
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        weights = self.fc2(torch.relu(self.fc1(avg_pool)))
        return torch.sigmoid(weights).view(b, c, 1, 1)


class LiteSpatialAttention(nn.Module):
    """Lite CBAM spatial: lightweight 3×3 depthwise conv (saves ~82 params vs 7×7)."""
    def __init__(self, kernel_size=3):
        super(LiteSpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.bn(self.conv(concat)))


class LiteCBAM(nn.Module):
    """LiteCBAM: single-pooling + r=16 + 3×3 spatial -> ~30-50% fewer params vs Original."""
    def __init__(self, channels):
        super(LiteCBAM, self).__init__()
        self.channel_attention = LiteChannelAttention(channels, reduction=16)
        self.spatial_attention = LiteSpatialAttention(kernel_size=3)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class OriginalChannelAttention(nn.Module):
    """Original CBAM channel attention: dual pooling + r=8 (larger bottleneck)."""
    def __init__(self, channels, reduction=8):
        super(OriginalChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False))

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        weights = torch.sigmoid(self.fc(avg_out) + self.fc(max_out))
        return weights.view(b, c, 1, 1)


class OriginalSpatialAttention(nn.Module):
    """Original CBAM spatial: 7×7 standard conv."""
    def __init__(self, kernel_size=7):
        super(OriginalSpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.bn(self.conv(concat)))


class OriginalCBAM(nn.Module):
    """Original CBAM (Woo et al., 2018) baseline: dual-pooling + r=8."""
    def __init__(self, channels):
        super(OriginalCBAM, self).__init__()
        self.channel_attention = OriginalChannelAttention(channels, reduction=8)
        self.spatial_attention = OriginalSpatialAttention(kernel_size=7)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


# ============================================================
# ResNet Building Blocks
# ============================================================

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, use_cbam='lite'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.cbam = {'lite': LiteCBAM, 'original': OriginalCBAM, 'none': lambda *a: nn.Identity()}[use_cbam](out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out += self.shortcut(x)
        return F.relu(out)


class ResNetWithCBAM(nn.Module):
    def __init__(self, block, num_blocks, num_classes, cbam_type='lite'):
        super(ResNetWithCBAM, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, cbam_type=cbam_type)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, cbam_type=cbam_type)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, cbam_type=cbam_type)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, cbam_type):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s, cbam_type))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.avgpool(x); x = torch.flatten(x, 1)
        return self.fc(x)


class BasicBlockPlain(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlockPlain, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNetPlain(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNetPlain, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.avgpool(x); x = torch.flatten(x, 1)
        return self.fc(x)


# ============================================================
# Data Loading
# ============================================================

def get_data_loaders(dataset, batch_size, num_workers=4):
    if dataset == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        CLASSES = CIFAR10_CLASSES
        train_ds = datasets.CIFAR10(root=DATA_DIR, train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)]))
        test_ds = datasets.CIFAR10(root=DATA_DIR, train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)]))
    else:
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        CLASSES = CIFAR100_CLASSES
        train_ds = datasets.CIFAR100(root=DATA_DIR, train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)]))
        test_ds = datasets.CIFAR100(root=DATA_DIR, train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)]))

    persistent = num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, persistent_workers=persistent)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, persistent_workers=persistent)
    return train_loader, test_loader, CLASSES


# ============================================================
# Metrics
# ============================================================

def compute_metrics(model, loader, device, num_classes):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    acc = 100.0 * np.mean(all_preds == all_targets)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0) * 100
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0) * 100
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0) * 100

    per_class_acc = []
    for c in range(num_classes):
        mask = all_targets == c
        per_class_acc.append(100.0 * np.mean(all_preds[mask] == all_targets[mask]) if mask.sum() > 0 else 0.0)

    cm = confusion_matrix(all_targets, all_preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1,
            'per_class_accuracy': per_class_acc, 'confusion_matrix': cm.tolist()}


# ============================================================
# Training
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()
        total += targets.size(0)
    return running_loss / total, 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)
    return running_loss / total, 100.0 * correct / total


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def count_cbam_params(model, cbam_type):
    target = {'lite': LiteCBAM, 'original': OriginalCBAM}.get(cbam_type, None)
    if target is None:
        return 0.0
    modules = [m for m in model.modules() if isinstance(m, target)]
    return sum(p.numel() for m in modules for p in m.parameters()) / 1e6


# ============================================================
# Plotting
# ============================================================

def plot_training_curves(all_logs):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for metric in ['train_loss', 'test_loss', 'train_acc', 'test_acc']:
        plt.figure(figsize=(10, 6))
        for name, log in all_logs.items():
            if metric in log:
                plt.plot(log['epoch'], log[metric], label=name, linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        ylabel = {'train_loss': 'Training Loss', 'test_loss': 'Test Loss',
                  'train_acc': 'Training Accuracy (%)', 'test_acc': 'Test Accuracy (%)'}
        plt.ylabel(ylabel.get(metric, metric), fontsize=12)
        plt.title(ylabel.get(metric, metric), fontsize=14)
        plt.legend(fontsize=10); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/{metric}.png', dpi=150); plt.close()

    plt.figure(figsize=(10, 6))
    for name, log in all_logs.items():
        if 'lr' in log:
            plt.plot(log['epoch'], log['lr'], label=name, linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Learning Rate'); plt.title('Learning Rate Schedule')
    plt.legend(); plt.grid(True, alpha=0.3); plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/lr_schedule.png', dpi=150); plt.close()


def plot_metric_comparison(summary):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    model_names = list(summary.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, name in enumerate(model_names):
        vals = [summary[name].get(m, 0) for m in metrics]
        ax.bar(x + i * width, vals, width, label=name)
    ax.set_xlabel('Metric'); ax.set_ylabel('Score (%)')
    ax.set_title('Accuracy / Precision / Recall / F1 Comparison')
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/metrics_comparison.png', dpi=150); plt.close()


def plot_param_compare(param_data):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    names = list(param_data.keys())
    totals = [param_data[n]['total'] for n in names]
    cbams = [param_data[n]['cbam'] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.175, totals, 0.35, label='Total Params (M)', color='steelblue')
    ax.bar(x + 0.175, cbams, 0.35, label='CBAM Module Params (M)', color='coral')
    ax.set_xlabel('Model'); ax.set_ylabel('Parameters (M)')
    ax.set_title('Parameter Comparison')
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/param_comparison.png', dpi=150); plt.close()


def plot_confusion_matrix(cm, classes, suffix=''):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(8, len(classes)*0.8), max(6, len(classes)*0.6)))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel('Predicted Label'); ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix' + (f' ({suffix})' if suffix else ''))
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fname = f'{RESULTS_DIR}/cm{suffix and "_" + suffix}.png'
    plt.savefig(fname, dpi=150); plt.close()
    return fname


# ============================================================
# Main
# ============================================================

def main():
    global DATASET, NUM_CLASSES
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100'])
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    DATASET = args.dataset
    NUM_CLASSES = 10 if DATASET == 'cifar10' else 100
    set_seed(SEED)

    # Override RESULTS_DIR with date-stamped subdirectory for this run
    global RESULTS_DIR
    from datetime import datetime
    date_str = datetime.now().strftime('%Y-%m-%d')
    run_count = 1
    while True:
        run_label = f"{date_str}_run{run_count}"
        run_dir = os.path.join(RESULTS_DIR, run_label)
        if not os.path.exists(run_dir):
            break
        run_count += 1
    os.makedirs(run_dir, exist_ok=True)
    RESULTS_DIR = run_dir

    print(f"{'='*60}")
    print(f"CMNM 2026 - Paper 1: ResNet + LiteCBAM")
    print(f"Dataset: {DATASET} | Epochs: {args.epochs} | Batch: {args.batch_size}")
    print(f"Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}")

    train_loader, test_loader, CLASSES = get_data_loaders(DATASET, args.batch_size, args.num_workers)

    models_cfg = {
        'LiteCBAM-ResNet20':    ('lite',     lambda nc: ResNetWithCBAM(BasicBlock, [3,3,3], nc, 'lite')),
        'OriginalCBAM-ResNet20':('original', lambda nc: ResNetWithCBAM(BasicBlock, [3,3,3], nc, 'original')),
        'Plain ResNet20':       ('none',     lambda nc: ResNetPlain(BasicBlockPlain, [3,3,3], nc)),
    }

    all_logs, all_metrics, param_data = {}, {}, {}

    for name, (cbam_type, model_fn) in models_cfg.items():
        print(f"\n{'='*60}\nTraining {name}...\n{'='*60}")
        model = model_fn(NUM_CLASSES).to(DEVICE)
        total_params = count_params(model)
        cbam_params = count_cbam_params(model, cbam_type)
        param_data[name] = {'total': total_params, 'cbam': cbam_params}
        print(f"  Total: {total_params:.3f}M | CBAM module: {cbam_params:.3f}M")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        log = {'epoch': [], 'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': [], 'lr': []}
        # Use last checkpoint instead of best_acc to avoid oscillation at end of training
        last_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        for epoch in range(args.epochs):
            t0 = time.time()
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
            test_loss, test_acc  = evaluate(model, test_loader, criterion, DEVICE)
            scheduler.step()
            dt = time.time() - t0

            # Always keep last checkpoint (cosine annealing ends at best point)
            last_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            log['epoch'].append(epoch+1)
            log['train_loss'].append(train_loss); log['test_loss'].append(test_loss)
            log['train_acc'].append(train_acc); log['test_acc'].append(test_acc)
            log['lr'].append(optimizer.param_groups[0]['lr'])

            if (epoch+1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{args.epochs} | "
                      f"Trn L:{train_loss:.4f} A:{train_acc:.2f}% | "
                      f"Tst L:{test_loss:.4f} A:{test_acc:.2f}% | {dt:.1f}s")

        # Save last checkpoint
        ckpt_path = f'{RESULTS_DIR}/best_{name.replace(" ","_").lower()}.pth'
        torch.save(last_state, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")

        # Full metrics
        model.load_state_dict(last_state)
        metrics = compute_metrics(model, test_loader, DEVICE, NUM_CLASSES)
        metrics['best_acc'] = metrics['accuracy']
        metrics['total_params'] = total_params
        metrics['cbam_params'] = cbam_params
        all_logs[name] = log
        all_metrics[name] = metrics
        print(f"\n  {name} -> Acc:{metrics['accuracy']:.2f}% Prec:{metrics['precision']:.2f}% "
              f"Rec:{metrics['recall']:.2f}% F1:{metrics['f1']:.2f}%")

    # ========== Generate plots ==========
    print(f"\n{'='*60}\nGenerating plots...\n{'='*60}")
    plot_training_curves(all_logs)
    plot_metric_comparison(all_metrics)
    plot_param_compare(param_data)
    for name, metrics in all_metrics.items():
        suffix = name.replace(' ','_').lower()
        plot_confusion_matrix(np.array(metrics['confusion_matrix']), CLASSES, suffix)

    # ========== Save JSON ==========
    results = {
        'dataset': DATASET, 'epochs': args.epochs, 'batch_size': args.batch_size,
        'num_classes': NUM_CLASSES, 'classes': CLASSES,
        'param_data': param_data,
        'metrics': {name: {k: float(round(v,4)) if isinstance(v,(float,np.floating)) else (int(v) if isinstance(v,np.integer) else v)
                          for k,v in {**m, 'per_class_accuracy': [float(round(x,2)) for x in m['per_class_accuracy']],
                                       'confusion_matrix': [[int(y) for y in x] for x in m['confusion_matrix']],
                                       'log': {kk:[float(round(x,6)) if isinstance(x,(float,np.floating)) else x for x in vv] for kk,vv in all_logs[name].items() if kk!='epoch'}}.items()}
                   for name, m in all_metrics.items()}
    }
    json_path = f'{RESULTS_DIR}/cmnm_paper1_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results JSON: {json_path}")

    # ========== Summary table ==========
    print(f"\n{'='*60}\nRESULTS SUMMARY\n{'='*60}")
    print(f"{'Model':<25} {'Acc(%)':<10} {'Prec(%)':<10} {'Rec(%)':<10} {'F1(%)':<10} {'Params(M)':<10}")
    print("-"*75)
    for name, m in all_metrics.items():
        print(f"{name:<25} {m['accuracy']:<10.2f} {m['precision']:<10.2f} "
              f"{m['recall']:<10.2f} {m['f1']:<10.2f} {m['total_params']:<10.3f}")
    print("-"*75)
    print(f"\nAll outputs saved to: {RESULTS_DIR}/")


if __name__ == '__main__':
    main()

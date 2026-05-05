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
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

# ============================================================
SEED = 42
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']
CIFAR100_CLASSES = [f'class_{i}' for i in range(100)]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# ============================================================
# Channel Attention Variants
# ============================================================

class ChannelAttentionSingle(nn.Module):
    """Single pooling (GAP only) channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
    def forward(self, x):
        b, c, _, _ = x.size()
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        w = self.fc2(torch.relu(self.fc1(avg)))
        return torch.sigmoid(w).view(b, c, 1, 1)

class ChannelAttentionDual(nn.Module):
    """Dual pooling (GAP + GMP) channel attention."""
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
        avg = self.avg_pool(x).view(b, c)
        mx = self.max_pool(x).view(b, c)
        w = torch.sigmoid(self.fc(avg) + self.fc(mx))
        return w.view(b, c, 1, 1)

# ============================================================
# Spatial Attention Variants
# ============================================================

class SpatialAttentionStandard(nn.Module):
    """Standard 7x7 conv spatial attention."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(1)
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg, mx], dim=1)
        return torch.sigmoid(self.bn(self.conv(concat)))

class SpatialAttentionLight(nn.Module):
    """Lightweight 3x3 conv spatial attention."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(1)
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg, mx], dim=1)
        return torch.sigmoid(self.bn(self.conv(concat)))

# ============================================================
# LiteCBAM Variant: single pool + r=16 + 3x3 spatial (our main design)
# ============================================================
class LiteCBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel = ChannelAttentionSingle(channels, reduction=16)
        self.spatial = SpatialAttentionLight()
    def forward(self, x):
        return x * self.channel(x) * self.spatial(x)

# ============================================================
# Ablation Variants
# ============================================================

class CBAM_SinglePool_r8(nn.Module):
    """Ablation: single pool + r=8."""
    def __init__(self, channels):
        super().__init__()
        self.channel = ChannelAttentionSingle(channels, reduction=8)
        self.spatial = SpatialAttentionLight()
    def forward(self, x):
        return x * self.channel(x) * self.spatial(x)

class CBAM_SinglePool_r32(nn.Module):
    """Ablation: single pool + r=32."""
    def __init__(self, channels):
        super().__init__()
        self.channel = ChannelAttentionSingle(channels, reduction=32)
        self.spatial = SpatialAttentionLight()
    def forward(self, x):
        return x * self.channel(x) * self.spatial(x)

class CBAM_DualPool_r8_7x7(nn.Module):
    """Original CBAM: dual pool + r=8 + 7x7 spatial."""
    def __init__(self, channels):
        super().__init__()
        self.channel = ChannelAttentionDual(channels, reduction=8)
        self.spatial = SpatialAttentionStandard()
    def forward(self, x):
        return x * self.channel(x) * self.spatial(x)

class CBAM_DualPool_r16_7x7(nn.Module):
    """Ablation: dual pool + r=16 + 7x7 spatial."""
    def __init__(self, channels):
        super().__init__()
        self.channel = ChannelAttentionDual(channels, reduction=16)
        self.spatial = SpatialAttentionStandard()
    def forward(self, x):
        return x * self.channel(x) * self.spatial(x)

class CBAM_DualPool_r8_3x3(nn.Module):
    """Ablation: dual pool + r=8 + 3x3 spatial."""
    def __init__(self, channels):
        super().__init__()
        self.channel = ChannelAttentionDual(channels, reduction=8)
        self.spatial = SpatialAttentionLight()
    def forward(self, x):
        return x * self.channel(x) * self.spatial(x)

class CBAM_SinglePool_r16_7x7(nn.Module):
    """Ablation: single pool + r=16 + 7x7 spatial."""
    def __init__(self, channels):
        super().__init__()
        self.channel = ChannelAttentionSingle(channels, reduction=16)
        self.spatial = SpatialAttentionStandard()
    def forward(self, x):
        return x * self.channel(x) * self.spatial(x)

# ============================================================
# ResNet Building Blocks
# ============================================================

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1, cbam_fn=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.cbam = cbam_fn(out_ch) if cbam_fn else nn.Identity()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch * self.expansion))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        return F.relu(out + self.shortcut(x))

class PlainBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch * self.expansion))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))

class ResNetWithCBAM(nn.Module):
    def __init__(self, block, num_blocks, num_classes, cbam_fn):
        super().__init__()
        self.in_ch = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1, cbam_fn)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2, cbam_fn)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2, cbam_fn)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)
    def _make_layer(self, block, out_ch, num_blocks, stride, cbam_fn):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_ch, out_ch, s, cbam_fn))
            self.in_ch = out_ch * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.avgpool(x); x = torch.flatten(x, 1)
        return self.fc(x)

class ResNetPlain(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super().__init__()
        self.in_ch = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)
    def _make_layer(self, block, out_ch, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_ch, out_ch, s))
            self.in_ch = out_ch * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.avgpool(x); x = torch.flatten(x, 1)
        return self.fc(x)

# ============================================================
# Data
# ============================================================

def get_data_loaders(dataset, batch_size, num_workers=4):
    if dataset == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        CLASSES = CIFAR10_CLASSES
        train_ds = datasets.CIFAR10(root='./data', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)]))
        test_ds = datasets.CIFAR10(root='./data', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)]))
    else:
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        CLASSES = CIFAR100_CLASSES
        train_ds = datasets.CIFAR100(root='./data', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)]))
        test_ds = datasets.CIFAR100(root='./data', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)]))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, persistent_workers=True)
    return train_loader, test_loader, CLASSES

# ============================================================
# Training & Evaluation
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
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
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)
    return running_loss / total, 100.0 * correct / total

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
    all_preds, all_targets = np.array(all_preds), np.array(all_targets)
    acc = 100.0 * np.mean(all_preds == all_targets)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0) * 100
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0) * 100
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0) * 100
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100'])
    parser.add_argument('--lr', type=float, default=0.05)  # lower LR for stability
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_CLASSES = 10 if args.dataset == 'cifar10' else 100
    set_seed(SEED)

    date_str = datetime.now().strftime('%Y-%m-%d')
    run_count = 1
    while True:
        run_label = f"{date_str}_ablation_run{run_count}"
        run_dir = os.path.join(RESULTS_DIR, run_label)
        if not os.path.exists(run_dir): break
        run_count += 1
    os.makedirs(run_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"Ablation Study | Dataset: {args.dataset} | Epochs: {args.epochs} | LR: {args.lr}")
    print(f"Results: {run_dir}")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}")

    train_loader, test_loader, CLASSES = get_data_loaders(args.dataset, args.batch_size, args.num_workers)

    # Model configurations: (name, cbam_fn or None for plain)
    model_cfgs = [
        ('Plain ResNet20', None),
        ('CBAM (Dual, r=8, 7x7) [Original]', CBAM_DualPool_r8_7x7),
        ('CBAM (Dual, r=16, 7x7)', CBAM_DualPool_r16_7x7),
        ('CBAM (Dual, r=8, 3x3)', CBAM_DualPool_r8_3x3),
        ('CBAM (Single, r=16, 3x3) [Ours]', LiteCBAM),
        ('CBAM (Single, r=8, 3x3)', CBAM_SinglePool_r8),
        ('CBAM (Single, r=16, 7x7)', CBAM_SinglePool_r16_7x7),
        ('CBAM (Single, r=32, 3x3)', CBAM_SinglePool_r32),
    ]

    all_metrics, all_logs = {}, {}

    for name, cbam_fn in model_cfgs:
        print(f"\n{'='*50}\nTraining {name}...\n{'='*50}")
        if cbam_fn:
            model = ResNetWithCBAM(BasicBlock, [3,3,3], NUM_CLASSES, cbam_fn)
        else:
            model = ResNetPlain(PlainBlock, [3,3,3], NUM_CLASSES)
        model = model.to(DEVICE)
        total_params = count_params(model)
        print(f"  Total params: {total_params:.3f}M")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        log = {'epoch': [], 'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': [], 'lr': []}
        best_acc = 0.0
        last_state = None

        for epoch in range(args.epochs):
            t0 = time.time()
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
            test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
            scheduler.step()

            last_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if test_acc > best_acc:
                best_acc = test_acc

            log['epoch'].append(epoch+1)
            log['train_loss'].append(train_loss)
            log['test_loss'].append(test_loss)
            log['train_acc'].append(train_acc)
            log['test_acc'].append(test_acc)
            log['lr'].append(optimizer.param_groups[0]['lr'])

            if (epoch+1) % 20 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{args.epochs} | "
                      f"Trn L:{train_loss:.4f} A:{train_acc:.2f}% | "
                      f"Tst L:{test_loss:.4f} A:{test_acc:.2f}% | Best:{best_acc:.2f}%")

        # Load last checkpoint and evaluate
        model.load_state_dict(last_state)
        metrics = compute_metrics(model, test_loader, DEVICE, NUM_CLASSES)
        metrics['total_params'] = float(total_params)
        metrics['best_acc'] = float(best_acc)
        # Convert numpy types to native Python for JSON serialization
        for kk in ['accuracy', 'precision', 'recall', 'f1', 'best_acc', 'total_params', 'cbam_params']:
            if kk in metrics:
                metrics[kk] = float(round(metrics[kk], 4))
        metrics['per_class_accuracy'] = [float(round(x, 2)) for x in metrics.get('per_class_accuracy', [])]
        metrics['confusion_matrix'] = [[int(y) for y in x] for x in metrics.get('confusion_matrix', [])]
        all_metrics[name] = metrics
        all_logs[name] = {kk: [float(round(x, 6)) if isinstance(x, (float, np.floating)) else x for x in vv] for kk, vv in log.items()}

        torch.save(last_state, f'{run_dir}/best_{name.replace(" ","_").replace("(","").replace(")","").replace(",","").replace("=","_").lower()}.pth')

        print(f"\n  {name} -> Acc:{metrics['accuracy']:.2f}% Prec:{metrics['precision']:.2f}% "
              f"Rec:{metrics['recall']:.2f}% F1:{metrics['f1']:.2f}%")

    # Save results
    results = {
        'dataset': args.dataset, 'epochs': args.epochs, 'batch_size': args.batch_size,
        'lr': args.lr, 'num_classes': NUM_CLASSES,
        'classes': CLASSES,
        'metrics': {k: {kk: float(round(vv, 4)) if isinstance(vv, (float, np.floating)) else vv
                        for kk, vv in {**v, 'log': {kk: [float(round(x, 4)) if isinstance(x, (float, np.floating)) else x for x in vv] for kk, vv in all_logs[k].items() if kk != 'epoch'}}.items()}
                    for k, v in all_metrics.items()}
    }
    json_path = f'{run_dir}/ablation_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"MODEL                                                  ACC(%)      PREC(%)      F1(%)       PARAMS(M)")
    print("-"*70)
    for name, m in all_metrics.items():
        print(f"{name:<45} {m['accuracy']:<10.2f} {m['precision']:<10.2f} {m['f1']:<10.2f} {m['total_params']:<10.3f}")
    print("-"*70)
    print(f"\nAll outputs saved to: {run_dir}/")

if __name__ == '__main__':
    main()

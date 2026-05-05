#!/usr/bin/env python3

import os, json, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec

# ============================================================
# Style: IEEE publication quality
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 11,
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__)) + '/results'
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


def load_json():
    """Load results from JSON. Exit if not found."""
    path = os.path.join(RESULTS_DIR, 'cmnm_paper1_results.json')
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run cmnm_paper1_code.py first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def Papers1Figure_accuracy_curves(results):
    """Figure 1: Training & Test Accuracy curves for all models."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), sharey=False)

    model_names = list(results['metrics'].keys())
    linestyles = {'train': '-', 'test': '--'}
    labels_used = set()

    for ax, (curve_type, ls) in zip(axes, linestyles.items()):
        for i, (name, metrics) in enumerate(results['metrics'].items()):
            log = metrics.get('log', {})
            key = f'{curve_type}_acc'
            if key not in log:
                continue
            epochs = range(1, len(log[key]) + 1)
            short_name = name.replace('ResNet20', 'R20').replace('-20', '')
            color = COLORS[i % len(COLORS)]
            ax.plot(epochs, log[key], label=short_name, color=color,
                     linestyle=ls, linewidth=1.5, marker='', alpha=0.85)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{curve_type.capitalize()} Accuracy')
        ax.set_ylim([None, 102])
        ax.legend(loc='lower right', framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Figure 1: Training and Test Accuracy vs Epoch on CIFAR-10', y=-0.02, fontsize=9, style='italic')
    plt.tight_layout()
    out = f'{RESULTS_DIR}/fig1_accuracy_curves.png'
    fig.savefig(out, dpi=300)
    plt.close()
    print(f"  Saved: {out}")
    return out


def Papers1Figure_loss_curves(results):
    """Figure 2: Training & Test Loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

    for ax, (curve_type, ls) in zip(axes, [('train_loss', '-'), ('test_loss', '--')]):
        for i, (name, metrics) in enumerate(results['metrics'].items()):
            log = metrics.get('log', {})
            if curve_type not in log:
                continue
            epochs = range(1, len(log[curve_type]) + 1)
            short_name = name.replace('ResNet20', 'R20').replace('-20', '')
            ax.plot(epochs, log[curve_type], label=short_name,
                     color=COLORS[i % len(COLORS)], linestyle=ls, linewidth=1.5, alpha=0.85)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{curve_type.replace("_", " ").capitalize()}')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Figure 2: Training and Test Loss vs Epoch on CIFAR-10', y=-0.02, fontsize=9, style='italic')
    plt.tight_layout()
    out = f'{RESULTS_DIR}/fig2_loss_curves.png'
    fig.savefig(out, dpi=300)
    plt.close()
    print(f"  Saved: {out}")
    return out


def Papers1Figure_metrics_bar(results):
    """Figure 3: Accuracy/Precision/Recall/F1 bar chart — data matches table exactly."""
    metrics_list = ['accuracy', 'precision', 'recall', 'f1']
    model_names = list(results['metrics'].keys())
    n_models = len(model_names)
    n_metrics = len(metrics_list)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(n_metrics)
    width = 0.8 / n_models

    for i, name in enumerate(model_names):
        vals = [results['metrics'][name].get(m, 0) for m in metrics_list]
        offset = (i - n_models/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.95, label=name, color=COLORS[i % len(COLORS)], alpha=0.85)
        # Annotate bars for our model
        if 'LiteCBAM' in name:
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')

    ax.set_xlabel('Metric')
    ax.set_ylabel('Score (%)')
    ax.set_title('Accuracy / Precision / Recall / F1 Comparison on CIFAR-10')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics_list])
    ax.set_ylim([85, 97])
    ax.legend(loc='upper right', framealpha=0.9, ncol=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('Figure 3: Classification Metrics Comparison (values = Table 1 data)', y=-0.02, fontsize=9, style='italic')
    plt.tight_layout()
    out = f'{RESULTS_DIR}/fig3_metrics_comparison.png'
    fig.savefig(out, dpi=300)
    plt.close()
    print(f"  Saved: {out}")
    return out


def Papers1Figure_param_comparison(results):
    """Figure 4: Parameter comparison — total params & CBAM module params."""
    model_names = list(results['metrics'].keys())
    totals = [results['metrics'][n]['total_params_M'] for n in model_names]
    modules = [results['metrics'][n].get('cbam_params_M', 0) for n in model_names]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

    # Left: total params
    bars1 = axes[0].bar(range(len(model_names)), totals, color=COLORS[:len(model_names)], alpha=0.8)
    axes[0].set_xticks(range(len(model_names)))
    short = [n.replace('ResNet20', '\nR20').replace('-20', '') for n in model_names]
    axes[0].set_xticklabels(short, fontsize=7)
    axes[0].set_ylabel('Total Parameters (M)')
    axes[0].set_title('Total Network Parameters')
    for bar, val in zip(bars1, totals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{val:.2f}M', ha='center', va='bottom', fontsize=7)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Right: CBAM module params
    bars2 = axes[1].bar(range(len(model_names)), modules, color=COLORS[:len(model_names)], alpha=0.8)
    axes[1].set_xticks(range(len(model_names)))
    axes[1].set_xticklabels(short, fontsize=7)
    axes[1].set_ylabel('CBAM Module Parameters (M)')
    axes[1].set_title('Attention Module Parameters')
    for bar, val in zip(bars2, modules):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                     f'{val:.3f}M', ha='center', va='bottom', fontsize=7)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    fig.suptitle('Figure 4: Parameter Comparison (values = Table 2 data)', y=-0.02, fontsize=9, style='italic')
    plt.tight_layout()
    out = f'{RESULTS_DIR}/fig4_param_comparison.png'
    fig.savefig(out, dpi=300)
    plt.close()
    print(f"  Saved: {out}")
    return out


def Papers1Figure_confusion_matrix(results):
    """Figure 5: Confusion matrix for proposed model (LiteCBAM-ResNet20)."""
    proposed_key = 'LiteCBAM-ResNet20'
    if proposed_key not in results['metrics']:
        print(f"  WARNING: {proposed_key} not found, skipping confusion matrix")
        return None

    cm = np.array(results['metrics'][proposed_key]['confusion_matrix'])
    classes = results.get('classes', ['airplane','automobile','bird','cat','deer',
                                        'dog','frog','horse','ship','truck'])
    n = len(classes)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap='Blues', aspect='auto')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Count', fontsize=8)

    # Ticks & labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=7.5)
    ax.set_yticklabels(classes, fontsize=7.5)
    ax.set_xlabel('Predicted Label', fontsize=9)
    ax.set_ylabel('True Label', fontsize=9)
    ax.set_title(f'Confusion Matrix: {proposed_key} on CIFAR-10', fontsize=10)

    # Annotate cells
    max_val = cm.max()
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            text_color = 'white' if val > max_val * 0.55 else 'black'
            fontsize = 8 if n <= 10 else 6
            ax.text(j, i, f'{val}', ha='center', va='center', color=text_color, fontsize=fontsize)

    plt.tight_layout()
    out = f'{RESULTS_DIR}/fig5_confusion_matrix.png'
    fig.savefig(out, dpi=300)
    plt.close()
    print(f"  Saved: {out}")
    return out


def Papers1Figure_per_class_accuracy(results):
    """Figure 6: Per-class accuracy for all models."""
    proposed_key = 'LiteCBAM-ResNet20'
    if proposed_key not in results['metrics']:
        return None

    per_class = results['metrics'][proposed_key].get('per_class_accuracy', {})
    if not per_class:
        return None

    classes = results.get('classes', ['airplane','automobile','bird','cat','deer',
                                        'dog','frog','horse','ship','truck'])
    accs = [per_class.get(c, 0) for c in classes]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(range(len(classes)), accs, color='#1f77b4', alpha=0.8)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Per-Class Accuracy: {proposed_key} on CIFAR-10', fontsize=10)
    ax.set_ylim([70, 102])
    ax.axhline(y=sum(accs)/len(accs), color='red', linestyle='--', linewidth=1, label=f'Mean={sum(accs)/len(accs):.1f}%')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}', ha='center', va='bottom', fontsize=6.5)

    plt.tight_layout()
    out = f'{RESULTS_DIR}/fig6_per_class_accuracy.png'
    fig.savefig(out, dpi=300)
    plt.close()
    print(f"  Saved: {out}")
    return out


def Papers1Figure_LR_schedule(results):
    """Figure 7: Learning rate schedule."""
    proposed_key = 'LiteCBAM-ResNet20'
    if proposed_key not in results['metrics']:
        return None
    log = results['metrics'][proposed_key].get('log', {})
    if 'lr' not in log:
        return None

    fig, ax = plt.subplots(figsize=(5, 2.8))
    epochs = range(1, len(log['lr']) + 1)
    ax.plot(epochs, log['lr'], color='#1f77b4', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Cosine Annealing Learning Rate Schedule')
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out = f'{RESULTS_DIR}/fig7_lr_schedule.png'
    fig.savefig(out, dpi=300)
    plt.close()
    print(f"  Saved: {out}")
    return out


def generate_all_figures():
    """Main: load results and generate all figures."""
    print("=" * 60)
    print("CMNM 2026 Paper 1 — Generating Publication Figures")
    print("=" * 60)

    results = load_json()
    print(f"\nLoaded: {results.get('dataset')}, {results.get('epochs')} epochs")
    print(f"Models: {list(results['metrics'].keys())}")

    # Verify table-figure data consistency
    print("\n[Data Consistency Check]")
    for name, m in results['metrics'].items():
        print(f"  {name}: Acc={m['accuracy']:.2f}%  F1={m['f1']:.2f}%  "
              f"Total={m['total_params_M']:.3f}M  CBAM={m.get('cbam_params_M', 0):.3f}M")

    print("\n[Generating Figures]")
    figures = []
    figures.append(Papers1Figure_accuracy_curves(results))
    figures.append(Papers1Figure_loss_curves(results))
    figures.append(Papers1Figure_metrics_bar(results))
    figures.append(Papers1Figure_param_comparison(results))
    figures.append(Papers1Figure_confusion_matrix(results))
    figures.append(Papers1Figure_per_class_accuracy(results))
    figures.append(Papers1Figure_LR_schedule(results))

    print(f"\n✅ All {len([f for f in figures if f])} figures saved to {RESULTS_DIR}/")
    return results


if __name__ == '__main__':
    generate_all_figures()

# LiteCBAM: A Lightweight Channel-Spatial Attention Module for Efficient Image Classification with ResNet

A PyTorch implementation of LiteCBAM, a lightweight version of CBAM (Convolutional Block Attention Module) that reduces parameters by ~30-50% while maintaining comparable accuracy.

## Features

- **LiteCBAM**: Lightweight channel-spatial attention module
- **CIFAR-10/100** dataset support
- **Multi-seed experiments** for robust results
- **Ablation studies** for component analysis

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Tommy186/LiteCBAM.git
cd LiteCBAM

# Install dependencies
pip install -r requirements.txt

# Train the model
cd paper1_litecbam
python cmnm_paper1_code.py --dataset cifar100 --epochs 200
```

## Project Structure

```
LiteCBAM/
├── paper1_litecbam/
│   ├── cmnm_paper1_code.py    # Main training script
│   ├── ablation_study.py       # Ablation experiments
│   ├── run_multi_seed.py       # Multi-seed training
│   ├── plot_results.py         # Generate plots
│   └── results/                # Experimental results
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- PyTorch 2.1.0
- CUDA 12.1 (for GPU training)

See `requirements.txt` for full dependencies.

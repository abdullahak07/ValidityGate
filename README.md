# 🧠 Hessian-Guided Gradient Unlearning

> 📄 Complete implementation of  
> **“Hessian-Guided Gradient Unlearning” (XX 2026)**

---

## 🚀 Overview

This repository reproduces all experiments, tables, and figures from the paper.

### 🔬 Method Pipeline

The approach integrates three complementary stages:

- ⚡ **Gradient-based Unlearning**  
  Fast first-order ascent on forget loss  

- 🧮 **Hessian-based Refinement**  
  Second-order influence approximation via conjugate gradient  

- 🛠️ **Post-processing Stabilization**
  - 🧊 **Masking** → suppress neurons responding to forgotten class  
  - 🎨 **Inpainting** → smooth parameter discontinuities  

---

## 🗂️ Repository Structure

```
hessian_unlearning/
│
├── main.py            # 🚪 Entry point (all experiments + figures)
├── config.py          # ⚙️ Hyperparameters and paths
├── models.py          # 🧱 ResNet-18, VGG-16
├── datasets.py        # 📦 Dataset loaders
├── trainer.py         # 🏋️ Training + checkpoints
├── unlearning.py      # ♻️ Unlearning algorithms
├── metrics.py         # 📊 TA, FA, RA, MIA, Privacy
├── experiments.py     # 📈 Table 1–5 generation
├── visualization.py   # 🎯 Figure 2–7 plotting
│
├── data/              # 📁 Datasets
├── checkpoints/       # 💾 Model weights
├── results/           # 📑 CSV tables
├── plots/             # 🖼️ Figures
│
└── requirements.txt   # 📌 Dependencies
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

### 📋 Requirements
- 🐍 Python ≥ 3.10  
- 🔥 PyTorch ≥ 2.0  
- 🎮 GPU recommended (RTX 3090 / 4090)

---

## ▶️ Quick Start

### 🧪 Smoke Test (CPU ~5 min)
```bash
python main.py --quick
```

### 🎯 Demo (CIFAR-10)
```bash
python main.py --demo --dataset cifar10 --method hessian_guided
```

### 📦 Full Reproduction
```bash
python main.py
```

### 🧾 Run Specific Experiments
```bash
python main.py --exp table1
python main.py --exp table2
python main.py --exp table3
python main.py --exp table4
python main.py --exp table5
python main.py --exp figures
```

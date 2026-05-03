# Chapter 1 — Setup & Tooling

## What You Need to Know Before We Start

### "What is Python, really?"

Python is just a **language for telling the computer what to do**. You write instructions in a `.py` file, and Python "reads" them and executes them one by one. If you've written any Python before — even just `print("hello")` — you're ready.

### "What is a GPU and why do I need one?"

**Analogy:** Imagine you need to paint 10,000 tiny tiles.

- A **CPU** is like a master artist who paints tiles ONE at a time — precise but slow.
- A **GPU** is like 10,000 art students who each paint ONE tile simultaneously — faster, even though each student is "dumber" than the master.

Training neural networks involves millions of **identical, independent math operations** (matrix multiplications). GPUs have thousands of small cores designed exactly for this. A GPU can be 50-100x faster than CPU for training.

**Do you absolutely need a GPU?** No — our tiny test model will run on CPU, just very slowly (minutes vs hours). For real training, a GPU is essential.

| Your Hardware | What You Can Train | Approximate Speed |
|---|---|---|
| CPU only | Tiny model (4 layers, 256 dims) | Hours |
| Apple M1/M2/M3 | Small model (12 layers, 768 dims) | Hours |
| RTX 3060/4060 (12GB) | GPT-2 small (124M params) | Few hours |
| RTX 3090/4090 (24GB) | GPT-2 medium (350M) | Few hours |
| A100 (80GB) | GPT-2 large (774M) | Hours |

### "What is a virtual environment?"

A virtual environment (`venv`) is like a **clean, empty kitchen** just for this project. Without it, you'd be mixing your project's ingredients (Python packages) with everything else on your computer — leading to conflicts when two projects need different versions of the same package.

```bash
# Create a clean kitchen
python -m venv gpt_env

# Step into it
source gpt_env/bin/activate          # Mac/Linux
# OR:
gpt_env\Scripts\activate             # Windows

# Now pip install only affects this kitchen
# To leave: type `deactivate`
```

### "What is pip?"

`pip` is Python's **package installer**. It downloads code other people have written (libraries) from the internet and installs them into your environment. Think of it as an "app store" for Python code.

### "What is PyTorch?"

PyTorch is the framework we'll use to build our neural network. It provides:

| PyTorch Feature | What It Does | Analogy |
|---|---|---|
| `torch.Tensor` | Multi-dimensional arrays | Like NumPy arrays, but can live on GPU |
| `torch.nn.Module` | Building blocks for networks | LEGO pieces you snap together |
| `torch.optim` | Algorithms that update weights | The "learning" part of machine learning |
| `autograd` | Automatic gradient calculation | Does calculus for you automatically |
| `DataLoader` | Feeds data efficiently | A conveyor belt delivering training data |

## Installation — Step by Step

```bash
# Step 1: Create the virtual environment
python -m venv gpt_env

# Step 2: Activate it
source gpt_env/bin/activate          # Mac/Linux
# gpt_env\Scripts\activate           # Windows

# Step 3: Install PyTorch (choose the right one)
# For NVIDIA GPU (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For NVIDIA GPU (CUDA 12.1 - newer cards like RTX 40 series):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For Apple Silicon (M1/M2/M3):
# pip install torch torchvision torchaudio

# For CPU only:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Step 4: Install remaining packages
pip install tiktoken datasets numpy matplotlib

# Step 5: Verify everything works
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## What Each Library Does (In Detail)

| Library | What It Does | Why We Need It |
|---|---|---|
| **torch** | Core PyTorch: tensors, GPU ops, autograd | The foundation — everything else builds on this |
| **tiktoken** | Fast BPE tokenizer from OpenAI | Same tokenizer GPT-3.5/4 use. Written in Rust, extremely fast |
| **datasets** (HuggingFace) | Downloads + caches training data | Saves us from manually downloading and parsing Wikipedia |
| **numpy** | Fast numerical arrays on CPU | For quick data manipulation (though PyTorch handles most) |
| **matplotlib** | Creates charts and graphs | To visualize our training loss — is the model learning? |
| **math** (built-in) | sqrt, sin, cos, pi | Mathematical constants for positional encoding |
| **time** (built-in) | Measure elapsed time | Track training speed in tokens/second |
| **os** (built-in) | Create directories, save files | Save model checkpoints so we don't lose progress |

## Our Complete Import Block

```python
# ===== WHAT: Standard Python libraries =====
import math              # WHY: sqrt(), sin(), cos() for positional encoding math
import time              # WHY: measure training speed (tokens per second)
import os                # WHY: create directories, save/load model checkpoint files
from dataclasses import dataclass  # WHY: clean config class — no messy dictionaries

# ===== WHAT: NumPy — the CPU array library =====
import numpy as np       # WHY: fast numerical operations on CPU arrays
                         #      (mostly used for quick data checks, not heavy lifting)

# ===== WHAT: PyTorch — the neural network framework =====
import torch             # WHY: core library — tensors, GPU support, autograd
import torch.nn as nn               # WHY: neural network building blocks:
                                     #      Linear (dense layers), Embedding (lookup tables),
                                     #      Dropout (regularization), ModuleList (stacking layers)
import torch.nn.functional as F     # WHY: stateless functions used inside forward():
                                     #      softmax (convert to probabilities),
                                     #      cross_entropy (measure prediction error),
                                     #      silu (SwiGLU activation function)
from torch.utils.data import Dataset, DataLoader  # WHY: efficient data pipeline
#                                  Dataset = define how to load one sample
#                                  DataLoader = batch them, shuffle, prefetch

# ===== WHAT: tiktoken — OpenAI's fast BPE tokenizer =====
import tiktoken          # WHY: same Byte Pair Encoding tokenizer as GPT-3.5/GPT-4
                         #      Written in Rust, ~100x faster than pure Python tokenizers
                         #      Handles 50K+ vocabulary efficiently

# ===== WHAT: HuggingFace datasets — download training text =====
from datasets import load_dataset    # WHY: one line to download WikiText-103
                                     #      Handles caching (only downloads once),
                                     #      streaming (for datasets too big for disk),
                                     #      and format conversion automatically

# ===== WHAT: matplotlib — plot loss curves =====
import matplotlib.pyplot as plt      # WHY: visualize training progress
                                     #      Is the loss going down? Is it plateauing?
                                     #      A picture is worth 1,000 log lines

# ===== WHAT: Quick verification =====
# WHY: Always test your environment before writing 500 lines of code.
#      A missing import now saves hours of debugging later.
print("All imports ready!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory:      {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

**Expected output (with GPU):**
```
All imports ready!
PyTorch version: 2.1.0
CUDA available:  True
GPU:             NVIDIA GeForce RTX 3090
GPU Memory:      24.0 GB
```

**Expected output (CPU only):**
```
All imports ready!
PyTorch version: 2.1.0
CUDA available:  False
```

If you see the GPU output, you're ready to train. If you see CPU only, training will work — just slower. Either way, let's continue.

---

## How to Think About the Rest of This Guide

Every chapter follows this pattern:

1. **Analogy** — Explain the concept in plain English (like teaching a 5-year-old)
2. **Math** — Show the actual formulas and why they work
3. **Code** — Every single line annotated with WHAT it does and WHY
4. **Visual** — Diagram or worked example showing data flowing through

If you ever feel lost, go back to the analogy. If the code feels overwhelming, focus on the WHAT/WHY comments — they're designed to be read top-to-bottom like a story.

---

**Previous:** [Chapter 0 — Overview](00_overview.md)
**Next:** [Chapter 2 — Tokenization](02_tokenization.md)

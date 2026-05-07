# 🧠 How to Train Your GPT

> *A guide to building a world-class language model from absolute scratch. Taught like you're five. Built like you're an engineer.*
>
> *I made this with the goal of learning something I didn't understand completely. Specifically the attention part. I use AI a lot to understand key concepts and verifying them.*

<p align="center">
  <img src="https://img.shields.io/badge/chapters-12-blue" alt="12 chapters">
  <img src="https://img.shields.io/badge/lines-3%2C900%2B-green" alt="3,900+ lines">
  <img src="https://img.shields.io/badge/code%20commented-100%25-brightgreen" alt="100% commented">
  <img src="https://img.shields.io/badge/prerequisite-python%20basics-orange" alt="Python basics only">
  <img src="https://img.shields.io/badge/architecture-LLaMA%203%20style-purple" alt="LLaMA 3 style">
  <img src="https://img.shields.io/badge/purpose-learning%20only-lightgrey" alt="Learning only">
</p>

---

## 📖 What Is This?

This is a **12-chapter, 3,900+ line interactive textbook** that teaches you how to build, train and run a modern language model from absolute scratch. The same family of architecture behind ChatGPT, Claude, LLaMA and Mistral.

You won't just read about Transformers. You'll **write every line yourself**: tokenizer, embeddings, attention, training loop, inference engine. Every single line annotated to explain **what** it does and **why** it's there.

---

## 🤔 Why This Exists

Most ML tutorials fall into one of two traps:

| ❌ Too Shallow | ❌ Too Academic | ✅ This Guide |
|---|---|---|
| `model = GPT().fit(data)` | 40-page papers, dense notation | 5-year-old analogies → full working code |
| You learn to call APIs | Assumes PhD in ML | Zero ML experience required |
| No understanding of internals | No worked examples | Every line annotated with WHAT & WHY |

**The goal:** After finishing, you won't just know that attention "works". You'll understand the variance argument behind `1/√d_k`. How RoPE captures relative position through rotation. Why pre-norm beats post-norm for deep networks. And exactly where every gradient flows during backpropagation.

---

## 👥 Who Is This For?

| 🧑‍💻 You Are... | 📚 You Need... |
|---|---|
| A Python developer curious about how ChatGPT actually works | Basic Python (functions, classes, lists). No ML experience |
| A student who wants to deeply understand Transformers | Willingness to read ~3,500 lines of commented code |
| An engineer evaluating LLM architectures | Understanding of tradeoffs (RoPE vs learned, RMSNorm vs LayerNorm) |
| Someone who got lost at "attention" in other tutorials | Party analogy + worked numeric example with real numbers |

**🔧 Prerequisites:** Python basics (variables, functions, classes, `pip install`). That's it. No calculus, no linear algebra, no PyTorch experience required. We teach those as we go.

---

## 🗺️ Chapters

| Chapter | What You'll Learn |
|---|---|
| **[0: Overview](chapters/00_overview.md)** | What is a GPT? The big picture |
| **[1: Setup](chapters/01_setup.md)** | Install tools, GPU vs CPU, venv, PyTorch basics |
| **[2: Tokenization](chapters/02_tokenization.md)** | BPE walkthrough: how "unbelievably" becomes tokens |
| **[3: Embeddings](chapters/03_embeddings.md)** | How numbers become meaning. king − man + woman = queen |
| **[4: Positional Encoding](chapters/04_positional_encoding.md)** | RoPE: why LLaMA rotates vectors, not adds numbers |
| **[5: Attention](chapters/05_attention.md)** | ⭐ THE CORE. Q,K,V, scaling, causal mask, 8-step walkthrough |
| **[6: Transformer Block](chapters/06_transformer_block.md)** | RMSNorm, SwiGLU, residuals, pre-norm vs post-norm |
| **[7: Complete GPT Model](chapters/07_gpt_model.md)** | 151M parameter model (with SwiGLU), weight tying, logits explained |
| **[8: Training Pipeline](chapters/08_training.md)** | Cross-entropy, backprop, AdamW, cosine warmup, mixed precision |
| **[9: Inference](chapters/09_inference.md)** | KV cache, temperature, top-k/p, beam search, repetition penalty |
| **[10: Full Script](chapters/10_full_script.md)** | Runnable `main.py`: everything in one file |
| **[11: Glossary](chapters/11_glossary.md)** | Architecture provenance table, parameter breakdown |

> ⭐ **Start with [Chapter 0](chapters/00_overview.md) and read sequentially.** Each builds on the previous.

---

## 🏗️ What You'll Build

| 🧩 Component | 📝 Lines | 💡 What You'll Understand |
|---|---|---|
| **BPE Tokenizer** | ~60 | How GPT-4 splits "unbelievably" → "un" + "believ" + "ably" |
| **Embeddings** | ~30 | How "cat" and "dog" end up near each other in 768D space |
| **RoPE** | ~70 | Why LLaMA rotates vectors instead of adding position numbers |
| **Multi-Head Attention** | ~120 | The exact 8-step computation behind every modern LLM |
| **Transformer Block** | ~50 | Why residual connections are the "gradient highway" |
| **Full GPT Model** | ~200 | 151M parameter model with SwiGLU, weight tying and pre-norm |
| **Training Pipeline** | ~250 | AdamW, cosine warmup, mixed precision, gradient accumulation |
| **Inference Engine** | ~80 | KV cache, temperature, top-k/p, beam search |

> 💎 **~860 lines of core model code, ~2,600 lines of explanation and diagrams**

---

## 🏛️ Architecture

This guide implements the **latest publicly-documented** decoder-only Transformer:

| 🧬 Technique | 📦 Source Model | ⚡ Why It Matters |
|---|---|---|
| **RoPE** | LLaMA, Mistral, Qwen | Relative position without learned parameters |
| **RMSNorm** | LLaMA, Mistral, Gemma | 15% faster than LayerNorm, equally effective |
| **SwiGLU** | PaLM, LLaMA, Gemini | Learns which information to pass or block |
| **Pre-Norm** | GPT-3, all modern | Stable training at 100+ layers |
| **AdamW** | GPT-3+ | Better generalization than vanilla Adam |
| **BPE** | GPT-2/3/4 | Handles any text. Even unseen words and emoji |
| **Weight Tying** | GPT-2/3 | Saves 30% parameters, improves training signal |
| **Mixed Precision** | All production LLMs | 2× speed, half memory, same quality |

> ℹ️ GPT-4 and Claude architectures are proprietary/undisclosed. This teaches the best publicly-confirmed architecture: what LLaMA 3, Mistral and Qwen 2.5 use.

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/raiyanyahya/how-to-train-your-gpt.git
cd how-to-train-your-gpt

# 2. Create environment
python -m venv gpt_env
source gpt_env/bin/activate          # Mac/Linux
# gpt_env\Scripts\activate           # Windows

# 3. Install dependencies (CPU version — for GPU see below)
pip install torch tiktoken datasets numpy matplotlib --index-url https://download.pytorch.org/whl/cpu

# 4. Verify GPU (optional but recommended)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 5. Start reading!
open chapters/00_overview.md
```

Run the training script:

```bash
python main.py
```

This uses the tiny config (d_model=256, 4 layers) by default. Training takes a few minutes on CPU. For the GPT-2 scale config (151M params, 768 dims, 12 layers), edit the config in main.py and uncomment the larger configuration.

> 💻 The default config uses a tiny model (d_model=256, 4 layers, 17M params) that runs in minutes on CPU. For the full GPT-2 scale (151M params, 768 dims, 12 layers), edit the config in `main.py` and uncomment the larger configuration. You'll need a GPU for that one.

---

## 📓 Jupyter Notebooks

Alongside the textbook, each chapter has a companion notebook you can run live. These strip away the explanations and give you pure, clean code that executes from top to bottom. If the textbook teaches you why, the notebooks let you see it happen.

We're going to run this whole project on a very small dataset so you can watch training happen in minutes rather than weeks. Every notebook is self-contained — open it, run all cells, and you'll see the model learn in real time.

```bash
# Install everything you need
pip install jupyter tiktoken torch numpy datasets matplotlib --index-url https://download.pytorch.org/whl/cpu

# Start with chapter 2 (tokenization)
jupyter notebook notebooks/02_tokenization.ipynb
```

Notebooks live in the `notebooks/` directory, one per chapter. Open any of them and hit **Cell → Run All**.

---

## 📖 How to Read

Each chapter follows the same **4-step structure**:

| Step | Format | Purpose |
|---|---|---|
| 1️⃣ **Analogy** | Plain English, 5-year-old level | Build intuition before math |
| 2️⃣ **Worked Example** | Real numbers traced through | See exactly what happens |
| 3️⃣ **Annotated Code** | Every line: `WHAT` + `WHY` | Understand every decision |
| 4️⃣ **Diagram** | Mermaid flowchart or ASCII | Visualize data flow |

> 💡 **Tip:** Lost in the code? Jump back to the analogy. Confused by the math? Skip to the worked example.

---

## ✨ What Makes This Different

| Aspect | 😴 Typical Tutorial | 🔥 This Guide |
|---|---|---|
| **Explanation depth** | "Attention helps the model focus" | 8-step worked example with real numbers + variance math + causal mask visualization |
| **Code comments** | Few or none | Every single line: WHAT + WHY |
| **Modern techniques** | GPT-2 style (2019) | LLaMA 3 style (2024): RoPE, RMSNorm, SwiGLU |
| **Training** | Uses HuggingFace Trainer | Full custom loop: AdamW, cosine warmup, mixed precision, grad accumulation |
| **Inference** | `model.generate()` | Temperature, top-k, top-p, beam search, KV cache explained |
| **Target audience** | ML engineers | Python developers with zero ML experience |
| **Diagrams** | None | Mermaid flowcharts + ASCII matrices + worked examples |

---

## 🎯 Skills You'll Gain

- ✅ Explain how GPT-4 tokenizes text using BPE
- ✅ Understand why RoPE, RMSNorm and SwiGLU replaced older techniques
- ✅ Compute attention scores manually for a 3-token sentence
- ✅ Debug a Transformer training loop (loss spikes, flat lines, overfitting)
- ✅ Choose sampling parameters (temperature, top_k, top_p) for different use cases
- ✅ Understand why KV caching is critical for production inference
- ✅ Read modern ML papers with confidence (you'll recognize every component)

---

## 🔮 Next Steps After Finishing

| Experiment | What to Change | What You'll Learn |
|---|---|---|
| **Bigger model** | `num_layers` 12 → 24 | How depth improves reasoning |
| **More data** | Add BookCorpus, C4, The Pile | Impact of data quality and diversity |
| **Flash Attention** | Install `flash-attn`, swap attention | 2-5× faster training, longer context |
| **Grouped Query Attention** | Set `num_kv_heads` < `num_heads` | How Mistral achieves efficient inference |
| **LoRA fine-tuning** | Add low-rank adapter layers | Customize models without full retraining |
| **RLHF / DPO** | Add reward model training | How ChatGPT learns to follow instructions |
| **KV Cache** | Implement persistent key-value storage | 500× faster text generation |
| **Mixture of Experts** | Route tokens through different FFN experts | How GPT-4 scales to trillions of params |

---

## 📁 File Structure

```
📦 how-to-train-your-gpt/
├── 📄 README.md              ← You are here
└── 📂 chapters/
    ├── 🏠 00_overview.md     ← What is a GPT? Why build one?
    ├── 🔧 01_setup.md        ← Install tools, GPU vs CPU, venv basics
    ├── 🔪 02_tokenization.md ← BPE walkthrough, EOS tokens, emoji handling
    ├── 🧊 03_embeddings.md   ← How numbers become meaning, king − man + woman
    ├── 📍 04_positional_encoding.md ← RoPE math, numerical example, theta
    ├── 🧠 05_attention.md    ← ⭐ THE CORE (713 lines). Q,K,V, scaling, causal mask
    ├── 🧱 06_transformer_block.md ← RMSNorm, SwiGLU, residuals, pre-norm vs post
    ├── 🏗️ 07_gpt_model.md    ← Complete 151M model, weight tying, logits explained
    ├── 🏋️ 08_training.md     ← Cross-entropy, backprop, AdamW, cosine warmup
    ├── 🎤 09_inference.md    ← KV cache, temperature, top-k/p, beam search
    ├── 📜 10_full_script.md  ← Runnable main.py
    └── 📊 11_glossary.md     ← Architecture provenance, parameter breakdown
```

---

<p align="center">
  <i>"Any sufficiently explained technology is indistinguishable from magic. Until you build it yourself."</i>
</p>

<p align="center">
  <sub>⭐ Star this repo if you found it useful | 🐛 Issues & PRs welcome | 📖 Happy learning!</sub>
</p>

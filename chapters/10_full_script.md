# Chapter 10 — The Full Script (main.py)

## Complete Runnable Script

```python
# ===== main.py =====
# WHAT: The complete script to train a GPT from scratch.
# WHY: Everything we've built comes together here.
#      Run this file and watch your GPT learn!

import torch

# Import all our modules (assuming they're in the same directory)
# In practice, you'd organize these into separate .py files:
#   tokenizer.py -> SimpleTokenizer, TokenizerConfig
#   model.py     -> GPT, GPTConfig, TransformerBlock, MultiHeadAttention,
#                   RMSNorm, SwiGLU, RotaryPositionalEmbedding, create_causal_mask
#   training.py  -> TextDataset, load_training_data, CosineWarmupScheduler,
#                   create_optimizer, train, plot_loss
#   inference.py -> load_checkpoint, generate_text


def main():
    """
    WHAT: Main entry point.
    WHY: Setup -> data -> model -> train -> generate
    """

    # ===== 1. CONFIGURATION =====
    print("How to Train Your GPT — Starting up!\n")

    # WHAT: Choose model size
    # WHY: Tiny for testing, small for real training

    # TINY MODEL (for quick testing, ~5 min on GPU)
    config = GPTConfig(
        vocab_size=50257,
        d_model=256,          # Small embedding
        num_heads=4,          # Few heads
        num_layers=4,         # Few layers
        max_seq_len=256,      # Short context
        batch_size=4,         # Tiny batch
        grad_accum_steps=2,   # Effective batch = 8
        max_steps=5000,       # Quick training
    )

    # SMALL MODEL (GPT-2 small scale — recommended!)
    # config = GPTConfig(
    #     vocab_size=50257,
    #     d_model=768,          # GPT-2 small dimension
    #     num_heads=12,         # 12 heads x 64 dims
    #     num_layers=12,        # GPT-2 small depth
    #     max_seq_len=1024,     # 1024 tokens context
    #     max_steps=50000,      # Decent results
    #     batch_size=4,
    #     grad_accum_steps=8,   # Effective batch = 32
    # )

    # ===== 2. DEVICE SETUP =====
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon (MPS)")
    else:
        device = torch.device("cpu")
        print("Warning: Using CPU — training will be very slow!")

    # ===== 3. TOKENIZER =====
    tokenizer = SimpleTokenizer()

    # ===== 4. LOAD DATA =====
    print("\nLoading training data...")
    # WHAT: Start with 5000 documents for quick testing
    # WHY: Increase to None for full dataset once everything works
    texts = load_training_data(max_samples=5000)
    train_dataset = TextDataset(texts, tokenizer, max_seq_len=config.max_seq_len)

    # ===== 5. CREATE MODEL =====
    print("\nCreating model...")
    model = GPT(config)
    print(f"   Total parameters: {model.get_num_params():,}")
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"   Trainable: {trainable:,}")

    # ===== 6. TRAIN =====
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    loss_history = train(model, train_dataset, config, device)

    # ===== 7. PLOT LOSS =====
    plot_loss(loss_history)

    # ===== 8. GENERATE TEXT =====
    print("\n" + "="*60)
    print("GENERATING SAMPLE TEXT")
    print("="*60 + "\n")

    prompts = [
        "The history of artificial intelligence",
        "In the beginning, the universe",
        "The most important scientific discovery",
    ]

    for prompt in prompts:
        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=80,
            temperature=0.8,
            top_k=50,
            device=device,
        )
        print(f"Prompt: {prompt}")
        print(f"Output: {generated}")
        print("-" * 60)
        print()

    print("Done! Your GPT is trained and ready.")


if __name__ == "__main__":
    main()
```

## Running the Training

```bash
# From your project directory:
python main.py

# Watch GPU usage in another terminal:
watch -n 1 nvidia-smi
```

---

**Previous:** [Chapter 9 — Inference](09_inference.md)
**Next:** [Chapter 11 — Glossary](11_glossary.md)

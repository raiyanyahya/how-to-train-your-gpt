# Chapter 10 — The Full Script (main.py)

The complete runnable training script lives at the root of the repo as `main.py`.

It contains everything inline — tokenizer, model, training loop, and inference — so you can clone the repo and run it with one command.

```bash
python main.py
```

By default it uses a tiny model (d_model=256, 4 layers, 4 heads) that trains on 5,000 Wikipedia articles for 500 steps. This takes about 2-5 minutes on CPU or a few seconds on GPU. The script also contains a commented-out GPT-2 small configuration (768 dims, 12 layers, 12 heads) for when you have a GPU.

After training it saves a checkpoint to `checkpoints/model.pt`, plots a loss curve, and generates sample text from a few prompts.

---

**Previous:** [Chapter 9 — Inference](09_inference.md)
**Next:** [Chapter 11 — Glossary](11_glossary.md)

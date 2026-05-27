# Cheatsheet: Everything You Need in One Place

## Model sizes

| Model | Layers | d_model | Heads | Head Dim | Params |
|---|---|---|---|---|---|
| Tiny (our default) | 4 | 256 | 4 | 64 | 17.1M |
| GPT-2 Small | 12 | 768 | 12 | 64 | 124M |
| GPT-2 Medium | 24 | 1024 | 16 | 64 | 350M |
| GPT-2 Large | 36 | 1280 | 20 | 64 | 774M |
| GPT-3 1.3B | 24 | 2048 | 32 | 64 | 1.3B |
| GPT-3 6.7B | 32 | 4096 | 32 | 128 | 6.7B |
| GPT-3 175B | 96 | 12288 | 96 | 128 | 175B |
| LLaMA 7B | 32 | 4096 | 32 | 128 | 7B |
| LLaMA 13B | 40 | 5120 | 40 | 128 | 13B |
| LLaMA 70B | 80 | 8192 | 64 | 128 | 70B |

## Parameter count formula

For our model with SwiGLU and weight tying:

```
Embedding:        vocab_size × d_model
Per block QKV:    3 × d_model × d_model
Per block Output: d_model × d_model
Per block SwiGLU: 3 × d_model × (4 × d_model)
Per block Norms:  2 × d_model
LM Head:          0 (weight tied with embedding)
```

For GPT-2 Small equivalent (768 dims, 12 layers, 50257 vocab):
```
152M total = 38.6M (embedding) + 113.3M (12 blocks) + 768 (norm)
```

## Training hyperparameters

| Parameter | Our Default | Range | Notes |
|---|---|---|---|
| Learning rate | 3e-4 | 1e-5 to 5e-4 | Lower for fine-tuning (1e-5 to 5e-5) |
| Weight decay | 0.1 | 0.01 to 0.3 | Only on 2D+ params. Not norms/biases |
| Betas | (0.9, 0.95) | : | LLaMA defaults. Don't change |
| Epsilon | 1e-8 | : | Never needs tuning |
| Warmup steps | 2000 | 500 to 5000 | ~5% of total steps |
| Max steps | 100K | Depends on data | More data = more steps |
| Batch size | 8 (× 4 accum) | As large as fits | Effective batch = 32 |
| Grad clip | 1.0 | 0.5 to 2.0 | 1.0 is standard |
| Dropout | 0.1 | 0.0 to 0.3 | Higher = more regularization |

## Sampling parameters

| Parameter | Default | Use Case |
|---|---|---|
| temperature | 0.8 | General. Lower = focused (0.3-0.5), higher = creative (1.2-1.5) |
| top_k | 50 | Eliminate nonsense tokens. 0 = disabled |
| top_p | 0.9 | Adaptive cutoff. 1.0 = disabled |
| max_new_tokens | 100 | Depends on task. Longer for stories, shorter for answers |

## Key formulas

### Embedding
```
output = self.embed(token_ids)
No scaling with RoPE (LLaMA convention)
```

### RoPE rotation angles
```
θ_i = p / (10000^(2i / d_head))
cos_cached = cos(θ_i)  for all p and i
sin_cached = sin(θ_i)  for all p and i
x_rotated = x * cos + rotate_half(x) * sin
```

### Attention
```
Q = input @ W_q   [batch, heads, seq, head_dim]
K = input @ W_k   [batch, heads, seq, head_dim]
V = input @ W_v   [batch, heads, seq, head_dim]

Q_rot = RoPE(Q, seq_len)
K_rot = RoPE(K, seq_len)

scores = Q_rot @ K_rot^T / sqrt(head_dim)   [batch, heads, seq, seq]
scores = masks(scores)                        (causal: upper triangle = -inf)
weights = softmax(scores, dim=-1)              (row sum = 1.0)
output = weights @ V                          [batch, heads, seq, head_dim]

concat heads → [batch, seq, d_model]
output = concat @ W_o
```

### RMSNorm
```
rms = sqrt(mean(x^2) + eps)
output = x / rms * weight
```

### SwiGLU FFN
```
h = SiLU(input @ W1)     [batch, seq, 4×d_model]
g = input @ W2           [batch, seq, 4×d_model]
output = (h * g) @ W3     [batch, seq, d_model]
```

### Cross entropy loss
```
loss = -log(P_model(true_token))
For random model: loss ≈ ln(vocab_size) = ln(50257) ≈ 10.82
```

### AdamW update
```
momentum = β1 × momentum + (1 - β1) × gradient
velocity = β2 × velocity + (1 - β2) × gradient^2

m_hat = momentum / (1 - β1^step)
v_hat = velocity / (1 - β2^step)

weight = weight * (1 - lr * weight_decay)     (decoupled)
weight = weight - lr * m_hat / (sqrt(v_hat) + ε)
```

### Cosine warmup schedule
```
if step < warmup:
    lr = max_lr * step / warmup
else:
    progress = (step - warmup) / (total - warmup)
    lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + cos(π * progress))
```

### Gradient clipping
```
total_norm = sqrt(Σ grad_i^2)
if total_norm > max_norm:
    scale = max_norm / total_norm
    grad_i *= scale for all i
```

## Memory requirements (approximate)

| Model Size | bfloat16 Weights | Optimizer (AdamW) | Total (no batch) |
|---|---|---|---|
| 17M (tiny) | 34 MB | 136 MB | 170 MB |
| 152M (GPT-2 small) | 304 MB | 1.2 GB | 1.5 GB |
| 7B (LLaMA) | 14 GB | 56 GB | 70 GB |
| 70B (LLaMA) | 140 GB | 560 GB | 700 GB |

Optimizer states = 2 × params × 4 bytes (float32). Gradients = params × 4 bytes.
Total training memory ≈ 2×weights + 2×optimizer + gradients + activations.
With bfloat16 weights: ~8 bytes per parameter for full training.

## Shape conventions

| Component | Input Shape | Output Shape |
|---|---|---|
| Tokenizer | text | [batch, seq] |
| Embedding | [batch, seq] | [batch, seq, d_model] |
| RoPE | [batch, heads, seq, head_dim] | Same |
| Attention (internal Q/K/V) | [batch, heads, seq, head_dim] | Same |
| Attention (external) | [batch, seq, d_model] | [batch, seq, d_model] |
| RMSNorm | [batch, seq, d_model] | Same |
| SwiGLU | [batch, seq, d_model] | Same |
| TransformerBlock | [batch, seq, d_model] | Same |
| LM Head | [batch, seq, d_model] | [batch, seq, vocab_size] |
| Loss | logits + targets | scalar |

## What a good loss looks like

```
10.8: Random. Model knows nothing
9.0:  Starting to learn word frequencies
7.0:  Learning basic grammar
5.0:  Coherent phrases emerging
3.0:  Decent sentences. Still makes mistakes
2.0:  Good model. Plausible text
1.5:  Very good. Near production quality
<1.0: Overfitting or memorization (on small datasets)
```

## Dataset sizes

| Dataset | Documents | Tokens | Size |
|---|---|---|---|
| WikiText-103 | 28,475 | 103M | 516 MB |
| BookCorpus | ~11,000 | 985M | 5 GB |
| C4 | 364M | 156B | 305 GB |
| The Pile | : | 825B | 825 GB |

## Common error messages

| Error | Meaning |
|---|---|
| `CUDA out of memory` | Batch too big or model too big. Reduce batch size or use gradient accumulation |
| `nan in loss` | Learning rate too high or gradients exploded. Lower LR or add gradient clipping |
| `loss = 10.82` | Model is random. Normal at step 0. If it stays there check optimizer and loss function |
| `size mismatch` | Shape error. Check batch/seq/head dimensions. Common bug in reshape/permute |
| `weights_only load failed` | PyTorch 2.6+ requires `weights_only=False` for checkpoints with custom classes |

## Key files in the repo

```
📦 how-to-train-your-gpt/
├── main.py                         ← Single file training. Run this.
├── requirements.txt                ← Dependencies
├── chapters/                       ← 12 chapter textbook
├── notebooks/                      ← 8 chapter notebooks + attention viz + colab
├── fine-tuning/                    ← 7 files on fine-tuning + LoRA notebook
├── explanations and examples WIP/  ← 18 topic deep dives
│   ├── attention.md                ← Most popular. 363 lines, worked example
│   ├── the_complete_story.md       ← 1166 lines. Everything connected
│   ├── a_tokens_journey.md         ← 301 lines. Follow one sentence
│   └── ... (15 more topics)
└── README.md
```

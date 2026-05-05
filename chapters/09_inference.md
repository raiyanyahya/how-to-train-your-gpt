# Chapter 9 — Inference: Making It Talk

## How Generation Is Different from Training

| Aspect | Training | Inference (Generation) |
|---|---|---|
| **Goal** | Learn to predict next token correctly | Actually generate new text |
| **Input** | Full sequence with target | Prompt only (no target) |
| **Teacher forcing** | Yes — show correct answer | No — model generates its own future |
| **Forward pass** | One pass for whole sequence | One pass PER new token |
| **Speed** | Fast (batch parallel) | Slow (sequential token-by-token) |
| **Causal mask** | Prevents seeing future tokens | Future tokens don't exist yet |
| **Dropout** | Active (for regularization) | Disabled (for deterministic output) |
| **Gradient** | Yes (backward pass) | No (no learning happening) |

## The Naive Generation Loop

Our simplest implementation (loops over all tokens every step):

```python
for _ in range(max_new_tokens):
    # Recompute the ENTIRE sequence every time!  ← wasteful!
    logits, _ = model(input_ids)           # Process ALL tokens
    next_token = sample(logits[:, -1, :])  # Only use LAST prediction
    input_ids = torch.cat([input_ids, next_token], dim=1)
```

**Problem:** Token 500 has already been processed 499 times by the time we add token 501!

## KV Cache — The Biggest Speed-Up

### The Key Insight

During autoregressive generation, previously computed Keys and Values don't change. Token 0's K and V are the same whether we're predicting token 1 or token 500.

**Without KV Cache:** Recompute K and V for ALL tokens every step.  
**With KV Cache:** Compute K,V for NEW token only. Append to cache. Reuse old ones.

```
Step 1: Process "The"           → Store K["The"], V["The"] in cache
Step 2: Process "cat"           → Reuse K,V for "The", compute K,V for "cat"
Step 3: Process "sat"           → Reuse K,V for "The","cat", compute for "sat"
...
Step 500: Process "mat"         → Reuse K,V for 499 tokens, compute 1 new
```

### Speed Improvement

| Sequence Length | Without KV Cache | With KV Cache | Speedup |
|---|---|---|---|
| 100 | 5,050 ops | 100 ops | 50× |
| 500 | 125,250 ops | 500 ops | 250× |
| 1000 | 500,500 ops | 1000 ops | 500× |
| 4096 | 8.3M ops | 4096 ops | **2048×** |

The longer your generation, the more KV cache matters!

### Memory Cost

KV cache stores `2 * num_layers * num_heads * seq_len * head_dim` floats:

For GPT-2 small generating 1000 tokens:
```
2 × 12 × 12 × 1000 × 64 = 18,432,000 floats
= 18.4M × 4 bytes (float32) = 73.7 MB
= 18.4M × 2 bytes (bfloat16) = 36.8 MB
```

Manageable for small models, but for GPT-3 (96 layers, 96 heads) × 4096 tokens:
```
2 × 96 × 96 × 4096 × 128 = 9.66 BILLION floats = 38.6 GB!
```

This is why long-context inference needs enormous GPU memory or memory-efficient KV cache techniques.

### KV Cache Implementation Concept

```python
# Simplified KV cache (conceptual - not full implementation)
class GPTWithKVCache(GPT):
    def generate_with_cache(self, input_ids, max_new_tokens, ...):
        # Prefill: process prompt, store K,V
        kv_cache = []  # List of (K, V) tuples per layer
        
        # First forward: process full prompt
        logits, new_kv = self.forward_with_cache(input_ids, kv_cache=None)
        kv_cache = new_kv  # Store for reuse
        
        for _ in range(max_new_tokens):
            next_token = sample(logits[:, -1, :])
            # Forward only the NEW token, reuse cached K,V
            logits, new_kv = self.forward_with_cache(
                next_token.unsqueeze(1),  # Only 1 new token!
                kv_cache=kv_cache
            )
            kv_cache = new_kv  # Append new K,V to cache
            input_ids = torch.cat([input_ids, next_token], dim=1)
```

## Sampling Strategies — How to Pick the Next Token

### Greedy Sampling (temperature = 0)

Always pick the single MOST likely token.

```
Prompt: "The cat sat on the"
Logits: [the: 9.2,  a: 8.1,  my: 3.2,  their: 1.1, ...]
                                ↑ always pick this
Result: "The cat sat on the mat. The cat sat on the mat. The cat..."  ← repeats!
```

**Problem:** Deterministic → same prompt always gives same output. Tends to repeat.

### Temperature Sampling

Scale logits before softmax. Lower temperature = sharper distribution (more confident picks). Higher = flatter (more random).

```python
# Temperature effect on a toy distribution:
logits = [2.0, 1.0, 0.5, 0.1]  # 4 possible tokens

# T = 0.5 (cold — confident):
scaled = [2.0/0.5, 1.0/0.5, 0.5/0.5, 0.1/0.5] = [4.0, 2.0, 1.0, 0.2]
probs  = softmax([4.0, 2.0, 1.0, 0.2]) = [0.86, 0.12, 0.02, 0.00]
# Token 0 has 86% chance — very confident!

# T = 1.0 (standard):
probs = softmax([2.0, 1.0, 0.5, 0.1]) = [0.56, 0.21, 0.13, 0.10]
# Token 0 = 56% — balanced distribution

# T = 2.0 (hot — creative):
scaled = [2.0/2.0, 1.0/2.0, 0.5/2.0, 0.1/2.0] = [1.0, 0.5, 0.25, 0.05]
probs  = softmax([1.0, 0.5, 0.25, 0.05]) = [0.36, 0.22, 0.22, 0.20]
# Flatter — token 0 only 36%, tokens 2 and 3 are competitive
```

**Same prompt, different temperatures:**

```
T=0.2 (focused):  "The capital of France is Paris, which is located in the Île-de-France region."
T=0.8 (balanced): "The capital of France is Paris, a city known for its art, cuisine, and the Eiffel Tower."
T=1.5 (creative): "The capital of France is Paris, where baguettes dream of becoming croissants under moonlight."
```

### Top-K Sampling

Only consider the K most likely tokens. Everything else → probability 0.

```
K=50: Only top 50 tokens. Good default — filters obvious nonsense.
K=10: Aggressive filter. Tends to be repetitive but never nonsensical.
K=1:  Same as greedy (always pick #1).
```

### Top-P (Nucleus) Sampling

Only consider the SMALLEST set of tokens whose cumulative probability exceeds P.

```
Tokens sorted by probability: [0.45, 0.22, 0.13, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01]

Top-P = 0.9:
  Cumulative: 0.45+0.22+0.13+0.08+0.05 = 0.93 > 0.9
  Keep first 5 tokens. Drop the rest.

Top-P = 0.5:
  Cumulative: 0.45+0.22 = 0.67 > 0.5
  Keep first 2 tokens.
```

**Why Top-P over Top-K?** Top-P adapts to the model's confidence:
- If model is very sure: keeps few tokens (sharp distribution)
- If model is uncertain: keeps many tokens (flat distribution)

Top-K always keeps exactly K tokens regardless of confidence.

### Beam Search

Instead of picking one token at a time, maintain multiple "beams" (candidate sequences):

```
Beam width = 3:

Step 1: "The" → 3 best next tokens: ["cat"(0.3), "dog"(0.2), "man"(0.1)]
Step 2: "The cat" → 3 best continuations: ["sat"(0.4), "is"(0.2), "was"(0.15)]
        "The dog" → 3 best: ["ran"(0.35), "is"(0.2), "barked"(0.1)]
        "The man" → 3 best: ["walked"(0.3), "said"(0.25), "is"(0.1)]
        Pick top 3 overall sequences:
        "The cat sat" (0.3×0.4=0.12), "The dog ran" (0.2×0.35=0.07), ...
```

**Beam search**: Higher quality output, but deterministic (same output every time) and slower. Often used for translation, not creative writing.

### Repetition Penalty

During generation, penalize tokens that have already appeared:

```
For each candidate token:
  penalty = 1.0 if token NOT in recent history
  penalty = 0.5 if token appeared once recently
  penalty = 0.2 if token appeared multiple times

logits = logits * penalty
```

This prevents the model from looping: `"I like cats. I like cats. I like cats..."`

### Comparison Table

| Strategy | Randomness | Quality | Speed | Use Case |
|---|---|---|---|---|
| **Greedy** (T=0) | None | Good for facts | Fast | Translation, code |
| **Temperature** | Controllable | Varies | Fast | Creative writing |
| **Top-K=50** | Low-moderate | Good default | Fast | General generation |
| **Top-P=0.9** | Adaptive | Good default | Fast | Chat, conversation |
| **Beam Search** | None | Best | 3-5× slower | Translation, summarization |
| **T=0.7 + Top-P=0.9** | Moderate | Great | Fast | 🏆 Recommended default |

## Full Inference Code

### Loading a Checkpoint

```python
import torch


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """
    WHAT: Load a trained GPT model from a saved checkpoint file.
    WHY: After training, we save model state. To generate text,
         we need to load this state back — weights, config, everything.
    """
    # WHAT: Load the checkpoint dictionary from disk
    # WHY: map_location ensures it loads to the right device (CPU/GPU)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # WHAT: Recreate the model from the saved config
    model = GPT(checkpoint["config"])

    # WHAT: Load the trained weights into the model
    # WHY: state_dict contains every parameter value learned during training
    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)  # Move to GPU
    model.eval()              # Disable dropout for inference

    print(f"Loaded model from step {checkpoint['step']}, "
          f"loss: {checkpoint['loss']:.4f}")
    return model
```

### Text Generation Wrapper

```python
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    device: torch.device = None,
):
    """
    WHAT: Generate text from a prompt using a trained GPT model.
    WHY: High-level interface — tokenize → generate → decode.

    Parameter Guide:
      temperature: 0.2 = factual, 0.8 = balanced, 1.5 = wild
      top_k:       50 = standard, 10 = conservative, 0 = disabled
      top_p:       0.9 = recommended, 0.5 = narrow, 1.0 = disabled
    """
    device = device or next(model.parameters()).device

    # WHAT: Convert prompt string to token IDs
    input_ids = torch.tensor(
        [tokenizer.encode(prompt)], dtype=torch.long, device=device
    )

    # WHAT: Run autoregressive generation
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    # WHAT: Convert generated token IDs back to string
    return tokenizer.decode(output_ids[0].tolist())
```

### Interactive Generation Example

```python
# Load the trained model
model = load_checkpoint("checkpoints/best_model.pt", device)

# Test different generation strategies
prompts = [
    "Once upon a time, in a land far away,",
    "The secret to happiness is",
    "If I could travel anywhere in the world, I would go to",
]

for prompt in prompts:
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")

    # Conservative — good for facts
    text = generate_text(
        model, tokenizer, prompt, temperature=0.3, top_k=20,
    )
    print(f"\nConservative (T=0.3, K=20):")
    print(f"  {text[:300]}")

    # Balanced — good default
    text = generate_text(
        model, tokenizer, prompt, temperature=0.8, top_k=50, top_p=0.9,
    )
    print(f"\nBalanced (T=0.8, K=50, P=0.9):")
    print(f"  {text[:300]}")

    # Creative — good for writing
    text = generate_text(
        model, tokenizer, prompt, temperature=1.3, top_k=100, top_p=0.95,
    )
    print(f"\nCreative (T=1.3, K=100, P=0.95):")
    print(f"  {text[:300]}")
```

---

**Previous:** [Chapter 8 — Training](08_training.md)
**Next:** [Chapter 10 — Full Script](10_full_script.md)

# Chapter 4 — Positional Encoding: Teaching Order

## The 5-Year-Old Analogy

Consider two sentences:
- "The **dog** bit the **man**."  — scary
- "The **man** bit the **dog**."  — weird

Same words, different order -> **completely different meaning**.

But the Transformer reads all words **at once** (not one by one like humans do). It has **no idea** which word comes first! So we must **stamp each word with its position** before feeding it to the model.

## The Three Generations of Position Encoding

| Method | How It Works | Pros | Cons | Used By |
|---|---|---|---|---|
| **Learned** | Each position gets its own learned vector | Simple, flexible | Can't handle sequences longer than training | GPT-2, BERT |
| **Sinusoidal** | Fixed sine/cosine waves by position | Works for any length | Weaker at relative positions | Original Transformer |
| **RoPE** | Rotates Q,K vectors by position angle | Perfect relative positions, any length | Slightly more complex | LLaMA, Mistral, Qwen, Gemma |
| **ALiBi** | Adds a bias to attention scores based on distance | No learned params, very fast | Less expressive | BLOOM, MPT |

## Modern Approach: Rotary Position Embeddings (RoPE)

Instead of **adding** position numbers to embeddings, RoPE **rotates** the query and key vectors by an angle that depends on position.

### The Math Intuition

In 2D, rotating a vector `(x, y)` by angle `θ` gives:
```
x' = x*cos(θ) - y*sin(θ)
y' = x*sin(θ) + y*cos(θ)
```

RoPE does this for EVERY pair of dimensions in the query and key vectors. The rotation angle for position `p` and dimension pair `2i, 2i+1` is:

```
θ(p, i) = p / (10000^(2i/d_model))
```

**Key insight:** The angle depends on `p` (position) and `i` (dimension pair index). Lower dimension pairs rotate FAST (capturing local word relationships). Higher pairs rotate SLOW (capturing long-range relationships).

### Numerical Worked Example

Let's trace RoPE with a tiny model: `d_model=4`, processing position `p=1`:

**Step 1: Compute frequencies for each dimension pair**

```
Pair 0 (dims 0,1): freq = 1 / 10000^(0/4)   = 1 / 1       = 1.000
Pair 1 (dims 2,3): freq = 1 / 10000^(2/4)   = 1 / 10000^0.5 = 1 / 100 = 0.010
```

**Step 2: Compute rotation angle for position p=1**

```
Pair 0 angle: θ₀ = p * freq₀ = 1 * 1.000 = 1.000 radian (≈ 57.3°)
Pair 1 angle: θ₁ = p * freq₁ = 1 * 0.010 = 0.010 radian (≈ 0.57°)
```

**Step 3: Apply rotation to a query vector at position 1**

```
Before RoPE: q₁ = [0.8, 0.3, -0.5, 0.2]

Rotate pair 0 (dims 0,1) by 57.3°:
  dim0' = 0.8*cos(1.0) - 0.3*sin(1.0) = 0.8*0.540 - 0.3*0.842 = 0.432 - 0.253 = 0.179
  dim1' = 0.8*sin(1.0) + 0.3*cos(1.0) = 0.8*0.842 + 0.3*0.540 = 0.674 + 0.162 = 0.836

Rotate pair 1 (dims 2,3) by 0.57°:
  dim2' = -0.5*cos(0.01) - 0.2*sin(0.01) = -0.5*1.000 - 0.2*0.010 = -0.500 - 0.002 = -0.502
  dim3' = -0.5*sin(0.01) + 0.2*cos(0.01) = -0.5*0.010 + 0.2*1.000 = -0.005 + 0.200 = 0.195

After RoPE: q₁' = [0.179, 0.836, -0.502, 0.195]
```

Now let's compute what happens at positions 1 and 3:

```
Position 1: θ₀ = 1.0 rad,  θ₁ = 0.01 rad
Position 3: θ₀ = 3.0 rad,  θ₁ = 0.03 rad

The dot product q₁ · k₃ will depend on the DIFFERENCE:
  Δθ₀ = 3.0 - 1.0 = 2.0 rad
  Δθ₁ = 0.03 - 0.01 = 0.02 rad
  
This difference depends ONLY on (3-1)=2, the relative distance!
Absolute positions don't matter — only how far apart they are.
```

This is why RoPE is brilliant: the attention score between position `i` and `j` depends **only** on their relative distance `(j-i)`, not their absolute positions.

### Why theta=10000?

The base frequency `theta = 10000` controls the "spread" of frequencies:

```
Low theta (e.g., 100):
  - All dimension pairs rotate similarly
  - Model is more "position-agnostic" — better for long contexts
  - But loses fine-grained position resolution

High theta (e.g., 100000):
  - Very different rotation speeds across dimensions
  - Better at distinguishing nearby positions
  - But struggles with very long contexts

10000 was found empirically to balance these tradeoffs.
```

### Extending Context Beyond Training Length

What if we trained on 2048 tokens but want to use 4096 at inference?

**The problem:** RoPE was precomputed for positions 0-2047. Position 3000 was never seen.

**Solutions:**
| Method | How It Works | Quality |
|---|---|---|
| **Linear interpolation** | Position / scale (e.g., p/2 for 2x length) | OK, loses resolution |
| **NTK-aware scaling** | Scale theta differently per frequency | Good |
| **YaRN** | NTK + temperature scaling | Best (used in production) |
| **Retrain** | Just train on longer sequences | Perfect but expensive |

For our small training run, this doesn't matter — but know that it's a hot research area for production models.

## RoPE Code — Annotated

```python
import torch
import torch.nn as nn
import math


class RotaryPositionalEmbedding(nn.Module):
    """
    WHAT: Rotary Position Embeddings (RoPE).
    WHY: Instead of ADDING position info to embeddings,
         we ROTATE Q and K vectors by position-dependent angles.
         The dot product q_i · k_j then depends ONLY on (j-i),
         which is exactly what attention should care about.

         Paper: "RoFormer" (Su et al., 2021)
         Used in: LLaMA 1/2/3, Mistral, Mixtral, Qwen 1/2, Gemma

         How it works at a glance:
         1. For each pair of dimensions (0,1), (2,3), (4,5), ...
         2. Rotate by angle = position * frequency
         3. Lower dims rotate fast (local position)
            Higher dims rotate slow (global position)
         4. The dot product naturally depends on relative distance
    """

    def __init__(self, d_model: int, max_seq_len: int = 2048, theta: float = 10000.0):
        """
        WHAT: Precompute rotation frequencies for fast lookup.

        Args:
            d_model:     Head dimension (e.g., 64 for GPT-2). Must be even.
            max_seq_len: Precompute angles for positions 0..max_seq_len-1.
            theta:       Base frequency. 10000 is standard. Controls the
                         spread between fast and slow rotation frequencies.
        """
        super().__init__()

        # WHAT: Verify d_model is even (must have pairs to rotate)
        assert d_model % 2 == 0, (
            f"d_model ({d_model}) must be even for RoPE. "
            f"Each pair of dimensions needs a partner to rotate with."
        )

        # WHAT: Create dimension indices: [0, 2, 4, ..., d_model-2]
        # WHY: Each pair (2i, 2i+1) gets the same rotation frequency.
        #      We only need half the indices because pairs share.
        dim_indices = torch.arange(0, d_model, 2).float()

        # WHAT: Compute rotation frequencies
        # WHY: theta_i = 1 / (theta ^ (2i / d_model))
        #
        #      i=0:  1 / 10000^(0/64)      = 1.0      → fast rotation (local)
        #      i=30: 1 / 10000^(60/64)     ≈ 0.0001   → slow rotation (global)
        #
        #      This multi-scale approach means some dimensions
        #      capture local word order while others capture
        #      long-range position relationships.
        inv_freq = 1.0 / (theta ** (dim_indices / d_model))

        # WHAT: Precompute angles for all positions
        # WHY: Computing cos/sin during training is expensive.
        #      Precomputing them once and caching is 100x faster.
        positions = torch.arange(max_seq_len).float()     # [0, 1, 2, ..., 2047]

        # WHAT: Outer product: each position x each frequency
        #       freqs[p, i] = p * inv_freq[i] = angle for position p, dim pair i
        #       Shape: [max_seq_len, d_model/2]
        freqs = torch.outer(positions, inv_freq)

        # WHAT: Duplicate to full dimension
        # WHY: Each dim pair (2i, 2i+1) gets the same angle,
        #      so we copy each angle: [θ0, θ1, θ2, ...] -> [θ0, θ0, θ1, θ1, ...]
        emb = torch.cat([freqs, freqs], dim=-1)         # [max_seq_len, d_model]

        # WHAT: Cache cos and sin for all positions
        # WHY: register_buffer means these move with model.to(device)
        #      and are saved with model.state_dict(), but are NOT
        #      trainable parameters (no gradients needed).
        self.register_buffer("cos_cached", emb.cos())   # cos of each angle
        self.register_buffer("sin_cached", emb.sin())   # sin of each angle

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        WHAT: Prepare a vector for the rotation formula.
        WHY:  The rotation formula is: x' = x*cos + rotate_half(x)*sin
        
              For vector [x0, x1, x2, x3, x4, x5]:
              rotate_half returns [-x1, x0, -x3, x2, -x5, x4]
              
              Why this works: For pair (x0, x1) rotated by angle θ:
                x0' = x0*cos(θ) - x1*sin(θ)   ← matches: x0*cos + (-x1)*sin
                x1' = x0*sin(θ) + x1*cos(θ)   ← matches: x1*cos + (x0)*sin
              
              So executing (x*cos + rotate_half(x)*sin) performs rotation
              on every dimension pair simultaneously — no loop needed!
        """
        x1 = x[..., : x.shape[-1] // 2]   # First half:  [x0, x2, x4, ...]
        x2 = x[..., x.shape[-1] // 2 :]   # Second half: [x1, x3, x5, ...]
        return torch.cat([-x2, x1], dim=-1)  # [-x1, x0, -x3, x2, -x5, x4, ...]

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        WHAT: Apply RoPE to queries or keys.

        Input:  [batch, seq_len, num_heads, head_dim]
                x can be either Q or K (NOT V — values don't need position)
        Output: Same shape, rotated by position-dependent angles

        WHY applied only to Q and K:
        The attention score = Q_i · K_j controls WHICH values to attend to.
        We want this score to depend on relative position.
        The VALUE vectors carry content — position is irrelevant for the
        content itself. Position only matters for deciding which tokens
        to pay attention TO.
        """
        # WHAT: Extract cos and sin for current sequence length
        # WHY: If seq_len=512 but max_seq_len=2048, we only need
        #      the first 512 rows of the cached cos/sin tables.
        cos = self.cos_cached[:seq_len]   # [seq_len, head_dim]
        sin = self.sin_cached[:seq_len]   # [seq_len, head_dim]

        # WHAT: Add batch and head dimensions for broadcasting
        # WHY: cos/sin are [seq_len, head_dim]. We need them to
        #      multiply with x [batch, heads, seq_len, head_dim].
        #      unsqueeze(0).unsqueeze(0) adds dims at positions 0 and 1:
        #      [seq_len, head_dim] -> [1, 1, seq_len, head_dim]
        #      Now they broadcast correctly over batch and heads.
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # WHAT: Execute rotation: x_rotated = x*cos(θ) + rotate_half(x)*sin(θ)
        # WHY: This is mathematically equivalent to applying a 2D rotation
        #      matrix to each pair of dimensions, but implemented in pure
        #      element-wise operations — much faster and parallelizable.
        return (x * cos) + (self.rotate_half(x) * sin)
```

---

**Previous:** [Chapter 3 — Embeddings](03_embeddings.md)
**Next:** [Chapter 5 — Attention](05_attention.md)

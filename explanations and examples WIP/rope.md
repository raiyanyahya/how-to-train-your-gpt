# RoPE — Rotary Position Embeddings

## What is it

RoPE is a way to tell a language model *where* each word sits in
a sentence. Without it the model sees all words at once and has
no idea which word came first. RoPE stamps every word with its
position by giving it a tiny rotation. Words at the start get a
small spin. Words later in the sentence get a bigger spin. The
model can look at how much two words are rotated and figure out
their distance.

## Where is it used

RoPE lives inside the attention layer. Specifically it is applied
to the query and key vectors right before the dot product that
decides how much two words should pay attention to each other.

```
Input tokens
  → Embedding (word meanings)
    → Attention (where RoPE happens)
      → Transformer Block output
```

## Why use it

Before RoPE people used other tricks to mark word positions. Some
added position numbers to the word vectors. Others let the model
learn position from scratch. Both worked but had limits. Learned
positions could not handle sentences longer than training. Added
position numbers did not capture the relative distance between
words well.

RoPE fixes both problems. It captures relative distance perfectly.
Word five and word seven are always two steps apart no matter if
they appear at the start or the middle of a long paragraph. And
RoPE can handle any sentence length even if the model was trained
on shorter ones. This is why LLaMA Mistral and Qwen all use RoPE.

## When was it invented

RoPE was published in 2021 by a team of researchers in a paper
called RoFormer. It took a few years to catch on but by 2023
every major open source language model had switched to RoPE.

## How it works in simple terms

Imagine a clock with only one hand. At position zero the hand
points straight up. At position one the hand rotates a little.
At position two it rotates a little more. Each position gets a
unique angle. The model stores these angles as cosine and sine
values so it never has to compute them during training.

Now every word has a secret pair of numbers. RoPE takes that pair
and rotates it by the angle for that position. After rotation two
words that are close together will have similar rotations. Two
words far apart will have very different rotations. When attention
looks at the dot product between a query and a key the result
depends on how far apart they are. Not on their absolute position.

## A tiny code example

```python
import torch
import math

# Set up RoPE for a tiny model with 4 dimensions
d_model = 4
max_seq_len = 16
theta = 10000.0

dim_indices = torch.arange(0, d_model, 2).float()
inv_freq = 1.0 / (theta ** (dim_indices / d_model))
positions = torch.arange(max_seq_len).float()
freqs = torch.outer(positions, inv_freq)
emb = freqs.repeat_interleave(2, dim=-1)

cos_cached = emb.cos()
sin_cached = emb.sin()

# Pretend we have a query vector for a word at position 0
q = torch.tensor([0.8, 0.3, -0.5, 0.2])

seq_len = 4
cos = cos_cached[:seq_len]
sin = sin_cached[:seq_len]

# Apply rotation for position 0
rotated = q * cos[0] + torch.tensor([-0.3, 0.8, -0.2, -0.5]) * sin[0]
print(f"Position 0: {rotated.tolist()}")

# Apply rotation for position 2
rotated = q * cos[2] + torch.tensor([-0.3, 0.8, -0.2, -0.5]) * sin[2]
print(f"Position 2: {rotated.tolist()}")

print()
print("Same word at different positions gets different rotations.")
print("The model uses this difference to understand word order.")
```

## What you need to remember

RoPE rotates vectors. The rotation angle depends on position. The
dot product between two rotated vectors depends only on how far
apart they are. This is what attention should care about. Not
where the words are. But how far they are from each other.

RoPE is free. No learned parameters. No extra memory. No speed
penalty. It works for sequences of any length. Every modern
language model uses it.

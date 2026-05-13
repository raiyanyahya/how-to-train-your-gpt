# KV Cache: Making Text Generation Fast

## What is it

The KV cache stores the Key and Value vectors from all previous
tokens during text generation. When the model generates the next
token it reuses these stored vectors instead of recomputing them
from scratch. This makes generation hundreds of times faster.

Think of it like writing a long email. Without a KV cache you
would reread the entire email from the start every time you typed
a new word. With a KV cache you remember everything you already
wrote and only think about the new word. The difference in
effort is enormous.

## Where is it used

The KV cache lives inside the attention layer. Every attention
head in every transformer block has its own cache. For a twelve
block model with twelve heads there are one hundred and forty
four separate caches. Each stores the Keys and Values for every
token generated so far.

```
Text generation without cache:
  For each new token:
    Run the ENTIRE sequence through all layers
    This recomputes K and V for every past token
    Time grows quadratically with sequence length

Text generation with cache:
  For each new token:
    Only compute K and V for the new token
    Append to cache
    Reuse cached K and V for all past tokens
    Time grows linearly with sequence length
```

## Why we need it

The attention formula is Q times K transpose. The Q matrix has
one row per token. The K matrix has one row per token. If we have
five hundred tokens we multiply a five hundred by sixty four
matrix by a sixty four by five hundred matrix. That is already
some work.

Without a cache we would do this multiplication from scratch for
every new token. When the sequence is one token long we do one
comparison. When it is two tokens long we do two comparisons for
token zero and two for token one. When it is five hundred tokens
long we do five hundred comparisons for each of the five hundred
tokens. The total work grows with the square of the sequence
length. This is painfully slow.

With a cache we only compute the comparisons involving the new
token. Token five hundred compares itself against all five hundred
previous tokens. That is five hundred new comparisons. Not five
hundred squared. The cached Keys save us from recomputing the old
comparisons that have not changed.

### The speed numbers

```
Sequence length 100:
  Without cache: 100² = 10,000 comparisons per generation step
  With cache:    100 comparisons per generation step
  Speedup: 100×

Sequence length 1000:
  Without cache: 1,000² = 1,000,000 comparisons
  With cache:    1,000 comparisons
  Speedup: 1,000×
```

For a long conversation with thousands of tokens the KV cache
makes the difference between waiting seconds and waiting minutes
for each new word.

## When was it invented

The KV cache was described in the original transformer paper in
2017. It was not an optimization added later. It was part of the
design from the beginning. The authors knew that autoregressive
generation would be painfully slow without it. Every transformer
implementation since 2017 uses some form of KV cache.

## How it works step by step

### Step 1: the first token

We have a prompt that is five tokens long. The model processes
all five tokens in parallel during a prefill step.

```
Prefill:
  Tokens: [The, cat, sat, on, the]
  K cache for head 0: store K for all 5 tokens  (5 × 64 matrix)
  V cache for head 0: store V for all 5 tokens  (5 × 64 matrix)
  ... repeat for all 12 heads ...
```

This initial step is expensive but it only happens once.

### Step 2: generating the next token

The model predicts that the next token is *mat*. We append *mat*
to the sequence. Now we have six tokens.

```
Generation step 1:
  New token: [mat]
  Compute K for "mat" only: (1 × 64 matrix)
  Compute V for "mat" only: (1 × 64 matrix)
  Full K cache: 5 old rows + 1 new row = 6 × 64 matrix
  Full V cache: 5 old rows + 1 new row = 6 × 64 matrix
  Compute Q for "mat" only: (1 × 64 matrix)
  Attention scores: Q_mat × K_full^T = 1 × 6 comparisons
  Only the new token's Q was computed. Old Q values are not needed.
```

We computed only one new Key one new Value and one new Query. The
attention scores for *mat* are computed against all six tokens
because *mat* needs to attend to everything that came before. But
the attention scores for *The* and *cat* and *sat* are not
recomputed. They do not change. Why would they. Their context has
not changed. Only the new token has new context to process.

### Step 3: memory growth

For each new token we add one row to every K cache and every V
cache. The cache grows linearly with the sequence length.

```
Memory for a GPT-2 Small model generating 1000 tokens:

K cache: 12 layers × 12 heads × 1000 tokens × 64 dims × 2 bytes (bfloat16)
       = 18,432,000 bytes
       = 17.6 MB

V cache: same as K cache = 17.6 MB

Total KV cache: 35.2 MB
```

Thirty five megabytes is nothing for a modern GPU. But remember
this is GPT-2 Small with only twelve layers and 768 dimensions.

```
Memory for a GPT-3 Large model generating 1000 tokens:

K cache: 96 layers × 96 heads × 1000 tokens × 128 dims × 2 bytes
       = 2,359,296,000 bytes
       = 2.2 GB

V cache: same as K cache = 2.2 GB

Total KV cache: 4.4 GB
```

Now the cache is a significant portion of GPU memory. For long
conversations spanning thousands of tokens the KV cache can
become the dominant memory consumer. This is one reason why
running large models at long context lengths requires enormous
amounts of VRAM.

## A simplified code sketch

This is not production code but it shows the idea.

```python
class AttentionWithCache:
    def __init__(self):
        self.k_cache = None  # Will hold accumulated K values
        self.v_cache = None  # Will hold accumulated V values

    def forward(self, x, use_cache=False):
        q, k, v = self.qkv_proj(x)

        if use_cache and self.k_cache is not None:
            # Append new K and V to cache
            k = torch.cat([self.k_cache, k], dim=2)
            v = torch.cat([self.v_cache, v], dim=2)

        # Update cache for next step
        self.k_cache = k
        self.v_cache = v

        # Normal attention computation
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = F.softmax(scores, dim=-1)
        return weights @ v

    def reset_cache(self):
        self.k_cache = None
        self.v_cache = None
```

The key idea is on lines eleven and twelve. Instead of throwing
away the old K and V we append the new ones. The cache grows over
time. At the start of a new conversation we reset the cache to
empty.

## The tradeoff

KV cache trades memory for speed. We accept that generation will
use more GPU memory in exchange for making each step much faster.
For most real world applications this is a good trade. Memory is
relatively cheap compared to the user's patience waiting for
each word to appear.

However for very long sequences the memory cost becomes
prohibitive. Researchers have developed techniques like grouped
query attention and multi query attention that reduce the number
of K and V heads. Fewer heads means a smaller cache. These
techniques are standard in modern models like LLaMA 2 and Mistral.

## What you need to remember

The KV cache stores previously computed Keys and Values so they
do not need to be recomputed for each new token. This changes the
complexity of text generation from quadratic to linear in the
sequence length. For a thousand token sequence this is a thousand
times speedup.

The cost is memory. The cache grows with the sequence length and
the model size. For small models the memory cost is negligible.
For large models at long context lengths the cache can consume
tens of gigabytes.

Without a KV cache text generation would be slow enough that
chatbots would be unusable. Every response would take minutes
instead of seconds. The cache is a necessary optimization for
any real world deployment.

# Temperature Top-K and Top-P: Controlling Text Generation

## What are they

Temperature top-k and top-p are three knobs you can turn to
control how a language model generates text. They change how the
model picks the next word. Without them the model would always
pick the single most likely word. The output would be boring and
repetitive. With these knobs you can make the output more focused
or more creative. You can make it safe or adventurous.

Think of the model as a chef choosing ingredients. Without any
controls the chef always picks the most common ingredient for
every dish. Every meal is chicken. Every dessert is vanilla.
Temperature tells the chef to sometimes try the less common
ingredients. Top-k limits the pantry to only the most sensible
options. Top-p lets the chef grab ingredients until they have
enough variety and then stops.

## Where are they used

These knobs are applied right after the model produces its raw
scores and right before it picks the next token.

```
Model output (logits for 50257 tokens)
  → Divide by temperature
    → Keep only top-k tokens
      → Keep tokens until cumulative probability exceeds top-p
        → Softmax to probabilities
          → Pick one token randomly
```

They are used during text generation only. Not during training.
During training the model always sees the correct answer. During
generation there is no correct answer. The model must explore the
space of possible next words. These knobs control how it explores.

## Why we need them

Without any controls the model does one thing: it picks the token
with the highest probability. Always. Every time.

```
Prompt: "The cat sat on the"
Model prediction always: "mat"
Generated: "The cat sat on the mat. The cat sat on the mat. The cat sat on the mat..."
```

The output loops. It gets stuck in a cycle. This happens because
the highest probability path through language is often a loop.
Once the model says *The cat sat on the mat* the next most
likely continuation is *The cat sat on the mat* again. The
probabilities form a trap.

The knobs break this trap by introducing controlled randomness.
Instead of always picking the top token the model sometimes
picks the second best or the third best. The output stays
sensible but avoids repetition.

## Temperature: how adventurous should the model be

Temperature is a simple division. Take the model's raw scores and
divide every one by the temperature.

```
Low temperature (0.3):
  Scores are amplified. The top token gets even more probability.
  The model is confident and predictable.

  Prompt: "The capital of France is"
  Output: "Paris, which is located in the Île-de-France region."

High temperature (1.5):
  Scores are flattened. All tokens get more equal probability.
  The model is creative and unpredictable.

  Prompt: "The capital of France is"
  Output: "Paris, where baguettes dream of becoming croissants."
```

The math is simple. Here is a tiny example with four candidate
tokens.

```python
logits = [4.0, 2.0, 1.0, 0.5]

# Temperature 0.5 (focused)
scaled = [4.0/0.5, 2.0/0.5, 1.0/0.5, 0.5/0.5]
       = [8.0, 4.0, 2.0, 1.0]
probs  = softmax([8.0, 4.0, 2.0, 1.0])
       = [0.97, 0.02, 0.01, 0.00]
# Token 0 has 97% chance. Very confident.

# Temperature 2.0 (creative)
scaled = [4.0/2.0, 2.0/2.0, 1.0/2.0, 0.5/2.0]
       = [2.0, 1.0, 0.5, 0.25]
probs  = softmax([2.0, 1.0, 0.5, 0.25])
       = [0.48, 0.18, 0.18, 0.16]
# Token 0 has only 48% chance. Much more spread out.
```

Temperature of 0 means always pick the most likely token. This
is called greedy decoding. Temperature of 1 means use the natural
probabilities with no modification. Temperature above 1 makes the
model more random. Temperature below 1 makes the model more
focused.

## Top-K: only consider the best options

Temperature spreads the probabilities but even a tiny probability
is still a chance for complete nonsense. Top-k puts a hard limit.
Only the k most likely tokens are considered. Everything else gets
probability zero.

```python
# All 50257 tokens have some probability after temperature
# With top-k=50 we keep only the 50 most likely ones

v, _ = torch.topk(logits, 50)
logits[logits < v[:, -1:]] = float('-inf')
# Now only 50 tokens have non zero probability
```

The magic number is often 50. This eliminates truly nonsensical
completions while keeping enough variety for interesting output.
A smaller k like 10 makes the output more focused. A larger k
like 200 makes it more varied.

## Top-P: dynamic cutoff based on confidence

Top-k always keeps exactly k tokens. But the model's confidence
varies from word to word. Sometimes the model is very sure and
only a few tokens are reasonable. Sometimes the model is unsure
and many tokens are plausible. Top-p adapts to the situation.

Top-p also called nucleus sampling keeps the smallest set of
tokens whose cumulative probability exceeds p.

```
Tokens sorted by probability:
[0.45, 0.22, 0.13, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01]

Top-p = 0.9:
  Cumulative: 0.45 > keep
  Cumulative: 0.45 + 0.22 = 0.67 > keep
  Cumulative: 0.45 + 0.22 + 0.13 = 0.80 > keep
  Cumulative: 0.45 + 0.22 + 0.13 + 0.08 = 0.88 > keep
  Cumulative: 0.45 + 0.22 + 0.13 + 0.08 + 0.05 = 0.93 > stop!
  Keep first 5 tokens. Drop the rest.

Top-p = 0.5:
  Cumulative: 0.45 > keep
  Cumulative: 0.45 + 0.22 = 0.67 > stop!
  Keep first 2 tokens.
```

When the model is very confident the top few tokens might already
have total probability 0.9. Top-p keeps just those few. When the
model is uncertain it takes many more tokens to reach 0.9. Top-p
keeps more options. This adaptive behavior is why top-p is often
preferred over top-k.

## The recommended combination

Most production systems use all three together.

```python
logits = logits / temperature          # Step 1: control randomness
logits = filter_top_k(logits, k=50)   # Step 2: eliminate nonsense
logits = filter_top_p(logits, p=0.9)  # Step 3: adapt to confidence
probs = softmax(logits)                # Step 4: convert to probabilities
next_token = sample(probs)            # Step 5: pick one
```

A common default that works well for general conversation is
temperature 0.7 with top-p 0.9 and top-k 50. For factual
responses lower the temperature. For creative writing raise it.

## A tiny code example

```python
import torch
import torch.nn.functional as F

def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None):
    # Apply temperature
    logits = logits / temperature

    # Top-k filtering
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, -1:]] = float('-inf')

    # Top-p filtering
    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_mask = cumulative_probs > top_p
        sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
        sorted_mask[:, 0] = False

        mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
        logits[mask] = float('-inf')

    # Sample
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

# Test
logits = torch.tensor([[4.0, 2.0, 1.5, 0.8, 0.3, 0.1, 0.05, 0.02]])

print("Same prompt different temperatures:")
for temp in [0.3, 0.7, 1.5]:
    sampled = []
    for _ in range(5):
        t = sample_next_token(logits, temperature=temp, top_k=5)
        sampled.append(t.item())
    print(f"  T={temp}: samples={sampled}")
```

## What you need to remember

Temperature top-k and top-p control how the model picks the next
token during text generation. Temperature adjusts the randomness
of the whole distribution. Top-k keeps only the best k options.
Top-p adapts the number of options based on the model's confidence.

Without these controls text generation would be deterministic and
repetitive. The model would loop on the same phrases forever. With
these controls generation becomes varied and natural. Different
temperatures give different writing styles from the same model.
This is why the same language model can write both technical
documentation and poetry. The model is the same. The knobs are
different.

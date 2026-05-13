# Causal Masking: No Peeking at the Future

## What is it

Causal masking is a rule that prevents every word in a sentence
from seeing the words that come after it. Given a sentence like
*The cat sat on the mat* the word *cat* can see *The* but not
*sat*. The word *sat* can see *The* and *cat* but not *on*. Each
word lives in the dark about what comes next.

Think of it like reading a book one page at a time. You have read
pages one through five. Page six is face down on the table. You
are not allowed to peek. You must guess what happens on page six
using only what you have already read.

This rule seems restrictive. Why not let the model see everything.
Because the model's entire job is to predict the next word. If
the model could see the next word it would not need to predict
it. It would just copy. That is not learning. That is cheating.

## Where is it used

The causal mask lives inside the attention layer. Right after the
attention scores are computed and right before the softmax turns
them into percentages.

```
Attention scores → Apply causal mask → Softmax → Attention weights
```

Every attention layer in every transformer block applies the
causal mask. For a twelve block model the mask is applied twelve
times for every sentence. Always the same mask. Always the same
rule. No peeking at the future.

## Why we need it

During training the model sees complete sentences. The input is
*The cat sat on the mat* all at once. Without the causal mask the
word *cat* could look at *mat* and say *ah the sentence ends with
mat so cat must be followed by sat*. The model would learn a
trivial mapping from full sentences to themselves. It would never
learn to predict. It would only learn to copy.

With the causal mask the model is forced to earn its predictions.
At position two it sees *The* and *cat* and must guess *sat*. At
position three it sees *The cat* and *sat* and must guess *on*.
The model cannot look ahead for hints. Every prediction is made
using only the information available at that point in the
sentence. This is exactly how text generation works in the real
world. You only know what came before. You never know what comes
next.

## When was it invented

The causal mask was introduced alongside the transformer itself
in the 2017 paper Attention Is All You Need. It was not an
afterthought. It was a design requirement. The authors knew that
language models must be trained autoregressively meaning one word
at a time from left to right. The causal mask enforces this
constraint during training so the model behaves correctly during
generation.

## How it works step by step

### The attention matrix without a mask

For a three word sentence *I love dogs* the model computes an
attention score between every pair of words. Every word can see
every other word.

```
Attention scores (no mask):

           I       love    dogs
I         0.42    0.15    0.08     ← I can see love and dogs
love      0.33    0.51    0.22     ← love can see I and dogs
dogs      0.19    0.28    0.44     ← dogs can see I and love
```

This is a fully connected matrix. Every pair of words has a score.
The model can use information from anywhere in the sentence.

### The attention matrix with a causal mask

After applying the mask future positions are set to negative
infinity. After softmax negative infinity becomes exactly zero.
Those connections are severed.

```
Attention scores (with mask):

           I       love    dogs
I         0.42    -inf    -inf      ← I can only see itself
love      0.33    0.51    -inf      ← love can see I and itself
dogs      0.19    0.28    0.44      ← dogs can see all three
```

After softmax these scores become weights:

```
Attention weights (after softmax):

           I       love    dogs
I         1.00    0.00    0.00      ← 100% on itself
love      0.46    0.54    0.00      ← split between I and itself
dogs      0.27    0.30    0.43      ← split among all three
```

The upper right triangle is all zeros. This is the characteristic
pattern of a causal mask. It is a lower triangular matrix.

### The rule in one line

```
Token at position p can only attend to tokens at positions
0 through p. Token at position p cannot attend to any token
at position p+1 or beyond.
```

## How we implement it

Creating a causal mask is surprisingly simple. PyTorch has a
function called tril that returns the lower triangle of a matrix.

```python
import torch

seq_len = 4
mask = torch.tril(torch.ones(seq_len, seq_len))

print(mask)

# Output:
# tensor([[1., 0., 0., 0.],
#         [1., 1., 0., 0.],
#         [1., 1., 1., 0.],
#         [1., 1., 1., 1.]])
```

That is the entire mask. Four lines of ones and zeros. Ones mean
visible. Zeros mean hidden.

In the actual attention code we use this mask like so:

```python
def create_causal_mask(seq_len, device):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.view(1, 1, seq_len, seq_len)

# Inside attention forward:
if mask is not None:
    attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
```

The `masked_fill` operation puts negative infinity wherever the
mask has a zero. After softmax `e^(-inf)` is zero and those
positions contribute nothing to the final output.

## What happens during generation

During training the causal mask prevents cheating. During text
generation the causal mask is still there but it does less work.

When generating text we start with a prompt like *Once upon a*.
The model processes these three tokens using the causal mask.
Token two can see token zero and token one. Token one can see
only token zero. Normal.

Then the model predicts the next token. Let us say it predicts
*time*. We append *time* to the sequence making it *Once upon a
time*. Now we run the model again on these four tokens. The
causal mask still applies. The new token *time* can see *Once*
and *upon* and *a* but it cannot see the next token because the
next token does not exist yet.

The causal mask is built into the architecture. It is not a
training only feature. It is a fundamental constraint that makes
autoregressive language models possible. Without it we could
never generate text token by token because the model would keep
trying to peek at words that have not been written yet.

## What happens without the causal mask

If we removed the causal mask during training the model would
learn to cheat. Its training loss would be extremely low because
it can always see the answer. But during generation when future
tokens do not exist yet the model would be completely lost. It
would not know how to predict the unknown. Its output would be
gibberish.

This is a common failure mode for beginners who build their first
transformer. The training loss looks fantastic. The generated
text is nonsense. The causal mask was omitted.

## What you need to remember

The causal mask makes attention a one way street. Words can look
back but never forward. This forces the model to predict each
token using only the tokens that came before it. This is exactly
how text generation works. You write one word at a time. You
never know what comes next until you write it.

The mask is a lower triangular matrix of ones and zeros. Zeros
become negative infinity in the attention scores. Negative
infinity becomes zero after softmax. The connections to future
words are severed. Only the past remains. This simple rule is
what separates a language model from a text copier.

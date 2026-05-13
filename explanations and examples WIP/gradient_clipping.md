# Gradient Clipping: Preventing Training Explosions

## What is it

Gradient clipping is a safety net. During training the model
calculates how to change each weight to reduce the loss. These
change instructions are called gradients. Sometimes a gradient
gets very large. One particular example in the training data
sends a shockwave through the network. The weights take a massive
jump and the model falls off the cliff into a region where the
loss is astronomical. Training is ruined.

Gradient clipping says: no gradient can be larger than a certain
limit. If the total magnitude of all gradients is too high we
shrink them proportionally until they fit under the limit. The
direction of the update stays the same. Only the step size is
limited. The model takes small safe steps instead of wild leaps.

## Where is it used

Gradient clipping is applied right before the optimizer updates
the weights. The gradients have already been computed. They are
about to be used to change the model. At this moment gradient
clipping checks them and reins in any that have grown too large.

```python
loss.backward()  # Compute gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()  # Apply clipped gradients
```

It is a single function call. One line of code that can save
hours of wasted training time.

## Why we need it

Language models are trained on text. Some text is unusual. A
sentence might contain a very rare word. The model has never
seen it before. The loss for that sentence is very high. The
gradients that flow backward from that loss are very large. If
the optimizer applies these large gradients the model's weights
jump to a completely different configuration. Everything the
model learned over the past thousand steps is wiped out by one
unusual sentence.

Without gradient clipping the training loss curve has spikes.
Long periods of steady improvement followed by sudden jumps where
the loss doubles or triples. After each spike the model must
recover. Sometimes it never recovers. The gradients were too
large and the weights went to a place from which there is no
return. The model produces only garbage from that point forward.

With gradient clipping the loss curve is smooth. The unusual
sentence still produces larger gradients than normal but those
gradients are clipped to a safe size. The model takes a slightly
larger than normal step in the right direction instead of a
catastrophic leap. Training continues uninterrupted.

## When was it invented

Gradient clipping has been used since the early days of recurrent
neural networks in the 1990s. RNNs were notorious for gradient
explosion because they processed sequences one step at a time
and the gradients multiplied at each step. The problem was
solved by simply capping gradients at a maximum value. The same
technique was carried forward to transformers even though
transformers do not have the same multiplicative problem. It
turns out that any deep network benefits from gradient clipping
as a safety measure.

## How it works

Gradient clipping by norm is the standard method. Instead of
clipping each gradient individually we measure the total size of
all gradients together and clip them as a group. This preserves
the relative sizes of different gradients. If one parameter
needs a large update and another needs a small update the ratio
between them is preserved even after clipping.

### Step 1: measure the total gradient magnitude

We compute the L2 norm of all gradients. This is the square root
of the sum of all squared gradients.

```python
total_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        total_norm += p.grad.norm(2).item() ** 2
total_norm = total_norm ** 0.5
```

If the model has a million parameters with an average gradient of
0.01 the total norm would be about 100. A total norm of 100 is
manageable. A total norm of 10000 is dangerous.

### Step 2: clip if needed

If the total norm exceeds the maximum allowed we shrink every
gradient by the same factor.

```python
max_norm = 1.0
if total_norm > max_norm:
    scale = max_norm / total_norm
    for p in model.parameters():
        if p.grad is not None:
            p.grad *= scale
```

If the total norm was 100 and the maximum is 1 we divide every
gradient by 100. The largest gradients become 0.01. The smallest
gradients become even smaller. The direction of the update is
unchanged. Only the step size changes.

### Why max_norm of 1.0

The value 1.0 is the standard for transformer training. It was
chosen empirically. Smaller values like 0.1 make training too
slow because the model can only take tiny steps. Larger values
like 10.0 provide little protection because most gradient norms
are already below 10. A value of 1.0 catches the dangerous spikes
without interfering with normal training steps.

## A tiny code example

```python
import torch
import torch.nn as nn

# Create a small model and some fake gradients
model = nn.Linear(10, 10)
loss = model(torch.randn(1, 10)).sum()
loss.backward()

# Check the gradient norm before clipping
total_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        total_norm += p.grad.norm(2).item() ** 2
total_norm = total_norm ** 0.5

print(f"Gradient norm before clipping: {total_norm:.4f}")

# Clip
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Check after
total_norm_after = 0.0
for p in model.parameters():
    if p.grad is not None:
        total_norm_after += p.grad.norm(2).item() ** 2
total_norm_after = total_norm_after ** 0.5

print(f"Gradient norm after clipping:  {total_norm_after:.4f}")
print(f"Clipped: {total_norm > 1.0}")
```

## What happens without it

Training language models without gradient clipping is playing
with fire. Most steps will be fine. The gradients will be small
and the model will learn. But eventually the model will
encounter a batch of text that produces large gradients. The
loss will spike. If the model is lucky it will recover. If it is
unlucky the spike will push the weights into a region where every
subsequent step also produces large gradients. The loss will
diverge to infinity and the training run will be lost.

Gradient clipping costs nothing in terms of model quality. It has
no downside. It is a pure safety measure that prevents a rare but
catastrophic failure mode. Every production training run uses it.

## What you need to remember

Gradient clipping limits how much the model's weights can change
in a single training step. If gradients are too large they are
scaled down proportionally to a maximum norm. The standard
maximum is 1.0 for transformer training.

One function call. Zero downside. Infinite protection against a
training killing failure mode. Use it always.

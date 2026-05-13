# Cosine Warmup: The Learning Rate Schedule

## What is it

The cosine warmup schedule controls how the learning rate changes
during training. It starts low and rises to a peak. Then it
gradually falls following a cosine curve. By the end of training
the learning rate is very small and the model settles into a fine
minimum.

Think of it like learning to ride a bicycle. At first you go very
slowly. You wobble. You get a feel for the balance. Once you have
some stability you push harder and go faster. As you approach your
destination you slow down again to make a precise stop. You do
not sprint from the start and slam the brakes at the end.

Cosine warmup does the same thing for neural network training. The
model starts slow to find its balance. It accelerates to full
speed once stable. It decelerates at the end to land softly on
the best possible solution.

## Where is it used

The schedule controls the learning rate parameter inside the
optimizer. Every training step the schedule computes a new
learning rate and assigns it to the optimizer.

```python
scheduler = CosineWarmupScheduler(optimizer, warmup=2000, max_steps=100000)

for step in range(max_steps):
    loss = model(batch)
    loss.backward()
    optimizer.step()
    scheduler.step()  # Update learning rate every step
    optimizer.zero_grad()
```

## Why we need it

A constant learning rate seems simpler. Why not just pick one
value and train the whole way. Two reasons.

First early training is chaotic. The model's weights are random.
The gradients are large and noisy. Large learning rates at the
start can send the model flying off in random directions. The
warmup phase lets the model find its footing before taking large
steps.

Second late training is about precision. After thousands of steps
the model is close to a good solution. Large steps would overshoot
the minimum and bounce around it forever. The decay phase lets
the model take tiny careful steps to settle into the exact best
position.

A constant learning rate would be either too large for the start
or too small for the middle. Warmup plus decay is the only way to
have both stability at the start and precision at the end.

## When was it invented

Learning rate warmup was used for the original transformer in
2017. The authors noticed that training was unstable in the first
few thousand steps without it. Cosine decay was introduced around
the same time as an alternative to step decay schedules which
drop the learning rate abruptly at predetermined intervals. Step
decay works but the sudden drops can disturb the model. Cosine
decay is smooth and continuous. GPT-3 used cosine warmup. LLaMA
used cosine warmup. It is the standard for language model
training.

## How it works

The schedule has three phases. Each phase is a simple
mathematical formula.

### Phase 1: Linear warmup

The learning rate starts at zero and increases linearly to the
maximum value.

```python
if step < warmup_steps:
    lr = max_lr * step / warmup_steps
```

Example with warmup_steps of 2000 and max_lr of 0.0003:

```
Step 0:    lr = 0.0003 × (0 / 2000) = 0.0
Step 500:  lr = 0.0003 × (500 / 2000) = 0.000075
Step 1000: lr = 0.0003 × (1000 / 2000) = 0.00015
Step 2000: lr = 0.0003 × (2000 / 2000) = 0.0003
```

Every step the learning rate grows by the same tiny amount. No
sudden jumps. The model has two thousand steps to get stable
before it reaches full speed.

### Phase 2: Cosine decay

After warmup the learning rate follows a cosine curve from the
maximum down to a minimum.

```python
if step < max_steps:
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    cosine_decay = 0.5 × (1 + cos(π × progress))
    lr = min_lr + (max_lr - min_lr) × cosine_decay
```

The progress variable goes from zero to one over the remaining
steps. The cosine function creates a smooth S shape curve.

```
Step 2000:  progress = 0.0, cosine = 1.0, lr = 0.0003
Step 25000: progress = 0.23, cosine = 0.75, lr = 0.000225
Step 50000: progress = 0.49, cosine = 0.25, lr = 0.000075
Step 100000: progress = 1.0, cosine = 0.0, lr = 0.00001
```

The learning rate falls slowly at first then faster in the middle
then slowly again at the end. The minimum is usually 0.00001
which is thirty times smaller than the peak. This tiny rate at
the end lets the model refine its weights with extreme precision.

### Phase 3: Minimum

After max_steps the learning rate stays at the minimum forever.

```python
lr = min_lr
```

The model continues to learn but at a glacial pace. Each step
makes almost no difference. This is intentional. The model has
already learned everything it needs. The remaining steps just
polish.

## A tiny code example

```python
import math
import matplotlib.pyplot as plt

max_lr = 0.0003
min_lr = 0.00001
warmup_steps = 2000
max_steps = 100000

lrs = []
for step in range(max_steps):
    if step < warmup_steps:
        lr = max_lr * step / warmup_steps
    elif step < max_steps:
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = min_lr + (max_lr - min_lr) * cosine
    else:
        lr = min_lr
    lrs.append(lr)

plt.figure(figsize=(10, 4))
plt.plot(lrs)
plt.xlabel('Training Step')
plt.ylabel('Learning Rate')
plt.title('Cosine Warmup Schedule')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lrs_curve.png', dpi=100)
print("The curve rises from 0 over 2000 warmup steps")
print("then decays along a cosine curve for 98000 steps")
print("and stays at the minimum from step 100000 onward")
```

## The shape of the curve

```
Learning rate
     ^
     |
0.0003 +         ....----....
     |        ..              ..
     |      ..                  ..
     |    ..                      ....
     |  ..                            .............
     |..                                            ...........
0.0  +----+----+----+----+----+----+----+----+----+----+---->
     0   10k  20k  30k  40k  50k  60k  70k  80k  90k  100k
                             Training steps
```

The curve rises steeply during warmup. It stays near the peak for
a while. Then it starts a gentle descent that accelerates in the
middle and flattens at the end. The minimum is reached exactly at
the final training step. Not before. Not after.

## What you need to remember

Cosine warmup scheduling controls the learning rate across the
entire training run. The rate starts at zero and warms up to a
peak. Then it decays along a cosine curve to a minimum. The
schedule is smooth and continuous with no sudden drops.

Warmup prevents instability at the start of training when
gradients are chaotic. Decay allows precision at the end when the
model is close to the solution. Together they make training both
stable and precise. Every modern language model uses this
schedule.

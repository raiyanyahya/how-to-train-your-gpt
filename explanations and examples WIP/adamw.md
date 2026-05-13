# AdamW: The Optimizer That Trains Language Models

## What is it

AdamW is the algorithm that updates the model's weights during
training. After computing how wrong a prediction was and
calculating which direction to move each weight AdamW decides
exactly how far to move. It does this intelligently based on the
history of past gradients for each parameter.

Think of it like hiking down a mountain in the fog. You cannot
see the bottom. You can only feel which direction is downhill.
You take a step. Then you feel again. A naive hiker always takes
the same size step. But some parts of the mountain are steep
and need big steps. Others are flat and need small steps. AdamW
remembers how steep each parameter has been and adjusts the step
size accordingly. It also remembers the general direction to keep
momentum going.

The W in AdamW stands for decoupled weight decay. This is the key
innovation over the original Adam optimizer. Weight decay slowly
pushes all weights toward zero to prevent them from growing too
large. In AdamW this push is separated from the gradient
calculation. The separation makes weight decay work correctly.

## Where is it used

AdamW is called every training step after backward propagation.
It takes the gradients that have been computed and clipped and
applies them to the model weights.

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()  # AdamW updates weights here
optimizer.zero_grad()
```

## Why we use it instead of plain gradient descent

Plain gradient descent is simple. Move each weight in the
direction that reduces the loss. The step size is the same for
every weight.

```
weight = weight - learning_rate × gradient
```

This has three problems.

First the step size is fixed. A weight that needs a big change
gets the same size step as a weight that needs a tiny change. The
learning rate must be chosen for the most sensitive weights. This
makes training slow for all other weights.

Second there is no momentum. If the gradient is noisy and points
in a different direction each step the optimizer zigzags back and
forth making slow progress. Momentum smooths out the noise by
incorporating the direction from previous steps.

Third there is no weight decay. Without regularization weights
can grow arbitrarily large. Large weights mean the model is over
confident about some patterns and ignores others. The model
overfits.

AdamW solves all three problems.

## When was it invented

Adam was published in 2014 by Diederik Kingma and Jimmy Ba. It
quickly became the default optimizer for deep learning. But
researchers noticed that the weight decay implementation in Adam
was entangled with the adaptive learning rates. This meant weight
decay did not actually prevent large weights. It mostly just
slowed down training.

AdamW was proposed in 2017 by Ilya Loshchilov and Frank Hutter.
They showed that decoupling weight decay from the adaptive
learning rates fixed the problem. AdamW achieved better
generalization than Adam with the same hyperparameters. The fix
was simple but the impact was large. GPT-3 trained with AdamW.
LLaMA trained with AdamW. Every modern language model uses AdamW.

## How it works

AdamW maintains two running averages for each parameter. The first
is the momentum which tracks the average direction of recent
gradients. The second is the velocity which tracks the average
magnitude of recent gradients.

### Step 1: compute the noisy gradient

```python
gradient = compute_gradient(loss, weight)
```

This is the raw signal from one batch of data. It is noisy. A
single batch might give a misleading direction.

### Step 2: update the momentum

```python
momentum = beta1 × momentum + (1 - beta1) × gradient
```

The momentum is a weighted average of past gradients. Beta1 is
usually 0.9. This means recent gradients count for ninety percent
and older gradients fade away. The momentum smooths out noise and
gives a stable direction.

### Step 3: update the velocity

```python
velocity = beta2 × velocity + (1 - beta2) × gradient²
```

The velocity tracks how much each parameter has been moving.
Beta2 is usually 0.95. Parameters that have been making large
moves get a high velocity. Parameters that have been sitting
still get a low velocity.

### Step 4: bias correction

Both momentum and velocity start at zero. In the first few steps
they are biased toward zero. The bias correction fixes this.

```python
momentum_corrected = momentum / (1 - beta1^t)
velocity_corrected = velocity / (1 - beta2^t)
```

Where t is the current step number. After many steps the
correction becomes negligible. But in the first few steps it
prevents the optimizer from taking tiny useless steps.

### Step 5: decoupled weight decay

```python
weight = weight × (1 - learning_rate × weight_decay)
```

This shrinks every weight by a tiny fraction. Weight decay is
usually 0.1. With a learning rate of 0.0003 each weight is
multiplied by 0.99997 per step. Over thousands of steps this
gently pushes weights toward zero. Only weights that constantly
receive strong gradients survive. Weights that are not useful
fade away.

Note that this step happens before the gradient update and is
completely independent of the gradient. This is the decoupled
part of AdamW. In the original Adam weight decay was mixed in
with the gradient scaling which made it ineffective.

### Step 6: apply the gradient

```python
weight = weight - learning_rate × momentum_corrected / (sqrt(velocity_corrected) + eps)
```

The gradient step is scaled by the learning rate. Then it is
divided by the square root of the velocity. Parameters with high
velocity have been changing a lot so we take smaller steps.
Parameters with low velocity have been stable so we can take
larger steps. The epsilon prevents division by zero.

## Two parameter groups

Not all parameters should get weight decay. The biases and
normalization weights are one dimensional. They adjust the offset
and scale of activations. Pushing them toward zero would prevent
them from doing their job. We create two groups of parameters
with different weight decay values.

```python
def create_optimizer(model, config):
    decay_params = []      # Linear and embedding weights
    no_decay_params = []   # Biases and normalization weights

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() <= 1 or 'norm' in name.lower() or 'bias' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': 0.1},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
```

The decay group gets weight decay of 0.1. The no decay group gets
zero weight decay. Each group is treated separately by the
optimizer.

## The hyperparameters

Every optimizer has settings called hyperparameters. For AdamW
the important ones are:

```
learning_rate = 0.0003  (3e-4)
  How big a step to take. Smaller is safer but slower.

betas = (0.9, 0.95)
  How much to trust past gradients. Higher means smoother updates.

weight_decay = 0.1
  How aggressively to push weights toward zero. Higher prevents
  overfitting but too high makes the model forget.

eps = 0.00000001 (1e-8)
  A tiny number to prevent division by zero. Never needs tuning.
```

These values are the LLaMA defaults and have been battle tested
on models from one billion to seventy billion parameters. Unless
you are doing something unusual there is rarely a reason to
change them.

## What you need to remember

AdamW is the standard optimizer for training language models. It
combines momentum for stability adaptive learning rates for
efficiency and decoupled weight decay for regularization. The
three mechanisms work together to make training fast stable and
resistant to overfitting.

Every production language model trains with AdamW. The
hyperparameters are well established and rarely need tuning. Like
gradient clipping it has no meaningful downside. It is simply the
right tool for the job.

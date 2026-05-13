# SwiGLU: The Smart Activation Function

## What is it

SwiGLU is the activation function used inside the feed forward
network of modern transformers. An activation function decides
how much information passes through a layer. Old activation
functions were simple on or off switches. SwiGLU is smarter. It
has a second path that acts like a gate. The gate learns when to
let information through and when to block it.

Think of it like a water faucet. ReLU is a faucet that is either
fully open or fully closed. Nothing in between. SwiGLU is a
faucet you can turn to any position. A little open for a trickle.
Half open for moderate flow. Fully open when you need everything.
The model learns the right position for every input.

## Where is it used

SwiGLU lives inside every transformer block. It replaces the
older activation functions inside the feed forward network. Every
time the model processes a token through the FFN layer SwiGLU
decides what information to keep and what to throw away.

```
Transformer Block:
  x → RMSNorm → Attention → +x
    → RMSNorm → SwiGLU FFN → +x
                   ^^^^^^
                   This part
```

LLaMA PaLM Gemini and most models built since 2022 use SwiGLU.
GPT-2 and GPT-3 used GELU which was the previous best. SwiGLU
beats GELU at every scale.

## Why we use it instead of ReLU or GELU

ReLU is the simplest activation. It outputs zero for negative
numbers and does nothing for positive numbers.

```
ReLU(x): max(0, x)

ReLU(-3.2) = 0    (blocked)
ReLU(0.5)  = 0.5  (passed)
ReLU(4.1)  = 4.1  (passed)
```

The problem with ReLU is the hard cutoff at zero. Any negative
value is completely killed. The information is gone forever. This
is called the dying ReLU problem. Neurons that receive only
negative inputs never activate again. They become dead weight.

GELU fixes this by making the cutoff smooth. Instead of a hard
zero GELU outputs very small values for negative inputs.

```
GELU(-3.2) ≈ -0.002  (mostly blocked but not dead)
GELU(0.5)  ≈ 0.346   (partially passed)
GELU(4.1)  ≈ 4.100   (mostly passed)
```

GELU is better than ReLU but still has one decision point. Every
input gets the same treatment. There is no way for the model to
decide *this* input should pass through more than *that* input.

SwiGLU adds a gate. The input splits into two paths. One path
computes values like a normal activation. The other path computes
how much of those values to keep. The gate and the values are
computed from the same input using different learned weights.

```
SwiGLU(x) = (SiLU(x × W₁)) × (x × W₂)

Path 1 (values): SiLU(x × W₁) → the information
Path 2 (gate):   x × W₂       → how much information to pass
```

The gate can output any number. If the gate outputs 0.1 the value
path is reduced to ten percent. If the gate outputs 5.0 the value
path is amplified five times. The model learns what to amplify
and what to suppress. This is why SwiGLU outperforms both ReLU
and GELU at large scale.

## When was it invented

The paper that introduced SwiGLU was published in 2020 by Noam
Shazeer a well known researcher who also co invented the
transformer. The paper compared many activation variants and
found that gated linear units consistently won. PaLM adopted it
in 2022. LLaMA adopted it in 2023. Now it is the standard.

## How it works step by step

Let us trace a single number flowing through SwiGLU.

### The setup

```
Input x = 1.5

Weights (learned during training):
W₁ = 0.8   (for the value path)
W₂ = 2.0   (for the gate path)
```

### Path 1: compute the value

First multiply the input by W₁.

```
x × W₁ = 1.5 × 0.8 = 1.2
```

Then apply SiLU. SiLU is also called the Swish function. It is
x multiplied by the sigmoid of x.

```
SiLU(1.2) = 1.2 × sigmoid(1.2)

sigmoid(1.2) = 1 / (1 + e^(-1.2))
             = 1 / (1 + 0.301)
             = 1 / 1.301
             = 0.769

SiLU(1.2) = 1.2 × 0.769 = 0.922
```

SiLU gives 0.922. This is the processed value.

### Path 2: compute the gate

Simply multiply the input by W₂.

```
x × W₂ = 1.5 × 2.0 = 3.0
```

The gate value is 3.0. This means let three times the information
through. The gate is open wide.

### Combine the two paths

Multiply the value by the gate.

```
output = 0.922 × 3.0 = 2.766
```

If the gate had been smaller like 0.1 the output would have been
0.092. If the gate had been zero the output would have been zero.
The gate controls everything.

### What about negative inputs

Let us try an input of -2.0.

```
x = -2.0

Path 1 (value):
  x × W₁ = -2.0 × 0.8 = -1.6
  SiLU(-1.6) = -1.6 × sigmoid(-1.6)
  sigmoid(-1.6) = 1 / (1 + e^1.6) = 1 / 5.953 = 0.168
  SiLU(-1.6) = -1.6 × 0.168 = -0.269

Path 2 (gate):
  x × W₂ = -2.0 × 2.0 = -4.0

Combine:
  output = -0.269 × (-4.0) = 1.076
```

Even though the input was negative the output is positive. That
is because both the value path and the gate path became negative
and negative times negative equals positive. The gating mechanism
gives the model extra flexibility to transform negative signals
into positive ones when needed. ReLU would have just output zero
and lost all information.

## A tiny code example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d_model, expansion_factor=4):
        super().__init__()
        hidden_dim = expansion_factor * d_model
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        # Path 1: values processed by SiLU
        values = F.silu(self.w1(x))
        # Path 2: gates controlling how much passes
        gates = self.w2(x)
        # Combine and project back to original size
        return self.w3(values * gates)

# Test with random input
d_model = 4
ffn = SwiGLU(d_model)
x = torch.tensor([[1.5, -2.0, 0.3, 4.1]])

output = ffn(x)
print(f"Input:  {x}")
print(f"Output: {output}")
print(f"Shape preserved: {x.shape == output.shape}")
```

Running this code you will see something like:

```
Input:  tensor([[ 1.5000, -2.0000,  0.3000,  4.1000]])
Output: tensor([[-1.234,  0.567, -0.891,  2.345]])
Shape preserved: True
```

## Why the expansion factor matters

Notice the hidden dimension in the code is four times larger than
the input dimension. This is the expansion factor. The network
goes from d_model to four times d_model and back again.

```
768 → 3072 → 768
```

This expand then contract pattern gives the network room to
transform information. In the middle layer there are many more
neurons than at the input or output. This is like widening a pipe
to let more water flow through before narrowing it again. The
extra width lets the model learn more complex transformations.

SwiGLU uses three weight matrices instead of the two that ReLU
or GELU networks use. The extra matrix is for the gate. This
makes SwiGLU about fifty percent larger than a standard FFN at
the same expansion factor. For our GPT-2 scale model this adds
about twenty eight million extra parameters. Every one of those
parameters contributes to better performance.

## What you need to remember

SwiGLU is a gated activation function. It splits the feed forward
network into a value path and a gate path. The gate controls how
much of each value passes through. This is more flexible than
ReLU or GELU which treat every input the same way.

The SiLU function on the value path provides smooth non linearity.
The gate on the control path provides adaptive filtering. Together
they outperform every older activation function at large scale.

Every modern language model uses SwiGLU. It is one extra matrix
multiplication per forward pass for a measurable improvement in
every metric that matters.

# Pre-Norm vs Post-Norm: Where to Normalize

## What is it

Pre-norm and post-norm are two ways to place normalization inside
a transformer block. Pre-norm normalizes before each sublayer.
Post-norm normalizes after each sublayer. The difference is one
line of code but the impact on training is enormous.

Pre-norm (modern):
```
x = x + Attention(Norm(x))
x = x + FFN(Norm(x))
```

Post-norm (original):
```
x = Norm(x + Attention(x))
x = Norm(x + FFN(x))
```

Think of it like editing a document. Pre-norm cleans the messy
draft before you make changes. You start with a clear base. Post-
norm makes changes to a messy draft and then cleans the result.
The edits might be based on noise.

## Where is it used

The choice between pre-norm and post-norm affects every
transformer block in the model. For a twelve block model the
choice is made twenty four times per forward pass.

## Why pre-norm won

The original transformer paper used post-normalization. The
authors normalized the output of each sublayer. This worked for
models with six layers. When researchers tried to scale to more
layers training became unstable. The model would not converge
past about twelve layers.

The problem was the residual connection. In post-norm the
residual path goes through normalization. This means the gradient
flowing back through the residual connection is also normalized.
Normalization squashes the gradient magnitude. After many layers
the gradient becomes too small to train the early layers.

In pre-norm the residual path skips the normalization entirely.
The gradient flowing back through the residual connection is
never normalized. It arrives at the early layers with full
strength. This is why pre-norm enables training models with
ninety six layers or more.

```
Post-norm gradient path:
  loss → Norm → Sublayer → Norm → Sublayer → ... → input
  Each Norm compresses the gradient.
  After N layers the gradient is 0.5^N smaller.

Pre-norm gradient path:
  loss → + → ... → + → input (through residual connections)
  No normalization on the residual path.
  Gradient arrives at full strength regardless of depth.
```

## When was it discovered

Pre-norm was proposed in 2019 by researchers at Google studying
why deep transformers were hard to train. They found that
swapping the normalization position made training stable at any
depth. GPT-3 adopted pre-norm in 2020. Every model since has
used pre-norm. Post-norm is now only found in legacy code and
historical comparisons.

## How they differ step by step

Let us trace a single token through one block with both
approaches.

### Pre-norm (what we use)

```
x = some vector [0.5, -0.3, 0.8, -0.1]

Step 1: norm(x) = [0.6, -0.4, 1.0, -0.1]  (rescaled to be cleaner)
Step 2: attention(norm(x)) = [0.1, 0.0, -0.2, 0.3]
Step 3: x + attention = [0.6, -0.3, 0.6, 0.2]
Step 4: norm(step3_result) = [0.8, -0.4, 0.8, 0.3]
Step 5: ffn(norm(step4_result)) = [-0.1, 0.2, 0.0, 0.1]
Step 6: step3_result + ffn = [0.5, -0.1, 0.6, 0.3]
```

The output [0.5, -0.1, 0.6, 0.3] is similar to the input [0.5,
-0.3, 0.8, -0.1]. The model made small adjustments to a clean
base.

### Post-norm (original paper)

```
x = some vector [0.5, -0.3, 0.8, -0.1]

Step 1: attention(x) = [2.5, -0.1, -1.8, 0.9]  (big outputs)
Step 2: x + attention = [3.0, -0.4, -1.0, 0.8]
Step 3: norm(step2_result) = [1.5, -0.2, -0.5, 0.4]  (squashed)
Step 4: ffn(step3_result) = [-0.8, 1.2, -0.3, 0.7]
Step 5: step3_result + ffn = [0.7, 1.0, -0.8, 1.1]
Step 6: norm(step5_result) = [0.4, 0.6, -0.5, 0.7]
```

The output is more different from the input because the
normalization at each step changes everything. This sounds good
in theory. More transformation. But in practice the constant
normalization interferes with gradient flow and makes deep
networks untrainable.

## The gradient argument

Imagine a network with ninety six layers. Each layer has two
normalization operations. In post-norm the gradient flows through
one hundred and ninety two normalization operations on its way
back to the first layer. Each normalization compresses the
gradient slightly. After one hundred and ninety two compressions
the gradient at layer one is effectively zero.

In pre-norm the residual connections bypass the normalization.
The gradient flows straight down the residual highway. It never
passes through normalization on the shortcut path. Only the
path through the sublayers goes through normalization. The
shortcut path provides a strong gradient signal to every layer.

This is why pre-norm is a hard requirement for deep transformers.
It is not a preference. It is a necessary condition for training
to work at all beyond a certain depth.

## What you need to remember

Pre-norm normalizes the input before each sublayer. Post-norm
normalizes the output after each sublayer. Pre-norm enables
training deep networks because the residual connections bypass
normalization and preserve the gradient signal. Post-norm
restricts training to shallow networks because the normalization
squashes the gradients.

Every modern language model uses pre-norm. If you see post-norm
in a codebase it is either a bug or a historical artifact. The
original transformer paper was wrong about this one detail. The
fix was discovered a year later and has been standard ever since.

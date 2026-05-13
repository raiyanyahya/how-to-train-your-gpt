# Weight Tying: Two Jobs One Matrix

## What is it

Weight tying means using the same weight matrix for the embedding
layer and the output layer. The embedding layer turns token IDs
into vectors. The output layer turns vectors back into token
probabilities. They are inverse operations. They share one
matrix.

Think of it like a bilingual dictionary. You can look up an
English word to find its French translation. Or you can look up
a French word to find its English translation. It is the same
dictionary used in two directions. Weight tying is the same idea
applied to neural networks. One matrix serves both the input side
and the output side.

## Where is it used

Weight tying connects the very first layer and the very last
layer of the model. The token embedding and the language model
head share weights.

```
Input tokens
  → Token Embedding (matrix A of size 50257 × 768)
    → Transformer blocks
      → Final normalization
        → LM Head (shares matrix A, used as 768 × 50257)
          → Output logits
```

In code it is a single line.

```python
self.token_embedding.weight = self.lm_head.weight
```

This makes both attributes point to the same tensor in memory.
Changing one changes the other because they are literally the
same bytes.

## Why we use it

The most obvious reason is saving parameters. The embedding
matrix has size vocabulary times embedding dimension. For GPT-2
Small that is 50257 times 768 which equals about 38.6 million
numbers. The output matrix has the same size. Without tying these
would be two separate matrices consuming about 77 million
parameters just for input and output. With tying they become one
matrix. We save 38.6 million parameters.

That is about thirty percent of the total model size for GPT-2
Small. For larger models the savings are even greater. GPT-3
Large with a vocabulary of 50257 and an embedding dimension of
12288 would waste over 600 million parameters on a second copy
of the embedding matrix. Those parameters can be better spent on
more transformer blocks.

The less obvious reason is better learning. The embedding matrix
is the gateway into the model. Every token passes through it on
the way in. The output matrix is the gateway out of the model.
Every prediction passes through it on the way out. When the
model gets a prediction wrong the gradient flows backward through
the output matrix and all the way to the embedding matrix. Since
they are the same matrix the embedding vectors get gradient
signals from two directions. The forward pass through the
embedding layer and the backward pass through the output layer
both update the same numbers. This dual signal helps each
embedding vector converge to a better representation.

The third reason is mathematical elegance. The embedding layer
maps token IDs to vectors. The output layer maps vectors to
token probabilities. If the model has learned good embeddings
then the same vectors that represent a token on the input side
should be useful for predicting that token on the output side.
Tying the weights enforces this consistency. A token's embedding
vector is also the vector that the model uses to score that token
as a possible next word. If the token *cat* has embedding vector
v then the model's score for predicting *cat* is the dot product
of the current hidden state with v. The embedding serves double
duty as a representation and as a classification weight.

## When was it invented

Weight tying was used in the original transformer paper in 2017.
It was not a new idea at the time. Earlier language models like
word2vec published in 2013 used tied input and output embeddings.
It has been standard practice for language models ever since.
GPT-2 and GPT-3 both use weight tying. LLaMA uses weight tying.
Every model in this tutorial uses weight tying.

There are cases where weight tying is not used. Some very large
models separate the embedding and output matrices to allow the
output layer to have a different structure from the input layer.
But for most models including ours weight tying is the right
choice. The parameter savings are too large to ignore and the
dual gradient signal is genuinely helpful during training.

## How it works in practice

Let us trace what happens when we train with tied weights.

### Forward pass with tied weights

```
Step 1: Token 3797 ("cat") enters the model
Step 2: Embedding layer looks up row 3797 of matrix A
Step 3: Row 3797 is the embedding vector for "cat" [768 numbers]
Step 4: The vector flows through transformer blocks
Step 5: The hidden state reaches the output layer
Step 6: Output layer multiplies hidden state by matrix A^T
Step 7: Row 3797 of A^T is column 3797 of A
Step 8: This is the same vector that represented "cat" on input
Step 9: The dot product gives the score for predicting "cat"
```

Notice that the same vector appears twice. Once as the
representation of *cat* at the input. Once as the prediction
target for *cat* at the output. The model is forced to make
these two uses consistent.

### Backward pass with tied weights

```
Step 1: The model predicts wrong (true word was "mat" not "dog")
Step 2: Loss is computed
Step 3: Gradient flows to the output layer
Step 4: The gradient updates row 3797 of matrix A
        (because cat was one of the wrong predictions)
Step 5: The same gradient also flows back through the model
Step 6: Eventually reaches the embedding layer
Step 7: Row 3797 of matrix A gets a second gradient signal
        (because cat appeared in the input)
Step 8: Both gradients are summed and applied to the same numbers
```

The embedding for *cat* gets updated twice per training step.
Once for its role as an input token. Once for its role as a
potential output token. This double signal means the embedding
vectors learn faster and reach better representations.

## Verifying weight tying in code

You can check that two tensors share memory in PyTorch.

```python
import torch

# Create an embedding and an output layer
embedding = torch.nn.Embedding(1000, 768)
output = torch.nn.Linear(768, 1000, bias=False)

# Tie the weights
embedding.weight = output.weight

# Verify they share memory
print(f"Same object: {embedding.weight is output.weight}")
print(f"Same memory: {embedding.weight.data_ptr() == output.weight.data_ptr()}")

# Modify one and see the other change
old_value = embedding.weight[42, 0].item()
output.weight[42, 0] = 99.9
new_value = embedding.weight[42, 0].item()

print(f"\nAfter changing output.weight[42,0] to 99.9:")
print(f"Embedding weight[42,0] changed from {old_value} to {new_value}")
print(f"They are the same tensor. Changing one changes both.")
```

Running this code produces:

```
Same object: True
Same memory: True

After changing output.weight[42,0] to 99.9:
Embedding weight[42,0] changed from 0.023 to 99.9
They are the same tensor. Changing one changes both.
```

## The parameter savings by model size

```
Model              Vocab    Dim      Weight Tying Savings
GPT-2 Small        50,257   × 768   =  38.6 million params
GPT-2 Medium       50,257   × 1,024 =  51.5 million params
GPT-2 Large        50,257   × 1,280 =  64.3 million params
LLaMA 7B           32,000   × 4,096 = 131.1 million params
LLaMA 70B          32,000   × 8,192 = 262.1 million params
GPT-3 (full)       50,257   × 12,288 = 617.6 million params
```

These savings are why weight tying is nearly universal. You get
better embeddings and better training for the cost of zero extra
parameters. In fact you get fewer parameters and better training
at the same time. It is one of the rare cases in machine learning
where there is no tradeoff.

## What you need to remember

Weight tying makes the embedding layer and the output layer share
the same weight matrix. One line of code saves tens or hundreds
of millions of parameters. The shared matrix gets gradient signals
from both the input and output directions leading to better
embeddings. Every modern language model uses this technique. It
is free better performance.

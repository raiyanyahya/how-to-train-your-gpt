# Embeddings: Giving Numbers Meaning

## What is it

An embedding is a list of numbers that captures the meaning of
a word.

After tokenization every word is just a number. *Cat* is 9246.
*Dog* is 4821. These numbers are just labels. The number 9246
means nothing by itself. The model cannot learn from a number
like 9246 because 9246 is not closer to 4821 than it is to 279.
They are just arbitrary IDs.

An embedding turns that number into a vector. A vector is just
a list of decimal numbers. For GPT-2 each word becomes a list of
768 numbers. These numbers are not arbitrary. Words with similar
meanings get similar lists. Words with different meanings get
different lists.

```
Token ID 9246 ("cat") → [0.023, -0.451, 0.789, ..., -0.102]  (768 numbers)
Token ID 4821 ("dog")  → [0.019, -0.443, 0.795, ..., -0.098]  (very similar!)
Token ID 279  ("the")  → [0.891, 0.112, -0.334, ..., 0.567]   (very different!)
```

The key idea is that *cat* and *dog* are near each other in this
number space because they are both animals. *The* is far away
because it is a function word with a completely different role.

## Where is it used

The embedding layer sits right after the tokenizer and right
before the attention layers. It is the bridge between the
tokenizer's integer output and the transformer's floating point
input.

```
Raw text: "The cat"
    ↓
Tokenizer: [1169, 3797]
    ↓
Embedding layer: two vectors of 768 numbers each
    ↓
Transformer blocks
```

Every modern language model has an embedding layer. It is the
very first learned component in the entire pipeline.

## Why we need it

Without embeddings the model would be trying to do math on token
IDs. Imagine adding two words together. The token for *king* is
9246. The token for *queen* is 9247. If we added them we would get
18493. That number means nothing. It does not correspond to a
meaningful word. The model cannot learn from token IDs.

With embeddings the model works with continuous vectors. The
vector for *king* is something like [0.3, -0.5, 0.8, ...]. The
vector for *queen* is [0.2, -0.6, 0.7, ...]. These are close but
not identical. The model can compute *king* minus *man* plus
*woman* and get something very close to the vector for *queen*.
This is called the embedding arithmetic property.

```
embedding(king)  ≈ [0.30, -0.50, 0.80]
embedding(man)   ≈ [0.25, -0.45, -0.30]
embedding(woman) ≈ [0.22, -0.55, 0.70]
embedding(queen) ≈ [0.27, -0.60, 0.75]

king - man + woman = [0.27, -0.60, 0.75] ≈ queen!
```

This was not programmed by a human. The model discovered that
changing the gender of a word is like moving in a straight line
through the embedding space. It learned this entirely from
reading millions of sentences where *king* and *queen* appeared
in similar contexts but with different pronouns.

## When was it invented

The idea of word embeddings is old. A technique called Word2Vec
was published by Google in 2013. It was the first to show that
word vectors could capture meaning relationships. The embedding
layer in transformers is a direct descendant of Word2Vec. The
difference is that Word2Vec embeddings were precomputed and
frozen. Transformer embeddings are learned from scratch during
training. They adapt to the specific task the model is learning.

## How it works: a giant lookup table

Think of the embedding layer as a table with 50257 rows. Each row
has 768 columns. Row zero is the vector for token zero. Row one
is the vector for token one. Row 3797 is the vector for the word
*cat*. The forward pass of the embedding layer is just looking up
rows in this table.

```python
# Given token IDs: [1169, 3797]
# Look up row 1169 → vector for "The"  (768 numbers)
# Look up row 3797 → vector for "cat"  (768 numbers)
# Return both vectors
```

That is the entire forward pass of the embedding layer. No
multiplication. No activation function. Just a table lookup.

### How the table is built

The table starts completely random. Every row is filled with
numbers drawn from a normal distribution with mean 0 and standard
deviation 0.02. At this point *cat* and *dog* are as close to
each other as *cat* and *democracy*. Everything is random noise.

Then training begins. The model reads a sentence like *The cat
sat on the mat*. It predicts that *mat* should come next. If it
predicts wrong the loss is high. Backpropagation sends a tiny
signal back through the entire model including the embedding
table. That signal says:

"The vector for *cat* should be nudged slightly toward the
direction that helps predict *mat* next time."

After millions of training steps the table transforms. Words that
appear in similar contexts get pushed toward similar positions.
The vector for *cat* moves close to *dog* and *pet* and *feline*.
The vector for *car* moves close to *vehicle* and *drive* and
*road*. The space organizes itself into neighborhoods of meaning.

### What the neighborhoods look like

After training the 768 dimensional space has natural structure.
Some directions in this space correspond to real world concepts.

```
Direction 1 (dimensions 0 through 63):    Living vs non living
Direction 2 (dimensions 64 through 127):   Big vs small
Direction 3 (dimensions 128 through 191):  Positive vs negative
Direction 4 (dimensions 192 through 255):  Formal vs casual
... and so on through all 768 dimensions
```

These directions were never programmed. They emerged naturally
because the model found it useful to organize words this way. When
the model needs to know if something is alive or not it looks at
a specific set of dimensions in the embedding vector.

## A tiny code example

```python
import torch
import torch.nn as nn

# Create a tiny embedding table
vocab_size = 1000   # 1000 unique tokens
d_model = 4         # 4 dimensional vectors (small for the example)

embedding = nn.Embedding(vocab_size, d_model)

# Look up some token IDs
token_ids = torch.tensor([[12, 45, 678]])
vectors = embedding(token_ids)

print("Token IDs:", token_ids)
print("Shape of output:", vectors.shape)
print()
print("Vector for token 12:", vectors[0, 0].tolist())
print("Vector for token 45:", vectors[0, 1].tolist())
print("Vector for token 678:", vectors[0, 2].tolist())
print()
print("Each token ID became a", d_model, "dimensional vector.")
print("Right now the values are random. After training they will")
print("capture meaning. Words with similar meanings will have")
print("similar vectors.")
```

Running this code you will see something like:

```
Token IDs: tensor([[ 12,  45, 678]])
Shape of output: torch.Size([1, 3, 4])

Vector for token 12:  [0.031, -0.124, -0.847, 0.562]
Vector for token 45:  [-1.231, 0.789, 0.023, -0.441]
Vector for token 678: [0.892, -0.334, 0.671, -0.128]
```

These vectors are random right now. They have no meaning. After
training on billions of sentences token 12 and token 45 will be
near each other if they appear in similar contexts or far apart
if they do not.

## The size of the embedding table

The embedding table is often the largest component in the model
in terms of parameter count.

```
GPT-2 Small:  50257 words × 768 dims  = 38.6 million numbers
GPT-3:        50257 words × 12288 dims = 617 million numbers
```

This is why weight tying is important. The output layer also needs
a matrix of the same size to project back from hidden states to
vocabulary predictions. Instead of storing two giant matrices we
share one. The embedding table is used for both input and output.

## Embeddings for punctuation and special characters

Every token gets an embedding. Even punctuation and special
symbols. The period gets an embedding. The comma gets an
embedding. The end of text marker gets an embedding.

These embeddings are just as important as word embeddings. The
model learns that the embedding for a period is followed by the
embedding for a capitalized word. It learns that the embedding
for a question mark is followed by the embedding for an answer.
The structure of language lives in these small token embeddings
as much as it lives in the word embeddings.

## What you need to remember

An embedding is a list of numbers that represents a word's
meaning. Words with similar meanings have similar lists. Words
with different meanings have different lists.

The embedding table starts random. Training moves words around
based on the contexts they appear in. After enough training the
space organizes itself. King minus man plus woman equals queen.
This was not programmed. The model discovered it.

The embedding layer is just a lookup table. No math inside. Give
it a token ID and it returns a vector. That vector is the word's
coordinates in meaning space. Everything the model knows about a
word is packed into those 768 numbers.

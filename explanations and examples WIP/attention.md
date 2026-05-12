# Attention : How Words Talk To Each Other

## What is it

Attention is how a language model decides which words are
important when it reads a sentence.

Imagine you are at a loud party. Ten people are talking at the
same time. You are trying to understand one person. You do not
listen to everyone equally. You pay more attention to the person
you are talking to. You pay less attention to the person across
the room. Your brain naturally focuses on what matters.

Attention does the same thing for words. When the model reads a
sentence it looks at all the words at once. Then it decides how
much to care about each word. The word it cares about most gets
the most weight. The word it cares about least gets almost no
weight. Then it combines all the words together with these
weights. The result is a new understanding of the current word
that includes information from every other word that matters.

## Where is it used

Attention is the heart of every transformer model. It sits inside
each transformer block. The model runs attention once per block.
If the model has twelve blocks it runs attention twelve times.
Each time the words get better at understanding each other.

```
Sentence → Embeddings → [Attention → FFN] × 12 → Output
```

## Why we need it

Before attention existed models read words one at a time from
left to right. By the time they reached the end of a sentence
they had forgotten what was at the start. Like a person who
forgets the beginning of a story by the time you finish telling
it. Attention fixes this by letting the model look at every word
at the same time. No forgetting. No fading memory.

Here is a concrete example:

```
"The cat sat on the mat because it was warm"
```

What does *it* mean here. A human knows it means the mat. Not the
cat. How do we know. Because warmth is a property of objects like
mats. Not of animals like cats. We connect the word *warm* with
the word *mat* and ignore the word *cat*. We do this without
thinking. It is automatic.

A model without attention reads left to right. By the time it
reaches the word *it* the word *mat* was six words ago. The signal
has faded. The model cannot remember. It might think *it* refers
to *cat* which makes no sense with the word *warm*.

With attention the word *it* can look back at every word that
came before. It sees *mat* and notices that *mat* is connected to
*warm*. It sees *cat* and notices that *cat* is less connected to
*warm*. It puts more weight on *mat*. The model resolves the
meaning correctly.

## When was it invented

Attention was introduced in 2017 in a paper called Attention Is
All You Need. The title was bold. The authors claimed you do not
need any other mechanism. Just attention is enough. They were
right. Every major language model since 2018 has been built
entirely on attention.

## How it works : a step by step story

Let us trace through a real example with three words and real
numbers. You can follow along and see every calculation.

### The setup

We have three words in our sentence. The model gave each word a
vector of four numbers. These numbers represent the meaning.

```
Word 0 ("I"):    [ 0.5,  0.2, -0.3,  0.8]
Word 1 ("love"): [ 0.1, -0.5,  0.7, -0.2]
Word 2 ("dogs"): [ 0.9,  0.3, -0.1, -0.5]
```

The model wants to understand the word *dogs*. But it should think
about *I* and *love* too because they give context. *I love dogs*
is different from *I fear dogs*. The words around *dogs* matter.

### Step 1 : Create three things for each word

The model takes each word and multiplies it by three learned
matrices. The result is three new vectors called Query Key and
Value. Every word gets its own Q K V.

```
Query = "What am I looking for?"
Key   = "What do I have to offer?"
Value = "This is my actual content"
```

Think of it like a dating app. The Query is your profile saying
what you want. The Key is everyone else saying what they offer.
When your Query matches someone's Key you pay attention to them.

After the matrix multiplication our three words become:

```
Word    | Query              | Key                | Value
"I"     | [ 0.8,  0.1]      | [ 0.6, -0.3]      | [ 0.4,  0.9]
"love"  | [-0.2,  0.7]      | [ 0.1,  0.5]      | [-0.3,  0.2]
"dogs"  | [ 0.5, -0.4]      | [-0.4,  0.8]      | [ 0.7, -0.1]
```

Each vector has only two numbers here to keep things simple. Real
models use 64 numbers per head.

### Step 2 : Score every pair of words

Now we compare every Query with every Key. We take the dot
product. The dot product measures how well they match. A high
score means the Query really wants what that Key offers.

Let us compute how much *dogs* wants to attend to every word.

```
dogs looking at I:
  Q_dogs · K_I = (0.5 × 0.6) + (-0.4 × -0.3)
               = 0.30 + 0.12
               = 0.42

dogs looking at love:
  Q_dogs · K_love = (0.5 × 0.1) + (-0.4 × 0.5)
                  = 0.05 + (-0.20)
                  = -0.15

dogs looking at dogs (itself):
  Q_dogs · K_dogs = (0.5 × -0.4) + (-0.4 × 0.8)
                  = -0.20 + (-0.32)
                  = -0.52
```

The scores tell us: *dogs* matches *I* best (score is 0.42 which
is positive). *dogs* matches *love* somewhat (score is -0.15 which
is close to zero). *dogs* does not match itself well (score is
-0.52 which is quite negative).

### Step 3 : Scale the scores

We divide every score by the square root of the dimension size.
Our vectors have two dimensions. The square root of two is about
1.4. This scaling keeps the numbers from getting too big.

```
Scaled scores: [0.42/1.4, -0.15/1.4, -0.52/1.4]
             = [0.30, -0.11, -0.37]
```

Why do we scale. Without scaling the scores can get very large
when we use bigger vectors like 64 dimensions. Large scores make
the next step produce extreme results where one word gets all the
attention and everything else gets zero. Scaling keeps things
balanced.

### Step 4 : Apply the causal mask

During training each word can only see words that came before it.
Words that come after are hidden. This is like reading a book.
You do not know what is on the next page until you turn it.

In our example *I* can only see itself. *love* can see *I* and
itself. *dogs* can see all three. Words that should be hidden get
a score of negative infinity. After the next step negative
infinity becomes zero. Those words are completely ignored.

```
I can see:    [I, hidden, hidden]
love can see: [I, love, hidden]
dogs can see: [I, love, dogs]
```

### Step 5 : Turn scores into percentages (softmax)

Softmax takes our scores and turns them into percentages that
add up to 100 percent. This gives us attention weights.

```
For dogs looking at all words:
Scores: [0.30, -0.11, -0.37]

Step 1 : Take e to the power of each score:
         e^0.30 = 1.35
         e^-0.11 = 0.90
         e^-0.37 = 0.69

Step 2 : Sum them up:
         1.35 + 0.90 + 0.69 = 2.94

Step 3 : Divide each by the sum:
         1.35 / 2.94 = 0.46  (46 percent attention to I)
         0.90 / 2.94 = 0.31  (31 percent to love)
         0.69 / 2.94 = 0.23  (23 percent to itself)

Final attention weights for dogs: [0.46, 0.31, 0.23]
```

When reading the word *dogs* the model pays 46 percent attention
to *I* 31 percent to *love* and 23 percent to *dogs* itself.

### Step 6 : Mix the values using the attention weights

Now we have weights. We use them to mix the Value vectors. Words
we care about more get their Values multiplied by a bigger weight.

```
New representation of dogs = (0.46 × Value of I)
                            + (0.31 × Value of love)
                            + (0.23 × Value of dogs)

= 0.46 × [0.4, 0.9] + 0.31 × [-0.3, 0.2] + 0.23 × [0.7, -0.1]

= [0.184, 0.414] + [-0.093, 0.062] + [0.161, -0.023]

= [0.252, 0.453]
```

The new vector for *dogs* is [0.252, 0.453]. This is no longer
just the meaning of *dogs*. It now contains information about *I*
and *love* weighted by how much they matter. The word *dogs* now
knows about the words around it. It has context.

## The full attention matrix

Here is what all three words see after attention. Each row is a
word. Each column is what that word attends to.

```
           I       love    dogs
I         1.00    0.00    0.00     ← I can only see itself
love      0.45    0.55    0.00     ← love sees I and itself
dogs      0.46    0.31    0.23     ← dogs sees all three
```

The upper right corner is all zeros. That is the causal mask at
work. No word can see the future. This is how the model learns to
predict the next word without cheating.

## Multi head attention

Why stop at one attention calculation. Different aspects of
language need different kinds of attention. One head might focus
on grammar. Another head might focus on meaning. Another might
focus on whether words refer to the same thing.

In our real model we do attention twelve times in parallel. Each
time with different learned matrices for Q K and V. Each head
specializes in something different. Then we combine all twelve
heads back together.

```
Input → Head 1: grammar focus     ↘
        Head 2: meaning focus     →  Combine → Output
        Head 3: pronoun resolution ↗
        ...
        Head 12: positional focus
```

Each head works in a smaller space. If the model has 768
dimensions and 12 heads each head works with 64 dimensions. This
is like having twelve experts each looking at the same text
through a different lens. Their insights get combined.

## A tiny code example you can run

```python
import torch
import torch.nn.functional as F
import math

# Three words with four dimensions each
words = torch.tensor([[
    [0.5,  0.2, -0.3,  0.8],   # I
    [0.1, -0.5,  0.7, -0.2],   # love
    [0.9,  0.3, -0.1, -0.5],   # dogs
]])

# Create random Q K V matrices (normally these are learned)
d_model = 4
W_q = torch.randn(d_model, d_model)
W_k = torch.randn(d_model, d_model)
W_v = torch.randn(d_model, d_model)

# Project to Q K V
Q = words @ W_q
K = words @ W_k
V = words @ W_v

print("Query vectors:")
print(Q)
print()

# Compute attention scores
head_dim = 4
scores = (Q @ K.transpose(-2, -1)) / math.sqrt(head_dim)

# Create causal mask
seq_len = 3
mask = torch.tril(torch.ones(seq_len, seq_len))
mask = mask.view(1, 1, seq_len, seq_len)
scores = scores.masked_fill(mask == 0, float('-inf'))

# Softmax to get weights
weights = F.softmax(scores, dim=-1)

# Apply weights to values
output = weights @ V

print("Attention weights:")
print(weights)
print()

print("Output (context aware representations):")
print(output)
print()

print("The first row shows Token 0 attending only to itself.")
print("The last row shows Token 2 attending to all three tokens.")
```

## The big picture

Attention is like giving every word a flashlight. The word shines
its light on other words. Brighter light means more attention. The
model learns where to shine the light during training. After
enough training it knows that pronouns should light up the noun
they refer to. It knows that verbs should light up their subjects.
It knows that adjectives should light up the nouns they describe.

This is why attention is the secret sauce of modern AI. It lets
words understand each other instead of existing in isolation. A
sentence becomes a web of connections. Not just a list of words
in order.

## What you need to remember

Every word creates a Query a Key and a Value. The Query of one
word compares itself to the Keys of all other words. The match
scores become attention weights. The weights are used to mix the
Values. The result is a new understanding of each word that knows
about every other word that matters.

During training words cannot see the future. The causal mask
blocks them. This forces the model to predict what comes next
using only what came before. The same way you predict the ending
of a sentence when someone pauses mid thought.

Attention runs in parallel with multiple heads. Each head learns
a different pattern. One head might connect pronouns to nouns.
Another might connect verbs to subjects. The heads never talk to
each other during computation. They combine only at the end.

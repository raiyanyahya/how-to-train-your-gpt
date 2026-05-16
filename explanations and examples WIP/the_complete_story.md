# How a GPT Really Works: The Complete Story

This is the story of a language model. Not just one part. Not just one
step. The whole thing. From a text file on a hard drive to a machine
that can write poetry and answer questions and generate code. Every
single piece. Every single decision. Every single number.

We will build a GPT from scratch. We will train it. We will watch it
learn. We will run it. By the end you will understand every line of
code in every file. This story assumes you know Python. Nothing else.

---

## Part 1: What Are We Building

A GPT is a next word predictor. That is it. That is the whole thing.
You give it some words. It guesses what word comes next. Then it takes
that guess and guesses the next word. Then the next. Eventually it has
written a paragraph or a poem or a legal document or a recipe for
chocolate cake. But underneath it is always just guessing one word at
a time.

The model has about 150 million knobs. Each knob is a number. Training
means finding the right numbers for all 150 million knobs so that the
model's guesses match what a human would write. Once those numbers are
found the model can write text that is sometimes indistinguishable from
human writing.

How do we find those numbers. We show the model sentences from the
internet. Billions of sentences. For each sentence we hide the last word
and ask the model to guess it. When it guesses wrong we figure out which
knobs to turn and in which direction to make the guess better next time.
We repeat this billions of times. The knobs slowly converge to values
that capture the patterns of human language.

The architecture of the model determines which patterns it can capture.
A bigger model can capture more patterns. A better architecture can
capture more patterns with the same number of knobs. Our architecture
is the same one used by LLaMA 3 and Mistral and Qwen. It represents the
best publicly documented design for language models as of 2025.

---

## Part 2: The Data

Before we can train a model we need text. Lots of text. Billions of
words. For this project we will use Wikipedia because it is freely
available and well written and covers almost every topic humans have
thought about.

Wikipedia can be downloaded as a single XML file or accessed through the
HuggingFace datasets library. The datasets library handles downloading
and caching so we do not have to manage the raw files ourselves.

```python
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
texts = [item["text"] for item in dataset if item["text"].strip()]
```

The WikiText-103 dataset contains about 29000 Wikipedia articles. They
have been lightly processed to remove markup and metadata. What remains
is clean flowing English text. Exactly what we need.

But we cannot feed raw text to a neural network. Neural networks eat
numbers. We need to convert our text into numbers first.

---

## Part 3: Tokenization . Text Becomes Numbers

The conversion from text to numbers is called tokenization. The
algorithm we use is Byte Pair Encoding. BPE for short. It was invented
in 1994 for data compression and repurposed for language models in 2016.

The idea is simple in concept. Start with every character as its own
token. Find the most common pair of adjacent tokens in the training
data. Merge them into a new token. Repeat until you have 50000 tokens.

Let us see how this works on a tiny example. Imagine our training data
contains only four words with spaces marked as unders.

```
l o w _
l o w e r _
l o w e s t _
l o w e s t _
```

Each letter and underscore is a separate token. We have nine tokens
total. The alphabet is small. The model would need many tokens to
represent even a short sentence. So we merge.

The most common pair is l and o. They appear together four times in the
word low. We create a new token lo. Now our text is shorter.

```
lo w _
lo w e r _
lo w e s t _
lo w e s t _
```

We have ten tokens. We keep merging. The next most common pair is lo and
w. They appear together four times. We create low. Now our text is even
shorter.

```
low _
low e r _
low e s t _
low e s t _
```

We continue. After many rounds of merging our vocabulary contains useful
pieces like low and er and est and the space marker. Now the word lowest
which is not in our original training data can still be represented as
low plus est. Two tokens instead of six characters. Compression and
generalization in one step.

Real BPE tokenizers like GPT-2 use 50000 merges. They start from all
256 possible byte values as the base alphabet. This means they can
tokenize any text in any language that can be represented as bytes which
is all text. The 50000 merges capture the most common patterns across
billions of words. The result is a vocabulary that can represent common
words as single tokens and rare words as sequences of a few tokens and
completely unseen words as sequences of individual byte tokens.

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
text = "The cat sat on the mat."
tokens = tokenizer.encode(text)
print(tokens)  # [464, 3797, 3332, 319, 262, 2603, 13]
```

Seven tokens. Each token is an integer between 0 and 50256. These
integers are the only thing the model ever sees. The raw text is gone.
The model lives in a world of integers.

The tokenizer has one special token that deserves attention. Token
50256 is the end of text marker. It is placed between every document
in the training data. Without it the model would think that the last
sentence of one Wikipedia article flows naturally into the first
sentence of the next. The end of text token is the model's signal that
one thought has ended and a new unrelated thought has begun.

Every token has a unique ID. Token 464 is always The with a capital T.
Token 3797 is always cat. Token 13 is always a period. These mappings
are fixed. They never change during training. The tokenizer is not
part of the neural network. It is a preprocessing step with its own
separate algorithm.

But these integer IDs are just labels. The number 3797 has no
mathematical relationship to the number 2603. The model cannot learn
meaningful patterns from these raw integers. We need to give each
token a richer representation. We need embeddings.

---

## Part 4: Embeddings . Numbers Become Meaning

An embedding is a vector of floating point numbers that captures the
meaning of a token. For our model each token gets a vector of 768
numbers. Token 3797 gets 768 numbers. Token 2603 gets 768 numbers.
Every one of the 50257 tokens gets its own row in a giant lookup table
of shape 50257 by 768.

```python
embedding_table = torch.nn.Embedding(50257, 768)
```

This table is just a matrix. Row 3797 is the embedding for cat. Row 2603
is the embedding for mat. When the model needs the vector for token 3797
it just reads row 3797 from the table. No multiplication. No activation
function. Just a memory lookup.

At initialization every row is filled with random numbers drawn from a
normal distribution with mean 0 and standard deviation 0.02. This means
most values are between -0.04 and 0.04. At this moment cat and dog have
no special relationship. They are just two random rows in a random
table. Every token is equally random.

Training changes this. Over billions of training steps the rows are
updated. Tokens that appear in similar contexts get pushed toward
similar values. Tokens that appear in different contexts get pushed
apart. After training the embedding for cat will be very close to the
embedding for dog. Both will be far from the embedding for democracy.
The space organizes itself into neighborhoods of meaning.

```
cat    = [ 0.34, -0.12,  0.78, -0.56, ...]  (768 numbers)
dog    = [ 0.31, -0.15,  0.81, -0.52, ...]  (very similar to cat)
democracy = [ 0.89, 0.67, -0.23,  0.91, ...]  (completely different)
```

The embedding table is the largest single component in the model. It has
50257 rows times 768 columns equals about 38.6 million numbers. That is
roughly a quarter of all the parameters in our model. Those 38.6
million numbers encode everything the model knows about what words mean.

---

## Part 5: Positional Encoding . The Model Learns Order

The transformer reads all tokens at once. There is no left to right
processing. No step by step recurrence. Every token is processed
simultaneously. This is a strength because it is fast and parallelizable.
But it is also a problem because the model has no way to know which
token came first and which came last.

Consider two sentences. The dog bit the man. The man bit the dog. Same
words. Different order. Completely different meaning. If the model
treated every token independently it could not distinguish these
sentences. The meaning would be scrambled.

We need to stamp each token with its position. Tell the model where in
the sentence this token sits. The modern way to do this is called Rotary
Position Embeddings. RoPE for short. It was introduced in 2021 and
adopted by LLaMA in 2023. Every major model since uses it.

Instead of ADDING a position number to the embedding RoPE ROTATES the
query and key vectors by an angle that depends on the position. The
rotation preserves the vector's magnitude so it does not change the
meaning of the word. But the rotation changes the direction so the
attention dot product between two words becomes a function of their
distance apart.

```
Word at position 1: rotated by angle θ₁
Word at position 4: rotated by angle θ₄

Attention score between them:
Q₁ · K₄ = original_dot × cos(θ₁ - θ₄) + cross_term × sin(θ₁ - θ₄)

The result depends on (θ₁ - θ₄) which is a function of (4 - 1) = 3
steps apart. Not on positions 1 and 4 themselves. Only on the distance.
```

This is the key insight. RoPE makes attention depend on relative
position. Words three steps apart always get the same rotational
relationship regardless of whether they appear at positions 0 and 3 or
positions 497 and 500. The transformer cares about how far apart two
words are not about where they sit in absolute terms.

The angles are precomputed and stored. Each pair of dimensions rotates
at a different speed. The first pair of dimensions rotates fastest and
captures local word order. The last pair rotates slowest and captures
long range position. This multi scale approach means the model has both
fine grained local position information and coarse grained global
position information.

```python
# Precompute rotation angles for every position
dim_indices = torch.arange(0, d_model, 2).float()
inv_freq = 1.0 / (10000.0 ** (dim_indices / d_model))
positions = torch.arange(max_seq_len).float()
freqs = torch.outer(positions, inv_freq)
emb = freqs.repeat_interleave(2, dim=-1)  # [max_seq_len, d_model]
cos_cached = emb.cos()
sin_cached = emb.sin()
```

During the forward pass we look up the precomputed cosine and sine
values for each position and apply the rotation. The rotation formula
for each pair of dimensions (x₀, x₁) at position p is:

```
x₀' = x₀ × cos(θ_p) - x₁ × sin(θ_p)
x₁' = x₀ × sin(θ_p) + x₁ × cos(θ_p)
```

This is a standard 2D rotation. Applied to every pair of dimensions in
the query and key vectors. The values are not rotated because position
information is only needed for deciding WHICH tokens to attend to not
for the content of the tokens themselves.

---

## Part 6: Attention . The Core Mechanism

Attention is the heart of the transformer. Everything else is support
infrastructure. The embedding layer feeds attention. The feed forward
network refines attention's output. The normalization layers keep
attention stable. But attention is where the model actually understands
relationships between words.

### The intuition

Imagine you are reading a long sentence. Some words are more important
than others for understanding what is happening. If the sentence is The
cat that had been sitting on the mat for three hours finally stretched
and yawned you need to connect stretched and yawned with cat across
twelve intervening words. Your brain does this automatically. Attention
does it mathematically.

For every word in the sentence the model creates three vectors. A Query
vector that asks what am I looking for. A Key vector that says what do
I have to offer. A Value vector that holds my actual content. Every word
compares its Query against every other word's Key. Words with high match
scores get more attention. Their Values are weighted more heavily in the
output.

### The computation step by step

Let us trace through attention for a concrete sentence. Our sentence is
The cat sat on the mat. Seven tokens. We will look at one attention head
with a head dimension of 64.

The input to attention is a matrix of shape 7 by 768. Seven tokens each
represented by 768 numbers. We project this matrix into three new
matrices of shape 7 by 64. One for Query. One for Key. One for Value.

```
Q = input @ W_q  (7 × 768 @ 768 × 64 = 7 × 64)
K = input @ W_k  (7 × 768 @ 768 × 64 = 7 × 64)
V = input @ W_v  (7 × 768 @ 768 × 64 = 7 × 64)
```

The weight matrices W_q W_k and W_v are learned during training. They
are what make each attention head different. Different heads learn
different projections that capture different linguistic patterns.

Next we apply RoPE to the Query and Key vectors. This stamps each
query and key with its position information.

```
Q = RoPE(Q, seq_len=7)
K = RoPE(K, seq_len=7)
```

Now we compute the attention scores. The score between token i and
token j is the dot product of Query i with Key j.

```
scores = Q @ K^T / sqrt(64)
```

The result is a 7 by 7 matrix. Each row is a token acting as query. Each
column is a token acting as key. The value at row i column j is how much
token i wants to attend to token j.

```
scores matrix (before mask):

         The    cat    sat    on     the    mat    .
The      0.42   0.15   0.08  -0.03  -0.11   0.02  -0.18
cat      0.38   0.52   0.22   0.01  -0.05   0.09  -0.21
sat      0.21   0.78   0.15   0.31   0.22   0.28   0.05
on      -0.05   0.11   0.45   0.55   0.38   0.42   0.12
the     -0.09   0.03   0.21   0.48   0.61   0.35   0.08
mat     -0.12  -0.01   0.15   0.41   0.52   0.58   0.11
.       -0.22  -0.15  -0.08   0.12   0.18   0.22   0.48
```

Look at the row for sat (row index 2). It has a high score for cat
(0.78) and moderate scores for on (0.31) and mat (0.28). Sat wants to
pay attention to its subject and its prepositional phrase. It cares
less about itself (0.15) and the period (0.05). This pattern emerged
from the learned weights W_q and W_k and from the positional rotation.

Now we apply the causal mask. Tokens cannot see the future. The upper
right triangle of the matrix is set to negative infinity.

```
scores matrix (after mask):

         The    cat    sat    on     the    mat    .
The      0.42  -inf   -inf   -inf   -inf   -inf   -inf
cat      0.38   0.52  -inf   -inf   -inf   -inf   -inf
sat      0.21   0.78   0.15  -inf   -inf   -inf   -inf
on      -0.05   0.11   0.45   0.55  -inf   -inf   -inf
the     -0.09   0.03   0.21   0.48   0.61  -inf   -inf
mat     -0.12  -0.01   0.15   0.41   0.52   0.58  -inf
.       -0.22  -0.15  -0.08   0.12   0.18   0.22   0.48
```

After softmax negative infinity becomes zero. The scores become
attention weights that sum to one for each row.

```
attention weights matrix (after softmax):

         The    cat    sat    on     the    mat    .
The      1.00   0.00   0.00   0.00   0.00   0.00   0.00
cat      0.47   0.53   0.00   0.00   0.00   0.00   0.00
sat      0.18   0.35   0.10   0.00   0.00   0.00   0.00
on       0.08   0.10   0.22   0.25   0.20   0.15   0.00
the      0.05   0.06   0.10   0.19   0.28   0.17   0.00
mat      0.04   0.05   0.08   0.17   0.24   0.30   0.00
.        0.03   0.04   0.05   0.08   0.11   0.14   0.28
```

Look at the row for on (row index 3). It attends 25 percent to itself
and 22 percent to sat and 20 percent to the and 15 percent to mat. A
balanced distribution across the preceding tokens. The word on is a
preposition that connects everything around it. It needs context from
every nearby word.

Look at the row for The (row index 0). It attends 100 percent to itself.
There is nothing before it. The word The has no context. It must rely
entirely on its own meaning. This is always true for the first token in
every sequence.

Finally we use these weights to mix the Value vectors.

```
output = attention_weights @ V

For token sat (row 2):
new_sat = 0.18 × V_The + 0.35 × V_cat + 0.10 × V_sat
```

The new vector for sat now contains information from The and cat
weighted by how much sat cares about them. The original meaning of sat
is still there via self attention (10 percent) but it has been enriched
with context from the subject of the sentence.

This entire computation happens 12 times in parallel for 12 heads. Each
head has its own W_q W_k and W_v matrices. Each head learns different
attention patterns. After all heads have computed their outputs we
concatenate them back together into a single 768 dimensional vector and
project through a final linear layer.

```
all_heads = torch.cat([head_0, head_1, ..., head_11], dim=-1)  # 12 × 64 = 768
output = all_heads @ W_o  # 768 @ 768 = 768
```

The output projection W_o mixes information between heads. Each head
operated independently. Now they share their discoveries. The grammar
head tells the pronoun resolution head what it found. The position head
tells the semantic head about word distances. The mixed output is richer
than any single head's contribution.

---

## Part 7: RMSNorm . Keeping Numbers Under Control

Before attention and before the feed forward network we normalize the
input. Normalization keeps the numbers at a consistent scale as they
flow through dozens of layers.

We use RMSNorm. It is simpler and faster than the older LayerNorm. It
computes the root mean square of a vector and divides every element by
it. The result always has RMS equal to 1.0.

```
rms = sqrt(mean(x²))
output = x / rms × weight
```

The weight is a learned parameter. One weight per dimension. It starts
at 1.0 and learns during training. It lets the model amplify important
dimensions and suppress unimportant ones while keeping the overall
magnitude stable.

Without normalization the outputs of attention and feed forward layers
would grow without bound. After twelve layers some values might be a
thousand times larger than others. The softmax in the next attention
layer would become a one hot vector. Gradients would vanish. Training
would fail.

With normalization every layer gets clean well scaled inputs. The tower
of twelve blocks stays straight. The model trains smoothly.

---

## Part 8: SwiGLU . The Gated Feed Forward Network

After attention every token has mixed information from all other tokens.
But the mixing was linear. Attention is just a weighted sum. Weighted
sums are not enough to capture the complexity of language. We need non
linear processing.

The feed forward network provides this non linearity. It processes each
token independently with the same learned weights. Each token gets the
same transformation applied to its unique vector.

Our feed forward network uses SwiGLU. SwiGLU is a gated activation. It
splits the computation into two paths. One path produces values. The
other path produces gates. The gates control how much of each value
passes through.

```
h = input @ W₁  (768 → 3072)  # value path
g = input @ W₂  (768 → 3072)  # gate path
output = (SiLU(h) × g) @ W₃  (3072 → 768)  # combine and project
```

The expansion from 768 to 3072 gives the network room to transform
information. In the wider middle layer the network can represent more
complex patterns. The contraction back to 768 forces it to compress
those patterns into a dense representation.

The SiLU activation on the value path provides smooth non linearity.
Unlike ReLU which has a sharp corner at zero SiLU is smooth everywhere.
This makes gradients flow better during training. The gate path has no
activation. It can output any real number. A gate of zero blocks the
information. A gate of one passes it through unchanged. A gate of two
amplifies it. The model learns which inputs should be amplified and
which should be suppressed.

The gate learns context dependent filtering. When the token is a verb
the gate might amplify dimensions related to action and suppress
dimensions related to objects. When the token is a noun it might do the
opposite. The same network weights apply to every token but the behavior
differs because each token's vector leads to different gate values.

SwiGLU has three weight matrices instead of the two that a standard
feed forward network would have. The extra matrix is for the gate. This
adds about 28 million parameters to our model compared to a standard
FFN. Every one of those parameters contributes to better performance.
The gating mechanism is why SwiGLU outperforms ReLU and GELU at scale.

---

## Part 9: The Residual Connection . The Gradient Highway

Every sublayer has a residual connection. The attention output is added
to the attention input. The feed forward output is added to the feed
forward input.

```
x = x + attention(norm(x))
x = x + ffn(norm(x))
```

These plus signs are the most important operators in the entire model.
Without them deep transformers cannot be trained. The gradients would
vanish. The early layers would never learn.

Here is why. When the model makes a prediction and computes the loss it
sends a gradient backward through the network. This gradient tells each
weight how to change to reduce the loss. The gradient flows backward
through each layer in reverse order. At each layer it is multiplied by
the derivative of that layer's function. If the derivative is smaller
than one the gradient shrinks. After propagating backward through twelve
layers the gradient at the first layer is the product of eleven numbers
that are each less than one.

```
gradient_at_layer_1 = gradient_at_layer_12 × d₁ × d₂ × ... × d₁₁

If each derivative is 0.5:
gradient_at_layer_1 = gradient_at_layer_12 × 0.5¹¹
                    = gradient_at_layer_12 × 0.0005
```

The gradient at layer one is two thousand times smaller than the
gradient at layer twelve. The first layer receives almost no learning
signal. Its weights stay random. The model cannot train.

Residual connections fix this by providing a second path. The gradient
can flow backward through the sublayer like before. Or it can bypass the
sublayer entirely and flow straight to the input. The bypass path has a
derivative of exactly 1.0. Always. The gradient does not shrink.

```
With residual: output = input + sublayer(norm(input))
Derivative:    d(output)/d(input) = 1 + d(sublayer)/d(input)
```

The total derivative is 1 plus something. Even if the something is small
the 1 ensures the gradient never vanishes. After twelve layers the
gradient at layer one is at least as large as the gradient at layer
twelve. Every layer can learn.

This is why we can stack twelve blocks. Or twenty four. Or ninety six.
The gradient highway stays open regardless of depth. The only limit is
computational cost not trainability.

---

## Part 10: The Full Model . Putting It All Together

Let us assemble every piece into the complete model.

```python
class GPT(nn.Module):
    def __init__(self, config):
        self.token_embedding = nn.Embedding(vocab_size, d_model)  # Part 4
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads)  # Parts 6-9
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)  # Part 7
        self.lm_head = nn.Linear(d_model, vocab_size)  # Output projection

    def forward(self, input_ids):
        # Part 4: Embed tokens
        x = self.token_embedding(input_ids)  # [batch, seq, 768]

        # Parts 6-9: Process through transformer blocks
        for layer in self.layers:
            x = layer(x)  # Each block contains attention + FFN + residuals

        # Part 7: Final normalization
        x = self.final_norm(x)

        # Output: Project to vocabulary
        logits = self.lm_head(x)  # [batch, seq, 50257]
        return logits
```

That is the entire model. About fifty lines of code. Every component we
discussed is inside those fifty lines. The embedding table from Part 4.
The stacked transformer blocks from Parts 6 through 9. The final
normalization from Part 7. The output projection that converts hidden
states back to vocabulary predictions.

The model takes a batch of token sequences as input. For each position
in each sequence it produces 50257 scores. One score for each possible
next token. The highest scoring token is the model's prediction for what
word comes next.

---

## Part 11: The Output . From Vectors to Words

The final layer of the model projects from 768 dimensions to 50257
dimensions. This is a simple linear transformation. Multiply by a weight
matrix of shape 768 by 50257.

```python
logits = x @ W_lm_head  # [batch, seq, 768] @ [768, 50257] = [batch, seq, 50257]
```

These 50257 numbers are called logits. They are unnormalized scores.
Higher means the model thinks that token is more likely. They are not
probabilities yet because they do not sum to one and some may be
negative.

To convert logits to probabilities we apply softmax.

```python
probs = softmax(logits)  # Each row now sums to 1.0
```

Each row of the probability matrix sums to one. Row i column j is the
model's estimated probability that token j comes next given the first
i plus 1 tokens of the input.

The model does not output just one token. It outputs a probability
distribution over all 50257 tokens. During training we compare this
distribution to the actual next token. During generation we sample from
this distribution to pick the next word.

---

## Part 12: The Loss . Measuring Wrongness

Training needs a number that tells us how good the model's predictions
are. Lower is better. The number is called the loss.

We use cross entropy loss. It measures the difference between the
model's predicted probabilities and the actual next tokens.

For a single prediction where the true next token is j:

```
loss = -log(probs[j])
```

If the model assigns probability 0.9 to the correct token the loss is
negative log of 0.9 which is 0.105. Good. The model was confident and
right.

If the model assigns probability 0.1 to the correct token the loss is
negative log of 0.1 which is 2.303. Bad. The model was confident about
the wrong things.

If the model assigns probability 0.01 to the correct token the loss is
negative log of 0.01 which is 4.605. Terrible. The model barely
considered the correct answer.

The loss is always positive. It approaches zero as the model becomes
perfect. It approaches infinity as the model becomes completely wrong.
A random model that assigns equal probability to all 50257 tokens would
have a loss of negative log of one over 50257 which is about 10.8. This
is the baseline. Any loss above 10.8 means the model is worse than
random. Any loss below 10.8 means the model has learned something.

```python
def compute_loss(logits, targets):
    # logits:   [batch, seq, 50257]
    # targets:  [batch, seq]  (shifted by 1 from input)
    logits_flat = logits.view(-1, 50257)
    targets_flat = targets.view(-1)
    return F.cross_entropy(logits_flat, targets_flat)
```

We compute the loss over all positions in all sequences in the batch.
The average loss across millions of predictions gives us a single number
that measures the model's performance. Every training step we try to
make this number smaller.

---

## Part 13: Backpropagation . Figuring Out What to Change

We have a loss. The loss tells us how wrong the model was. But it does
not tell us which of the 150 million weights to change or in which
direction. Backpropagation answers this question.

Backpropagation applies the chain rule from calculus. For every weight
in the model it computes the partial derivative of the loss with respect
to that weight. This derivative tells us: if I increase this weight by
a tiny amount how much will the loss change.

```python
loss.backward()  # PyTorch does all the calculus automatically
```

After this call every weight in the model has a .grad attribute. The
gradient is a tensor of the same shape as the weight. Each element in
the gradient is the direction and magnitude to change that weight to
reduce the loss.

```
If weight[i,j].grad = 0.003:
  Increasing weight[i,j] makes the loss go up.
  We should decrease it.

If weight[i,j].grad = -0.005:
  Increasing weight[i,j] makes the loss go down.
  We should increase it.

If weight[i,j].grad = 0.000:
  Changing this weight does not affect the loss.
  We can leave it alone or change it without consequence.
```

The gradients flow backward from the loss through the output projection
through the final normalization through each transformer block in
reverse order through the embedding table and back to the input. At
each step the chain rule multiplies local derivatives. The residual
connections ensure that gradients survive the journey.

---

## Part 14: Gradient Clipping . Preventing Wild Jumps

Sometimes a batch of text produces very large gradients. A rare word
pattern or an unusual sentence structure sends a shockwave through the
gradients. If we applied these large gradients directly the model's
weights would jump to a completely different configuration. Training
would be destroyed.

Gradient clipping prevents this. After the backward pass we check the
total magnitude of all gradients. If it exceeds a threshold we shrink
all gradients proportionally to fit under the threshold.

```python
total_norm = sqrt(sum(g.norm(2)² for g in gradients))
if total_norm > 1.0:
    scale = 1.0 / total_norm
    for g in gradients:
        g *= scale
```

The direction of the update is preserved. Only the step size is limited.
The model takes small safe steps instead of wild leaps. The threshold
of 1.0 is standard for transformer training. It was found empirically.
It catches dangerous spikes without interfering with normal updates.

---

## Part 15: AdamW . Updating the Weights

We have gradients for every weight. Now we need to apply them. The
simplest approach is to move each weight a tiny bit in the opposite
direction of its gradient.

```
weight = weight - learning_rate × gradient
```

This is stochastic gradient descent. It works but it is slow and
unstable. The learning rate is the same for every weight regardless of
how much each weight needs to change. Noisy gradients cause zigzagging.
Large weights receive no regularization.

AdamW improves on all three fronts. It maintains running averages of
past gradients and their magnitudes. It uses these averages to adjust
the step size for each weight independently. It applies weight decay
separately from the gradient update.

```python
# AdamW for a single weight
momentum = β₁ × momentum + (1 - β₁) × gradient      # Running average of gradients
velocity = β₂ × velocity + (1 - β₂) × gradient²      # Running average of squared gradients

# Bias correction for early steps
mom_corrected = momentum / (1 - β₁^step)
vel_corrected = velocity / (1 - β₂^step)

# Decoupled weight decay
weight = weight × (1 - lr × weight_decay)

# Gradient update
weight = weight - lr × mom_corrected / (sqrt(vel_corrected) + ε)
```

The momentum term acts like inertia. It smooths out noise by
maintaining a running average of past gradients. If the gradient
points in the same direction for many steps momentum builds up and the
step size increases. If the gradient oscillates momentum cancels out
and the step size decreases.

The velocity term adjusts per weight learning rates. Weights that have
been making large moves get smaller steps. Weights that have been still
get larger steps. This adaptive behavior means we do not need to tune
the learning rate for every weight individually.

The weight decay term pushes all weights toward zero by a tiny fraction
each step. This prevents weights from growing without bound. Large
weights are a sign of overfitting. The model has become too confident
about a few patterns and ignores everything else. Weight decay forces it
to stay humble.

The epsilon term prevents division by zero. It is tiny and never needs
tuning.

AdamW is the standard optimizer for language model training. GPT-3
trained with it. LLaMA trained with it. Every model in this guide
trains with it. The specific hyperparameters β₁ of 0.9 β₂ of 0.95 and
weight decay of 0.1 are the LLaMA defaults. They have been validated
on models from one billion to seventy billion parameters.

---

## Part 16: Cosine Warmup . The Learning Rate Schedule

The learning rate is not constant throughout training. It follows a
schedule that warms up then decays.

At the very start of training the model's weights are random. The
gradients are large and noisy. A large learning rate would send the
model flying off in random directions. We start with a learning rate
of zero and linearly increase it to the maximum over several thousand
steps. This is the warmup phase.

```python
if step < warmup_steps:
    lr = max_lr × step / warmup_steps
```

Once the model is stable we can train at full speed. But as training
progresses and the model gets closer to a good solution we need to be
more careful. Large steps would overshoot the minimum. We gradually
reduce the learning rate following a cosine curve.

```python
progress = (step - warmup_steps) / (total_steps - warmup_steps)
lr = min_lr + (max_lr - min_lr) × 0.5 × (1 + cos(π × progress))
```

The cosine curve starts falling slowly then faster in the middle then
slowly again at the end. This smooth decay is gentler than step decay
which drops the learning rate abruptly at fixed intervals. Abrupt drops
can disturb the model. Cosine decay is continuous.

At the very end of training the learning rate reaches a small minimum.
The model takes tiny steps that refine its weights with precision. The
trusty phase.

All three phases together make training both stable at the start and
precise at the end. Every modern language model uses this schedule.

---

## Part 17: Mixed Precision . Faster Training

The model's weights are stored as 32 bit floating point numbers. This
is the standard for scientific computing. Good precision and good range.

But most operations inside the forward pass do not need 32 bits of
precision. The matrix multiplications in attention and the feed forward
network work almost as well with 16 bits. Using 16 bits instead of 32
cuts memory usage in half and nearly doubles speed on modern GPUs.

We use a format called bfloat16. It has the same range as float32 but
less precision. The maximum representable number is the same in both
formats. So bfloat16 never overflows even during the largest matrix
multiplications. The only difference is that bfloat16 can only represent
about two decimal digits of precision instead of seven.

This tradeoff is perfect for neural networks. We need the range to
prevent overflow during intermediate computations. But we do not need
seven digits of precision for every activation. Two digits is enough
for the model to learn effectively.

```python
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    # Every operation here uses bfloat16 where safe
    logits = model(input_ids)
    loss = compute_loss(logits, targets)
```

The master weights are always stored in float32. Only the forward and
backward passes use bfloat16. The weight updates are applied in float32
to preserve precision over thousands of training steps.

Some operations stay in float32 because they need more precision.
Normalization layers need full precision to keep activations properly
scaled. The softmax in attention needs full precision for numerical
stability. Autocast handles these exceptions automatically. We do not
need to specify which operations to convert.

---

## Part 18: The Training Loop . Putting It All Together

We have every piece. The model. The data. The tokenizer. The optimizer.
The scheduler. The loss function. Now we assemble them into a training
loop.

```python
for step in range(max_steps):
    # 1. Get a batch of text
    batch = next(dataloader)
    input_ids, target_ids = batch

    # 2. Forward pass
    with autocast(use_amp):
        logits = model(input_ids)
        loss = cross_entropy(logits, target_ids)

    # 3. Backward pass
    loss.backward()

    # 4. Clip gradients
    clip_grad_norm(model.parameters(), max_norm=1.0)

    # 5. Update weights
    optimizer.step()
    optimizer.zero_grad()

    # 6. Update learning rate
    scheduler.step()

    # 7. Log progress
    if step % 100 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")
```

Seven steps. Repeated thousands or millions of times. Each repetition
the loss gets slightly smaller. The model gets slightly better. After
enough repetitions the model can generate coherent text.

The first few hundred steps are chaotic. The loss bounces around. The
gradients are large. The model is searching. Around step one thousand
the loss starts a steady decline. The model has found a good direction.
From then on progress is slow but consistent. Each step shaves a tiny
fraction off the loss. After fifty thousand steps the loss has dropped
from around 10.8 to somewhere between 2 and 3. The model can write
sentences that are sometimes grammatical and sometimes nonsensical. It
knows that periods end sentences and that capital letters start them.
It knows that the is often followed by a noun. It knows that cat and dog
can both sit and run and sleep.

After five hundred thousand steps the model writes paragraphs that are
mostly coherent. It still makes mistakes. It invents facts. It repeats
itself. But it has captured a remarkable amount of the structure of
English. All from predicting the next word billions of times.

---

## Part 19: Text Generation . The Model Speaks

Once the model is trained we want it to write something. We give it a
starting phrase called a prompt. The model reads the prompt and predicts
the first word after it. Then it takes the prompt plus that predicted
word and predicts the second word. It repeats until it has generated
enough text or until it predicts an end of text token.

```python
prompt = "The cat sat on the"
input_ids = tokenizer.encode(prompt)  # [464, 3797, 3332, 319, 262]

for _ in range(50):
    logits = model(input_ids)          # Predict next token
    logits = logits[:, -1, :]          # Only the last position

    probs = softmax(logits / temperature)
    next_token = sample(probs, top_k=50)

    input_ids = append(input_ids, next_token)
```

The sampling parameters control how the model picks the next token.
Without any parameters the model would always pick the single most
likely token. The output would be deterministic and often repetitive.
The same prompt would always produce the same completion. The model
would loop on common phrases.

Temperature adds randomness. It divides the logits by a number before
softmax. Low temperature makes the distribution sharper. The top token
gets even more probability. The output is focused and predictable. High
temperature flattens the distribution. Less likely tokens get more
chance. The output is creative and unpredictable.

```
Temperature 0.3: "The cat sat on the windowsill gazing at the birds outside."
Temperature 0.8: "The cat sat on the edge of the couch watching me with sleepy eyes."
Temperature 1.5: "The cat sat on the piano keys and composed a midnight melody."
```

Top-k limits the choices to the k most likely tokens. Everything else
gets zero probability. This prevents the model from ever picking a
completely nonsensical token. A value of 50 is common. It eliminates the
bottom 50207 tokens while keeping enough variety for interesting output.

Top-p is an adaptive version of top-k. Instead of always keeping k
tokens it keeps the smallest set of tokens whose cumulative probability
exceeds p. If the model is very confident it might keep only three
tokens. If the model is uncertain it might keep five hundred. This
adapts to the model's confidence at each step.

Together these three parameters give us fine control over the model's
output. They are the reason the same model can write both technical
documentation and poetry. The model provides the probabilities. The
parameters control how we sample from them.

---

## Part 20: KV Cache . Making Generation Fast

The naive generation loop is slow. Every time we append a new token we
recompute the entire sequence from scratch. Token 500 has already been
processed 499 times by the time we add token 501. Most of the
computation is redundant. The Key and Value vectors for the first 500
tokens do not change when we add token 501.

The KV cache eliminates this redundancy. We store the Key and Value
vectors for every token we have already processed. When a new token
arrives we compute its Key and Value and append them to the cache. We
do not recompute anything for the old tokens.

```
Without cache:  Step 1 computes K and V for 1 token.
                Step 2 computes K and V for 2 tokens.
                Step 3 computes K and V for 3 tokens.
                Total work: 1 + 2 + 3 + ... + N ≈ N²/2

With cache:     Step 1 computes K and V for 1 token.
                Step 2 computes K and V for 1 new token. Reuses old.
                Step 3 computes K and V for 1 new token. Reuses old.
                Total work: N
```

For a thousand token generation the KV cache is roughly a thousand times
faster. The memory cost is manageable for small models. For GPT-2 Small
the cache for a thousand tokens is about 35 megabytes. For GPT-3 Large
it would be about 4 gigabytes. For very large models at very long
context lengths the cache can become the dominant memory consumer.

---

## Part 21: What the Model Actually Learned

After training on billions of words the model has learned patterns that
are invisible to the untrained eye. It has not learned facts in the way
a database stores facts. It has learned statistical regularities. The
word sequence the cat sat on the is almost always followed by mat or
floor or chair or bed. The sequence the capital of France is almost
always followed by Paris. The model does not know what France or Paris
or capital mean. It only knows the probability distribution over next
words given all previous words.

The embeddings have organized themselves into a space with structure.
The vector for king minus the vector for man plus the vector for woman
is very close to the vector for queen. This was not programmed. It
emerged from training data where king and queen appeared in similar
contexts but with different gendered pronouns.

The attention heads have specialized. Some heads consistently attend to
the subject of the current verb. Others attend to recent nouns mentioned
in the sentence. Others attend to punctuation to understand sentence
boundaries. These specializations were not designed. They emerged from
the training objective of predicting the next word.

The feed forward networks have become pattern recognizers. One part of
the network might activate strongly when it sees a list of items because
commas between items predict more items. Another part might activate for
dates because the word in followed by a year predicts a specific
temporal pattern. These patterns are distributed across thousands of
neurons in ways that are difficult to interpret but mathematically
optimal for prediction.

---

## Part 22: Why This Matters

A machine that can predict the next word with high accuracy is a machine
that has implicitly learned the rules of language. Grammar. Syntax.
Semantics. Discourse structure. World knowledge. All of it is necessary
to make accurate predictions. The model must know that verbs agree with
their subjects in number. It must know that Paris is in France and that
France is in Europe. It must know that a sentence that starts with
although expects a contrasting clause. It must know that a recipe for
cake includes flour and sugar and eggs not motor oil and concrete.

The model acquires all this knowledge through a single task: predict the
next token. It is a simple task with profound implications. A system
that can predict what humans will write next is a system that has
compressed a significant fraction of human knowledge into a set of
matrix multiplications.

The transformer architecture made this possible. Before transformers
language models could only capture local patterns within a few words.
Recurrent networks forgot information that appeared more than a few
dozen words back. Attention changed that. Attention lets every word
interact with every other word regardless of distance. A word at the
end of a paragraph can attend to a word at the beginning as easily as
to the word right next to it.

The scale made this powerful. GPT-2 with 1.5 billion parameters could
write plausible paragraphs. GPT-3 with 175 billion parameters could
write plausible essays and answer questions and generate code. The jump
in capability came entirely from more data and more parameters. The
architecture stayed almost the same.

The latest generation of models adds instruction following. They are
trained not just to predict the next word but to predict the next word
in a helpful and harmless assistant's response. This additional training
makes the models useful as tools rather than just interesting as
demonstrations.

But underneath the chat interface and the instruction tuning and the
safety filters the core mechanism is unchanged. Tokens in. Attention
across. Feed forward through. Logits out. The same story we have traced
from beginning to end. The same mathematics. The same architecture. The
same gradient descent optimizing cross entropy loss one step at a time.

---

## Epilogue: What You Can Build Next

You have now seen every piece of a modern language model. You could
build one from scratch with the code in this guide. You could modify it.
Add more layers. Use a bigger dataset. Experiment with different
attention patterns. Swap SwiGLU for a different activation.

The architecture described here is not the final word. Research
continues. State space models like Mamba challenge the transformer's
dominance. Mixture of experts routes tokens through different sub
networks to scale more efficiently. Retrieval augmented generation
connects models to external knowledge bases. But the core ideas are
stable. Embeddings. Attention. Residuals. Normalization. Gradient
descent. These will be relevant for as long as neural networks exist.

You now understand them. Not just what they are. Why they are. Every
design choice in this architecture was made to solve a specific problem.
The residual connections solve vanishing gradients. RMSNorm solves
activation drift. SwiGLU solves the inflexibility of simple activation
functions. RoPE solves position encoding without parameters. Every
piece tells a story.

The story of modern AI is the story of many people over many years
solving one problem at a time and stacking their solutions into
something greater than the sum of its parts. You now know every part.
You can be one of those people.

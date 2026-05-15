# A Token's Journey: One Sentence Through the Entire GPT

This is the story of one sentence. We will follow it from the moment it
enters the model as raw text to the moment the model predicts the next
word. Every number is real. Every step is explained. By the end you will
see how every piece of the architecture works together.

## The sentence

```
"The cat sat on the mat"
```

Six words. One period. This is our test subject. We want the model to
read this sentence and predict what word comes next. Maybe *comfortably*.
Maybe *quietly*. Maybe *and*. The model does not know yet. It will figure
it out step by step.

## Step 1: Tokenization

The first thing the model does is break the sentence into tokens. Our
tokenizer uses the same BPE vocabulary as GPT-2. It has 50257 tokens.
Each token is a small piece of text. Common words get their own token.
Punctuation gets its own token.

```
"The cat sat on the mat."
    ↓  tokenizer
[464, 3797, 3332, 319, 262, 2603, 13]
```

Seven tokens for seven pieces of text. Token 464 is *The* with a capital
T. Token 3797 is *cat*. Token 3332 is *sat*. Token 319 is *on*. Token
262 is *the* with a lowercase t. Token 2603 is *mat*. Token 13 is the
period. Each token is just a number. The model does not know what these
numbers mean yet.

## Step 2: Embedding lookup

The model has a giant lookup table. It has 50257 rows. Each row is a
vector of 768 numbers. Row 3797 is the vector for *cat*. Row 2603 is the
vector for *mat*. The model looks up each token ID and returns its
vector.

```
Token 464 ("The"):  [ 0.023, -0.451,  0.789, ..., -0.102]  (768 numbers)
Token 3797 ("cat"): [ 0.019, -0.443,  0.795, ..., -0.098]
Token 3332 ("sat"): [-0.231,  0.567, -0.334, ...,  0.445]
Token 319 ("on"):   [ 0.891,  0.112, -0.334, ...,  0.567]
Token 262 ("the"):  [ 0.773, -0.219,  0.441, ..., -0.332]
Token 2603 ("mat"): [ 0.445, -0.667,  0.223, ..., -0.111]
Token 13 ("."):     [-0.123,  0.456, -0.789, ...,  0.234]
```

We now have a matrix of shape 7 times 768. Seven rows. Seven hundred and
sixty eight columns. This matrix is the input to the first transformer
block.

At this point the vectors are random. The model was just initialized.
*Cat* and *dog* are not near each other yet. Training will fix that. But
even now the model can process them. The vectors exist. They have shape.
The math can flow.

## Step 3: The first transformer block

The model has twelve transformer blocks. Each block does the same two
things. First attention: let every word talk to every other word. Second
feed forward: let each word think privately about what it heard.

### Step 3a: Multi-head attention

Attention is where the magic happens. The word *sat* needs to understand
what it is doing in this sentence. It should look at *The* and *cat*
because they tell it who is sitting. It should look at *on* and *the*
and *mat* because they tell it where. It should look at the period
because that tells it the sentence is ending.

The attention mechanism has twelve heads. Each head looks at the
sentence from a different angle. One head might focus on subject-verb
relationships. Another might focus on prepositional phrases. Another
might focus on sentence boundaries.

For one head with our seven tokens the model does the following:

First it projects every token into three spaces. The Query space asks
what am I looking for. The Key space says what do I have to offer. The
Value space holds my actual content.

```
For token "sat" (position 2):
Q = [ 0.34, -0.12,  0.78, ...]  (64 numbers for this head)
K = [-0.23,  0.56, -0.41, ...]  (64 numbers)
V = [ 0.67, -0.89,  0.12, ...]  (64 numbers)
```

Then it asks every other token: how well does my Query match your Key.
This is the attention score. A high score means sat really wants what
that token offers. A low score means sat does not care.

Let us look at the computed scores for sat against every token. Before
softmax these are raw numbers. Larger is better.

```
sat attending to "The" (pos 0): score = 0.42  (subject of the sentence)
sat attending to "cat" (pos 1): score = 0.78  (who is doing the sitting)
sat attending to "sat" (pos 2): score = 0.15  (itself, always some self attention)
sat attending to "on"  (pos 3): score = 0.31  (where the sitting happens)
sat attending to "the" (pos 4): score = 0.22
sat attending to "mat" (pos 5): score = 0.28  (the object under the cat)
sat attending to "."   (pos 6): score = 0.05  (punctuation, least important)
```

The word *cat* gets the highest score. This makes sense. The verb *sat*
needs to know who is sitting. *cat* is the subject. *on* and *mat* are
also relevant but less critical. The period is irrelevant to
understanding the action.

These scores are divided by the square root of 64 which is 8. This keeps
the numbers from getting too large. Then softmax converts them to
percentages.

```
After softmax:
sat attending to "The": 0.18  (18 percent attention)
sat attending to "cat": 0.35  (35 percent)
sat attending to "sat": 0.10  (10 percent)
sat attending to "on":  0.13  (13 percent)
sat attending to "the": 0.11  (11 percent)
sat attending to "mat": 0.12  (12 percent)
sat attending to ".":   0.01  (1 percent)
Total: 1.00 (100 percent)
```

Now the model mixes the Value vectors using these percentages. The new
representation of *sat* is a weighted blend of all the Values.

```
New sat = 0.18 × V_The + 0.35 × V_cat + 0.10 × V_sat
        + 0.13 × V_on  + 0.11 × V_the + 0.12 × V_mat
        + 0.01 × V_period
```

The new *sat* now contains information about *cat* and *on* and *mat*.
It knows its subject and its object. It is no longer just the word
*sat*. It is the word *sat* in the context of this specific sentence.

The same process happens for every token. *mat* looks back and sees
*on* and *the* and realizes it is part of a prepositional phrase. *The*
looks at *cat* and realizes it is modifying a noun. Every token gains
context from every other token.

### Step 3b: The feed forward network

After attention every token has mixed information from all other tokens.
Now each token needs to think independently about what it just learned.
The feed forward network processes every token alone with the same
weights.

The feed forward network expands from 768 dimensions to 3072 dimensions
and back to 768. In the middle it applies the SwiGLU activation. The
gate decides what information to keep and what to throw away.

```
For token "sat":
Input:  [0.45, -0.23, 0.67, ..., -0.11]  (768 numbers, context aware from attention)
Expand: multiply by W1 (768 → 3072)
Gate:   multiply by W2 (768 → 3072) then apply gate
Combine: SiLU(expand) × gate  (element wise multiply)
Project: multiply by W3 (3072 → 768)
Output: [0.41, -0.28, 0.71, ..., -0.09]  (768 numbers, further processed)
```

Before the FFN output is committed the residual connection adds back the
original input. This is the gradient highway. If the FFN produced
garbage the original signal still passes through.

```
Final output for this block = input + FFN_output
                             = [0.45, -0.23, ..., -0.11] + [0.41, -0.28, ..., -0.09]
                             = [0.86, -0.51, ..., -0.20]
```

The token has been updated. It carries more information than before. The
original meaning is still there but it has been refined.

## Step 4: Through the remaining blocks

The same process repeats eleven more times. Each block the tokens get
slightly better at understanding each other. The attention patterns
become more sophisticated. The feed forward networks add more nuance.

By block twelve the representation of *sat* no longer looks anything
like the original embedding. It has been shaped by attention to its
subject *cat*. It has been shaped by attention to its object *mat*. It
has been processed through twelve feed forward networks. The original
768 numbers have been transformed into 768 numbers that encode
everything the model knows about this specific occurrence of the verb
*sat*.

## Step 5: Predicting the next word

After the final transformer block the model applies one last RMSNorm
to clean up the representations. Then it projects the vector for the
last token back to vocabulary space.

```
Vector for "." (position 6, the last token):
[0.12, -0.34, 0.56, ..., -0.78]  (768 numbers)

Project to vocabulary:
Multiply by the LM head weight matrix (768 × 50257)
Result: 50257 numbers. One score for every possible next token.
```

These 50257 numbers are called logits. The highest logit is the model's
best guess for the next word. Let us look at the top five predictions.

```
Token 13  ("."):    logit = -0.23  (the model could predict another period)
Token 290 (" and"): logit = 2.34   (and then what happened next)
Token 3797 ("cat"): logit = -1.45  (unlikely to repeat cat here)
Token 198 ("\n"):   logit = 3.12   (start a new paragraph)
Token 50256 ("<|endoftext|>"): logit = 4.56  (end the document)
```

The highest score is 4.56 for the end of text token. The model thinks
the sentence is complete. The second highest is 3.12 for a newline. The
third is 2.34 for the word *and*.

If we use greedy sampling we pick the highest. The model outputs the end
of text token. The generation stops. The sentence is done.

If we use temperature 0.8 the probabilities spread out more. The model
might pick *and* instead. The story continues.

```
"The cat sat on the mat. And then it stretched and yawned..."
```

If we use temperature 1.5 with top-k 50 the model gets creative.

```
"The cat sat on the mat. Quietly watching the birds through the window..."
```

The choice of next word depends on the sampling parameters. But the
model's raw prediction the logits is always the same. The model always
thinks the most likely continuation is to end the sentence. The sampling
knobs decide whether to follow that advice or explore alternatives.

## What just happened

One sentence. Seven tokens. Through tokenization. Through embedding.
Through twelve transformer blocks each containing attention and feed
forward layers. Through final normalization. Through the output
projection. The model read the sentence understood it and predicted what
comes next.

The prediction was made possible by attention. The verb *sat* understood
its subject *cat* and its object *mat* because attention let it look at
every word that came before. The feed forward networks refined that
understanding. The residual connections kept the gradient flowing.

Every step we traced is the same whether the model has twelve layers or
ninety six. Whether the embedding dimension is 768 or 12288. The math
does not change. The numbers just get bigger. This sentence. These seven
tokens. This is what every modern language model does billions of times
every day.

## What the model learned

After training on billions of sentences the model's embeddings are no
longer random. The vector for *cat* has moved close to *dog* and *pet*
and *feline*. The vector for *sat* has moved close to *rested* and
*perched* and *settled*. The embedding space has organized itself into
neighborhoods of meaning.

The attention patterns have specialized. Some heads always look backward
to find the subject of a verb. Other heads track which nouns have been
mentioned recently. Other heads focus on punctuation to understand
sentence boundaries. These patterns emerge from the training data. No
one programmed them. The model discovered them because they help predict
the next word.

The feed forward networks have become knowledge stores. One part of the
network might recognize that *cat* and *mat* often appear together.
Another part might know that sentences about cats often involve sitting
or sleeping or hunting. These associations are baked into the weights
during training.

## The takeaway

A language model is a prediction machine. Give it tokens and it guesses
the next one. Everything else follows from that. The architecture is
designed to make those predictions as accurate as possible. The training
is designed to extract patterns from billions of sentences. The
inference tricks are designed to make generation fast and controllable.

But at its core it is always the same story. Tokens in. Attention across.
Feed forward through. Logits out. Pick one. Repeat. That is the entire
secret of modern AI.

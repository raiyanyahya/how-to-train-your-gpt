# Encoder, Decoder and Encoder-Decoder: The Three Transformer Families

## The short answer

There are three ways to build a transformer. Decoder-only models like
GPT generate text one token at a time. They can only see what came
before. Encoder-only models like BERT look at the entire input at once
in both directions. They understand text but cannot generate it.
Encoder-decoder models like T5 read the full input with an encoder and
generate output with a decoder. They are built for tasks that transform
one piece of text into another.

This guide builds a decoder-only model. The same architecture as GPT
and LLaMA and Mistral. This file explains why and what the other
options do.

## Decoder-Only (GPT family)

### What it looks like

```
Input: "The cat sat on the"
  → Token embedding
    → Causal attention (can only look backward)
      → Feed forward
        → Repeat N times
          → Output projection
            → Predict next token: "mat"
```

The key feature is the causal mask. Every token can only attend to
tokens that came before it. Token 5 can see tokens 0 through 4. Token
5 cannot see token 6 because token 6 has not been written yet.

### How it's trained

Decoder-only models are trained on next token prediction. Show the
model a sequence of tokens. Ask it to predict each next token. The
model learns to guess what comes next. This is called autoregressive
training.

```
Training example:
  Input:  [The, cat, sat, on, the]
  Target: [cat, sat, on, the, mat]

  The model sees "The" and must predict "cat"
  The model sees "The cat" and must predict "sat"
  The model sees "The cat sat" and must predict "on"
  ... and so on
```

Every prediction is made using only past tokens. The model never sees
the future. The causal mask enforces this during training. During
generation no mask is needed because future tokens simply do not exist
yet.

### What it's good at

Text generation. Writing stories. Answering questions. Having
conversations. Completing code. Any task where you produce new text
one token at a time.

Decoder-only models are the universal tool. With enough scale and the
right training data they can do almost anything. GPT-3 showed this in
2020. The model could translate and summarize and answer questions
despite being trained only to predict the next word. It learned these
skills implicitly because the training data contained examples of
translation and summarization and question answering.

### Why we chose it

Decoder-only models are the simplest to build and train. One task.
Predict the next token. One architecture. Causal attention whose output
goes through a feed forward network. No separate encoder. No cross
attention between encoder and decoder. One stack of identical blocks.

They also scale the best. Every major breakthrough in capability from
GPT-2 to GPT-3 to GPT-4 came from decoder-only models. The simplicity
of the architecture means all resources go into making the model bigger
and the data better. There is no complexity budget spent on additional
components.

### The limitation

Decoder-only models cannot look at the full input bidirectionally.
Token 5 cannot use information from token 10 because token 10 does not
exist yet. This is fine for generation but suboptimal for understanding
tasks where the entire input is available from the start.

For tasks like classification or named entity recognition where you
have the complete input a bidirectional model can capture context from
both directions. A decoder-only model can only capture context from the
left. In practice this matters less than you might think. With enough
scale a decoder-only model learns to compensate for the missing right
context by building rich representations that anticipate what comes
next.

## Encoder-Only (BERT family)

### What it looks like

```
Input: "The cat sat on the [MASK]"
  → Token embedding
    → Bidirectional attention (can look everywhere)
      → Feed forward
        → Repeat N times
          → Output projection
            → Predict masked token: "mat"
```

The key feature is bidirectional attention. Every token can attend to
every other token regardless of position. There is no causal mask.
Token 5 can see token 0 and token 10 equally.

### How it's trained

Encoder-only models are trained on masked language modeling. Hide some
percentage of input tokens randomly. Ask the model to predict what was
hidden.

```
Training example:
  Original: "The cat sat on the mat"
  Masked:   "The cat [MASK] on the [MASK]"
  Target:   "sat" and "mat"

  The model sees the whole sentence including words after the mask.
  It uses context from both directions to predict the hidden words.
```

The model sees the entire input at once. It can use information from
words before AND after the masked token. This is fundamentally
different from decoder-only training where the model is blind to the
future.

### What it's good at

Understanding tasks. Classification. Named entity recognition. Question
answering where the answer is in the provided text. Sentiment analysis.
Any task where the input is complete and the output is a label or a
span of text rather than a generated sequence.

BERT embeddings became the standard for representing text. For years
the best approach for any NLP task was to take a pretrained BERT model
and add a small task specific head on top. Fine-tune for a few epochs.
The approach worked because BERT's bidirectional understanding captured
rich representations of word meaning in context.

### The limitation

Encoder-only models cannot generate text autoregressively. They have
no causal mask. They have no mechanism to produce one token at a time
conditioned on previous outputs. You cannot use BERT to write a story
or hold a conversation.

Encoder-only models are also limited by their training objective. Masked
language modeling teaches the model to fill in blanks. It does not teach
the model to produce coherent sequences. You can generate text by
iteratively masking and predicting but the output is typically worse
than what a decoder-only model produces.

## Encoder-Decoder (T5 family)

### What it looks like

```
Input: "Translate to French: The cat sat on the mat"
  → Encoder (bidirectional attention)
    → Hidden representation of the full input
      → Decoder (causal attention + cross attention)
        → Output: "Le chat s'est assis sur le tapis"
```

The encoder reads the entire input bidirectionally. It produces a dense
representation of the input. The decoder generates the output
autoregressively one token at a time. The decoder has both causal self
attention like a GPT and cross attention that looks at the encoder's
output.

The cross attention is the key difference from decoder-only models. At
every generation step the decoder can look back at the full encoded
input. This gives the decoder direct access to the input representation
without needing to encode it in the autoregressive state.

### How it's trained

Encoder-decoder models are trained on sequence to sequence tasks. Show
the model an input sequence and a target output sequence. The encoder
processes the input. The decoder generates the output one token at a
time.

```
Training example:
  Input:  "Summarize: The cat sat on the mat for three hours..."
  Target: "A cat stayed on a mat for a long time."

  The encoder reads the full input bidirectionally.
  The decoder generates "A" then "cat" then "stayed" and so on.
  At each step the decoder can cross attend to the encoder's output.
```

The training uses teacher forcing. The decoder is given the correct
previous tokens during training. The model learns to produce the next
token given the input and the correct history.

### What it's good at

Sequence to sequence tasks. Translation. Summarization. Any task where
the input and output are both text but have different lengths or
structures.

Encoder-decoder models separate the concerns. The encoder focuses on
understanding the input. The decoder focuses on generating the output.
This division of labor can be more efficient than a decoder-only model
which must do both in a single stack of layers.

### The limitation

Encoder-decoder models are more complex. Two separate stacks of layers.
Cross attention between them. More parameters for the same quality on
general language tasks. The architecture is specialized for sequence to
sequence tasks and less flexible for open ended generation.

The rise of decoder-only models has reduced the popularity of encoder
decoder architectures. A large enough decoder-only model can implicitly
perform the separation that an encoder-decoder model makes explicit.
GPT-3 showed this for translation and summarization. The decoder-only
model learned to understand the input and generate the output in a
single stack of layers.

## Why this guide teaches decoder-only

Decoder-only models are the foundation of modern AI. ChatGPT is a
decoder-only model. Claude is a decoder-only model. LLaMA and Mistral
are decoder-only models. Understanding how they work means
understanding the architecture behind the most capable AI systems ever
built.

The architecture is also the simplest. One stack of blocks. One
attention pattern with causal masking. One training objective. Next
token prediction. The simplicity makes it the best starting point for
learning. Once you understand the decoder-only transformer you can
understand any transformer variant.

Encoder-only models are still widely used. BERT and its variants power
search engines and classification systems and information retrieval.
But they cannot generate text. Understanding them is useful for
specialized applications but not essential for building generative AI.

Encoder-decoder models are becoming less common. The gap between
decoder-only and encoder-decoder performance has narrowed. For most
practical purposes a large decoder-only model matches or exceeds an
encoder-decoder model on the same task. The added complexity is harder
to justify.

## When to use each

```
Do you need to generate new text token by token?
  → Decoder-only (GPT, LLaMA, Mistral)

Do you need to understand text and produce a label or classification?
  → Encoder-only (BERT, RoBERTa, DeBERTa)

Do you need to transform text from one form to another and want the
best possible quality for a specific task?
  → Encoder-decoder (T5, BART)

Do you want one architecture that can do everything reasonably well
and is simple to understand and build?
  → Decoder-only
```

## What you need to remember

Decoder-only models generate text one token at a time using only past
context. They are trained on next token prediction. This is the GPT
family and what this entire guide teaches.

Encoder-only models understand text bidirectionally using the full
context. They are trained on masked language modeling. This is the
BERT family. They cannot generate text.

Encoder-decoder models combine both. An encoder reads the input
bidirectionally. A decoder generates the output autoregressively with
cross attention to the encoder. This is the T5 family. They are built
for sequence to sequence tasks.

All three use the same building blocks. Attention. Feed forward
networks. Residual connections. Normalization. The only differences are
the attention mask pattern and the training objective. Master the
decoder-only architecture and you have mastered the foundation of all
modern language models.

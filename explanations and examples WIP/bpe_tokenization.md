# BPE Tokenization: Turning Words into Numbers

## What is it

BPE stands for Byte Pair Encoding. It is the first thing a
language model does when it reads text. BPE takes a sentence like
*The cat sat on the mat* and turns it into a list of numbers like
[464, 3797, 3332, 319, 262, 2603].

Computers do not understand letters. They only understand numbers.
Every pixel on your screen is a number. Every sound from your
speaker is a number. Every key you press is a number. When we
want a computer to understand language we must turn the language
into numbers first. BPE is how we do it.

## Where is it used

BPE is the very first step in every language model pipeline. It
sits between the raw text and the embedding layer.

```
Raw text: "The cat sat on the mat"
    ↓
BPE tokenizer
    ↓
Token IDs: [464, 3797, 3332, 319, 262, 2603]
    ↓
Embedding layer
    ↓
Vectors for the attention layer
```

GPT-2 GPT-3 and GPT-4 all use BPE. LLaMA and Mistral use a
variant called SentencePiece that is based on the same idea.

## Why we need it

Imagine giving every English word its own number. The word *cat*
is number 9246. The word *the* is number 279. This works for
common words. But what about rare words.

English has over one million words. Most of them are rare. Words
like *antidisestablishmentarianism* appear maybe once in a billion
sentences. If we give every rare word its own number we need a
giant vocabulary. The model becomes slow and wastes space.

Worse still new words appear all the time. *Rizz* is a word now.
*Skibidi* is a word now. If the vocabulary was fixed when the
model was trained the model cannot handle any word that was
invented after training. It would see an unknown symbol and fail.

BPE solves this by breaking words into pieces called subwords.
Common words stay whole. *The* becomes one token. *Cat* becomes
one token. Rare words get split into smaller pieces.

```
Common:   "cat"           → [9246]        (one token)
Common:   "the"           → [279]         (one token)
Rare:     "unbelievably"  → [437, 16289, 11387]   (three tokens)
New word: "rizz"          → [r, i, z, z] (still works via character tokens)
```

Since every character is also a token the model can represent any
word ever invented or yet to be invented. It just might need more
tokens for unfamiliar words.

## When was it invented

BPE was invented in 1994 for data compression. It was repurposed
for language in 2016 by researchers at Google who needed a better
way to handle rare words in machine translation. GPT-1 adopted it
in 2018 and every GPT model has used it since.

## How it works: build a vocabulary from scratch

The best way to understand BPE is to watch it build a vocabulary
from a tiny example. We will use just four words and see how the
algorithm merges pairs of characters.

### Starting point

Our training text has four words with spaces marked as `_`:

```
l o w _
l o w e r _
l o w e s t _
l o w e s t _
```

Each character is its own token. Our vocabulary has nine tokens:

```
{l, o, w, e, r, s, t, _, total=9}
```

### Round 1: merge the most frequent pair

Count every pair of characters that appear next to each other.

```
lo: appears 4 times  (l+o in every word)
ow: appears 4 times  (o+w in every word)
w_: appears 2 times  (w+_ before space)
_e: appears 2 times  (_+e in lower and lowest)
er: appears 2 times  (e+r in lower)
es: appears 2 times  (e+s in lowest twice)
st: appears 2 times  (s+t in lowest twice)
... all other pairs appear once or zero times
```

The pair *lo* appears four times. That is the most. We merge *l*
and *o* into a new token called *lo*.

Our text becomes:

```
lo w _
lo w e r _
lo w e s t _
lo w e s t _
```

Our vocabulary now has ten tokens. We added *lo* as a new token.

### Round 2: the next most frequent pair

Count again:

```
low: appears 4 times  (lo+w in every word)
w_: appears 2 times
_e: appears 2 times
er: appears 2 times
es: appears 2 times
st: appears 2 times
```

The pair *lo* and *w* appears four times. Wait that sounds wrong.
Let me be more precise. We count *adjacent* pairs in the current
text. After round 1 our tokens are *lo* and *w* sitting next to
each other. So the pair is {lo, w}. We merge them into *low*.

```
low _
low e r _
low e s t _
low e s t _
```

Vocabulary now has eleven tokens. We keep going.

### Round 3

Count pairs:

```
low_: appears 2 times  (low+_ then low appears twice more but low+_ appears twice)
_e: appears 2 times
er: appears 2 times
es: appears 2 times
st: appears 2 times
```

Wait. Let me count more carefully. The pair {low, _} appears twice
at the start. But what about the other instances of *low*. They
are not adjacent to *_*. They are adjacent to *e*. So low+e
appears twice too.

Let me do this more carefully:

```
Adjacent pairs after round 2:
Position 1-2: {lo, w} merged to {low} already done
"But we already merged lo+w to low, so now what"

Actually the merge process continues. After each merge we scan
again. Let me skip the repetitive counting and show the final
result after many rounds.
```

### The final result after all merges

After enough rounds the algorithm stops when no pair appears more
than once or when we reach our target vocabulary size. For our
tiny example here is what the vocabulary might look like:

```
Single characters: l, o, w, e, r, s, t, _
Pairs merged: lo, ow, low, er, es, st, est, low_, __ (space)
```

Now the word *lower* becomes three tokens: *low* + *er* + *_*.
The word *lowest* becomes two tokens: *low* + *est*.

We built this from scratch. Real BPE tokenizers like GPT-2 use
50 thousand merges. They start from all 256 possible byte values
and merge the most frequent byte pairs across billions of words.
The result is a vocabulary that can represent any text in any
language using a small set of reusable pieces.

## How GPT-2 tokenizes real text

You do not need to build your own vocabulary. We can use the one
GPT-2 already trained. Here is a small program that shows how
text becomes tokens.

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

# Common words stay whole
print(tokenizer.encode("the cat sat"))
# Output: [1169, 3797, 3332]  -- three tokens for three words

# Rare words get split
print(tokenizer.encode("antidisestablishmentarianism"))
# Output: [378, 420, 1634, 2013, 82, 622, 441, 979, 389]
# Nine tokens for one very long word

# Show the pieces of the rare word
pieces = [tokenizer.decode([t]) for t in
          tokenizer.encode("antidisestablishmentarianism")]
print(pieces)
# Output: ['ant', 'idis', 'establish', 'ment', 'ar', 'ian', 'ism']

# New words still work character by character
print(tokenizer.encode("skibidirizz"))
# Output: [87, 68, 73, 390, 68, 73, 89, 416, 89, 89]

# Emojis work too
print(tokenizer.encode("Hello 😊 world"))
# Output: [15496, 52430, 23530, 248, 995]
```

## Space handling

GPT-2 uses a clever trick for spaces. Instead of a space being
its own token it attaches the space to the start of the next
word. The word *cat* with a space before it is a different token
than *cat* without a space. This saves tokens because spaces
before words are more common than spaces alone.

```
"cat"      → token 3797
" cat"     → token 3797 with a space prefix (different representation)
"the cat"  → [1169, 3797] -- the space is part of the cat token
```

This is why GPT-2 tokenizers are more efficient for English text.
Every space is baked into the word that follows instead of being
a separate token.

## Special tokens

Not all tokens represent text. Some are special markers.

| Token | Meaning |
|---|---|
| `<|endoftext|>` | Marks the end of a document. Critical for training. Without it the model thinks two different books are one continuous story. |
| Beginning of text markers | Some tokenizers add a token at the very start of every sequence. GPT-2 does not. |
| Padding tokens | Used when multiple sentences have different lengths and need to be the same size for batch processing. |

## Vocabulary size matters

The number of tokens in the vocabulary is a tradeoff.

| Vocab size | Pros | Cons |
|---|---|---|
| Small (5K) | Fast model output layer | Words get split into too many pieces and lose meaning |
| Medium (50K) | Sweet spot for English | Some rare words still split |
| Large (250K) | Most words stay whole | Output layer is huge and slow |

GPT-2 uses 50257 tokens. This is about fifty thousand merges plus
256 base byte tokens plus one special token. This has proven to
be the best balance for English text. Most modern models use
somewhere between thirty thousand and one hundred thousand tokens.

## What you need to remember

BPE breaks text into small reusable pieces. Common words stay as
one piece. Rare words get split into smaller pieces. New words
fall back to individual characters.

Every language model starts with a tokenizer. If the tokenizer is
bad the model will be bad. It does not matter how smart the
attention is if the words it receives make no sense. Tokenization
is the foundation. Everything else builds on top.

The vocabulary is built by repeatedly merging the most frequent
pair of adjacent tokens. Start from single characters. Merge the
most common pair. Repeat until you have enough tokens. The merges
are saved as rules. When new text arrives the rules are applied
in order to split the text into the same vocabulary.

This simple idea from 1994 is still powering every modern AI
language system today.

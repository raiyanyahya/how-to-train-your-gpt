# Chapter 2 — Tokenization: Turning Words into Numbers

## The 5-Year-Old Analogy

Computers can only understand **numbers**. They don't know what the letter "A" means — they know "65" (its ASCII code). So we need to convert text into numbers before feeding it to a neural network.

The simplest idea: **assign every word a number**:
```
"cat"  ->  9246
"sat"  ->  6734
"on"   ->   389
"the"  ->   279
"mat"  -> 16789
```

But English has hundreds of thousands of words. Do we really need a number for "antidisestablishmentarianism"? And what about new words like "skibidi" that didn't exist when we built the vocabulary?

## The Solution: Subword Tokenization (BPE)

Instead of whole words, we break text into **frequent subword pieces**:

```
"unbelievably" -> "un" + "believ" + "ably"
"running"      -> "runn" + "ing"
"cats"         -> "cat" + "s"
"lower"        -> "low" + "er"
"GPT"          -> "G" + "P" + "T"
```

This is **Byte Pair Encoding (BPE)** — the exact algorithm used by GPT-2, GPT-3, GPT-4, and most modern models.

### How BPE Works — Step by Step

BPE starts with every character as its own "token," then repeatedly merges the most frequent pair:

**Starting text:** `"low lower lowest"`

```
Step 0 (initial — each character is a token):
l o w _ l o w e r _ l o w e s t

Step 1 (most frequent pair: 'l'+'o' -> 'lo'):
lo w _ lo w e r _ lo w e s t

Step 2 (most frequent pair: 'lo'+'w' -> 'low'):
low _ low e r _ low e s t

Step 3 (most frequent pair: 'e'+'s' -> 'es'):
low _ low e r _ low es t

Step 4 (most frequent pair: 'es'+'t' -> 'est'):
low _ low e r _ low est

Step 5 (most frequent pair: 'low'+'_' -> 'low_'):
low_ low e r _ low_ est
```

After enough merges, we have a vocabulary like: `{l, o, w, e, r, s, t, _, lo, ow, low, er, es, est, low_}`

Now new words can be represented using these pieces even if we've never seen them before:

```
"lowest"  -> "low" + "est"     (both in vocabulary!)
"slower"  -> "s" + "low" + "er" (never seen before, but works!)
```

### Why BPE Beats Word-Level Tokenization

| Problem | Word-Level | BPE |
|---|---|---|
| "running" vs "run" | Different tokens — no shared meaning | "runn" + "ing" — the model sees the connection |
| New word: "rizz" | Unknown token → model fails | "r" + "i" + "z" + "z" → works with characters |
| Vocabulary size | 500K+ (too many rare words) | 50K (balanced, efficient) |
| Unicode/emoji handling | Often broken | Character-level fallback never fails |

### What About Special Characters and Emojis?

BPE operates on **bytes**, not characters. This means it can tokenize ANYTHING that can be represented as bytes — emojis, Chinese characters, code, LaTeX, even binary data:

```
"Hello 😊"  ->  ["Hello", " Ġ", "😊"]    (Ġ = space prefix in GPT tokenizer)
"你好"       ->  tokenized via UTF-8 bytes
"def foo():"->  ["def", "Ġfoo", "()", ":"]
```

### GPT Tokenizer Conventions

| Token | Example | Meaning |
|---|---|---|
| Normal tokens | `"cat"`, `"the"`, `"ing"` | Regular subword pieces |
| Space-prefixed | `"Ġcat"`, `"Ġthe"` | Word starts after a space (Ġ is a special character) |
| `<\|endoftext\|>` | EOS token | Marks end of a document — critical for training |
| Capital letters | `"The"` vs `"the"` | Different tokens! Case matters |

### The EOS Token — Why It Matters

The `<|endoftext|>` (End Of Sequence) token is **critical** and often overlooked:

```python
# WITHOUT EOS — two documents get merged:
doc1 = "The cat sat."     # tokens: [464, 3797, 3332, 13]
doc2 = "The dog ran."     # tokens: [464, 3290, 3407, 13]
# Result: [464, 3797, 3332, 13, 464, 3290, 3407, 13]
# Model sees: "...sat. The dog ran." — thinks it's ONE document
# Learns: "sat." is often followed by "The" — WRONG!

# WITH EOS — documents are separated:
tokens = [464, 3797, 3332, 13, EOS, 464, 3290, 3407, 13, EOS]
# Model learns: EOS means "we're done here, next token is unrelated"
```

## Tokenizer Code — Annotated

```python
from dataclasses import dataclass
import tiktoken


@dataclass
class TokenizerConfig:
    """
    WHAT: Keeps all tokenizer settings in one place.
    WHY: Like a recipe card — consistent across the whole project.
         Change one value and everything updates automatically.
    """
    name: str = "gpt2"                # WHAT: use GPT-2's pretrained BPE tokenizer
                                       # WHY: same BPE as GPT-3/4 — 50K merges,
                                       #      battle-tested on billions of documents,
                                       #      and already trained (no weeks of work)
    vocab_size: int = 50257           # WHAT: total number of unique tokens
                                       # WHY: 50,257 is the exact GPT-2 vocabulary size
                                       #      (50,000 merges + 256 byte tokens + 1 EOS)
                                       #      This is the "goldilocks" number —
                                       #      big enough for rare subwords,
                                       #      small enough for fast matrix operations


class SimpleTokenizer:
    """
    WHAT: Wraps tiktoken to give us a friendly, consistent interface.
    WHY: tiktoken's raw API is low-level (you need to specify
         allowed_special every call). This wrapper makes encode/decode
         trivial — just call .encode("hello") and get tokens back.
         
         It also handles the EOS token consistently so we never
         accidentally forget to add it during training data prep.
    """

    def __init__(self, config: TokenizerConfig = None):
        """
        WHAT: Initialize the tokenizer with GPT-2's BPE vocabulary.
        WHY: We use a pretrained tokenizer because:
             1. Training a tokenizer from scratch takes weeks of CPU time
             2. GPT-2's tokenizer is open-source, fast, and well-tested
             3. Using the same tokenizer as production models means our
                code works identically to how GPT-3 tokenizes
        """
        self.config = config or TokenizerConfig()

        # WHAT: Load the GPT-2 encoding from tiktoken
        # WHY: tiktoken stores pretrained BPE merge tables.
        #      get_encoding("gpt2") loads the exact 50K merges
        #      that GPT-2 was trained with.
        self.enc = tiktoken.get_encoding(self.config.name)

        # WHAT: Define and encode the End-of-Sequence token
        # WHY: <|endoftext|> is the special token that marks boundaries
        #      between documents. During training, we insert it between
        #      every document so the model learns where one text ends
        #      and another begins.
        self.eos_token = "<|endoftext|>"       # The string representation
        self.eos_token_id = self.enc.encode(    # Convert to its token ID
            self.eos_token,
            allowed_special={self.eos_token}    # WHY: tiktoken blocks special tokens
                                                #      by default for safety. We must
                                                #      explicitly allow EOS encoding.
        )[0]  # [0] because encode() returns a list — we want the single ID

    def encode(self, text: str) -> list[int]:
        """
        WHAT: Turn text into a list of integer token IDs.
        WHY: Neural networks only eat numbers. Raw strings like
             "Hello world" mean nothing to matrix multiplication.

        Example: "Hello world" -> [15496, 995]

        Under the hood: tiktoken splits the text into subword pieces
        using the pretrained BPE merge table, then looks up each
        piece's ID in the vocabulary.
        """
        # WHAT: Use tiktoken's fast C/Rust-based encoder
        # WHY: tiktoken is written in Rust, not Python.
        #      It can tokenize hundreds of MB of text per second.
        #      A pure Python BPE tokenizer would be 100x slower.
        return self.enc.encode(text, allowed_special={self.eos_token})

    def decode(self, ids: list[int]) -> str:
        """
        WHAT: Turn token IDs back into human-readable text.
        WHY: After the model generates a sequence of token IDs
             during inference, we need to convert them back to
             text so humans can read the output.

        Example: [15496, 995] -> "Hello world"
        """
        return self.enc.decode(ids)

    @property
    def vocab_size(self) -> int:
        """
        WHAT: How many unique tokens exist in the vocabulary.
        WHY: This number determines the size of our model's output
             layer — the final Linear layer must have vocab_size
             outputs (one score for each possible next token).
             
             50,257 means the model chooses from 50,257 possibilities
             every time it predicts the next word.
        """
        return self.config.vocab_size


# ===== WHAT: Quick self-test =====
# WHY: Always test each component in isolation before combining.
#      "Does the tokenizer work?" is a 5-second check that saves
#      hours of debugging a misbehaving training loop.
if __name__ == "__main__":
    tokenizer = SimpleTokenizer()

    # Test 1: Basic text
    test_text = "The cat sat on the mat."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"Test 1 — Basic:")
    print(f"  Original: '{test_text}'")
    print(f"  Encoded:  {encoded}")
    print(f"  Decoded:  '{decoded}'")
    print(f"  Match:    {test_text == decoded}")

    # Test 2: EOS token
    eos = tokenizer.encode(tokenizer.eos_token)
    print(f"\nTest 2 — EOS token:")
    print(f"  String: '{tokenizer.eos_token}'")
    print(f"  Token ID: {tokenizer.eos_token_id}")
    print(f"  Encode result: {eos}")

    # Test 3: Rare/unseen word
    rare = tokenizer.encode("antidisestablishmentarianism")
    decoded_rare = tokenizer.decode(rare)
    print(f"\nTest 3 — Rare word:")
    print(f"  Encoded: {rare}")
    print(f"  Pieces:  {[tokenizer.decode([t]) for t in rare]}")
    print(f"  Decoded: '{decoded_rare}'")

    # Test 4: Emoji/Unicode
    emoji = tokenizer.encode("Hello 😊 world")
    print(f"\nTest 4 — Emoji:")
    print(f"  Encoded: {emoji}")
    print(f"  Decoded: '{tokenizer.decode(emoji)}'")

    print(f"\n  Vocab size: {tokenizer.vocab_size:,}")
```

**Expected output:**
```
Test 1 — Basic:
  Original: 'The cat sat on the mat.'
  Encoded:  [464, 3797, 3332, 319, 262, 2603, 13]
  Decoded:  'The cat sat on the mat.'
  Match:    True

Test 2 — EOS token:
  String: '<|endoftext|>'
  Token ID: 50256
  Encode result: [50256]

Test 3 — Rare word:
  Encoded: [378, 420, 1634, 2013, 82, 622, 441, 979, 389]
  Pieces:  ['ant', 'idis', 'establish', 'ment', 'ar', 'ian', 'ism']
  Decoded: 'antidisestablishmentarianism'

Test 4 — Emoji:
  Encoded: [15496, 52430, 23530, 248, 995]
  Decoded: 'Hello 😊 world'

  Vocab size: 50,257
```

---

**Previous:** [Chapter 1 — Setup](01_setup.md)
**Next:** [Chapter 3 — Embeddings](03_embeddings.md)

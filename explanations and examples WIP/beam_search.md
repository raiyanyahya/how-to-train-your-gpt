# Beam Search: Making Generation More Accurate

## The short answer

Beam search keeps multiple candidate sequences alive at every step
instead of picking just one token. At each position it considers the
top K best ways to extend the sequence so far. The beam width K is
typically 3 to 5. The result is more accurate but less creative text.
Beam search is the standard for translation and summarization where
getting the right answer matters more than being interesting.

Think of it like parallel parking. Greedy search picks the first
available parking spot and takes it. Sampling might park anywhere in
the lot. Beam search backs up and tries multiple parallel parking
attempts simultaneously and picks the best final position.

## Where it fits

Beam search replaces the token-by-token sampling in the generation
loop. It does not change the model or the training. It only changes
how the model's predictions are used to build the output sequence.

```
Generation strategies:
  Greedy:   Pick the single highest probability token. Fast. Repetitive.
  Sampling: Pick randomly from the probability distribution. Creative. Variable.
  Beam:     Maintain K sequences. Check all K×V next tokens. Keep top K.
            Accurate. Deterministic. Slower.
```

## How it works step by step

Imagine we ask the model to complete *The cat sat on the* with beam
width 3.

### Step 1: Start

We have one empty sequence. The beam contains one item that is the
prompt. We want to generate 4 more tokens.

```
Beam (size 1): ["The cat sat on the"]
```

### Step 2: First token

The model processes the prompt and produces logits for the next token.
Instead of picking just one we look at the top 3.

```
Top tokens and their probabilities:
  "mat"   : logit = 4.2, prob = 0.45
  "floor" : logit = 3.1, prob = 0.22
  "table" : logit = 2.5, prob = 0.12
  "chair" : logit = 1.8, prob = 0.08
  "rug"   : logit = 1.2, prob = 0.05
  ... (50257 total)
```

We extend the beam with the top 3.

```
Beam (size 3):
  Sequence A: "The cat sat on the mat"     (score = 0.45)
  Sequence B: "The cat sat on the floor"   (score = 0.22)
  Sequence C: "The cat sat on the table"   (score = 0.12)
```

### Step 3: Second token

Now each of the 3 sequences asks the model for their next token. For
each sequence the model produces 50257 probabilities. We combine each
candidate with its parent sequence's cumulative score.

```
For Sequence A ("... mat"):
  "and"    : prob = 0.35  → cumulative = 0.45 × 0.35 = 0.158
  ","      : prob = 0.25  → cumulative = 0.45 × 0.25 = 0.113
  "because": prob = 0.18  → cumulative = 0.45 × 0.18 = 0.081
  ...

For Sequence B ("... floor"):
  "and"    : prob = 0.30  → cumulative = 0.22 × 0.30 = 0.066
  ","      : prob = 0.20  → cumulative = 0.22 × 0.20 = 0.044
  ...

For Sequence C ("... table"):
  "and"    : prob = 0.28  → cumulative = 0.12 × 0.28 = 0.034
  ...
```

We have 3 × 50257 candidates. We pick the top 3 by cumulative score.

```
Beam (size 3):
  "The cat sat on the mat and"      (score = 0.158)
  "The cat sat on the mat,"         (score = 0.113)
  "The cat sat on the mat because"  (score = 0.081)
```

Notice something. All three top sequences start with *mat*. The beam
converged on a single prefix. This is common in beam search. The top
candidates often share a prefix. The beam is not three independent
searches. It is one search that maintains three candidate completions.

### Step 4 and beyond

Continue until we reach the maximum length or all sequences end with
the end-of-text token. At the end we pick the sequence with the highest
cumulative score.

```
Final beam (after 4 tokens):
  "The cat sat on the mat and then"          (score = 0.098)
  "The cat sat on the mat and waited"        (score = 0.072)
  "The cat sat on the mat and the"           (score = 0.054)

Top result:
  "The cat sat on the mat and then"
```

## Why we use log probabilities

Notice the scores are multiplying at each step. 0.45 times 0.35 equals
0.158. At step 10 the cumulative score might be 0.000001. These tiny
numbers are hard to compare because floating point precision is limited.

The solution uses log probabilities. Instead of multiplying we add.
This keeps all numbers in a manageable range.

```
Cumulative score = prob_1 × prob_2 × prob_3 × ...
Cumulative log score = log(prob_1) + log(prob_2) + log(prob_3) + ...

For mat → and:
  log score = log(0.45) + log(0.35) = -0.798 + (-1.050) = -1.848
  Score = e^(-1.848) = 0.158  (same as before)
```

Since the log probabilities are always negative the cumulative score
becomes more negative at each step. This means shorter sequences have
higher scores simply because they have fewer negative terms added.
Beam search naturally favors shorter outputs unless we normalize.

The fix is length normalization. Divide the cumulative log score by
the number of tokens generated.

```
Normalized score = cumulative_log_score / length^α

α = 0: no normalization (favors short sequences)
α = 1: full normalization (favors long sequences)
α = 0.6 to 0.8: standard (balanced)
```

Without normalization the model might output a period after one token
because any continuation makes the score worse. With normalization
the model is encouraged to produce complete answers.

## Beam search versus greedy versus sampling

```
Prompt: "The cat sat on the"

Greedy (beam=1):
  "The cat sat on the mat and then the cat sat on the mat and then..."

Sampling (T=0.8, top_k=50):
  "The cat sat on the windowsill watching birds flutter past the garden."

Beam search (beam=5):
  "The cat sat on the mat and waited patiently for its owner to come home."
```

Greedy repeats because it always picks the single most likely token.
The most likely continuation after *the cat sat on the mat* is another
*the cat sat on the mat*. The probability trap is inescapable.

Sampling escapes the trap by sometimes picking less likely tokens.
The result is more interesting but sometimes nonsensical. The quality
varies with each generation.

Beam search explores multiple paths and picks the best complete
sequence. It avoids the repetition trap because *and waited patiently*
might have a higher cumulative probability than looping back to *the*.
The result is accurate and coherent but less creative.

## A simplified beam search implementation

```python
import torch
import torch.nn.functional as F

def beam_search_generate(model, input_ids, max_new_tokens, beam_width=5,
                         length_penalty=0.7, temperature=1.0):
    """
    Generate text using beam search.
    Returns the single best sequence.
    """
    model.eval()
    device = input_ids.device

    # Each beam is a sequence. We maintain beam_width sequences.
    beams = [(input_ids.clone(), 0.0)]  # (sequence, cumulative_log_score)
    finished_beams = []

    for step in range(max_new_tokens):
        candidates = []

        for seq, score in beams:
            if seq[0, -1].item() == eos_token_id:
                finished_beams.append((seq, score / (len(seq[0]) ** length_penalty)))
                continue

            with torch.no_grad():
                logits, _ = model(seq[:, -max_seq_len:])
                logits = logits[:, -1, :] / temperature
                top_k_logits, top_k_indices = torch.topk(
                    logits, beam_width * 2, dim=-1
                )
                log_probs = F.log_softmax(top_k_logits, dim=-1)

            for j in range(beam_width * 2):
                token = top_k_indices[0, j:j+1]
                new_seq = torch.cat([seq, token.unsqueeze(0)], dim=1)
                new_score = score + log_probs[0, j].item()
                candidates.append((new_seq, new_score))

        # Keep top beam_width candidates by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]

        if len(finished_beams) >= beam_width:
            break

    # Add any remaining beams as finished
    for seq, score in beams:
        finished_beams.append(
            (seq, score / (len(seq[0]) ** length_penalty))
        )

    # Pick best finished beam
    finished_beams.sort(key=lambda x: x[1], reverse=True)
    return finished_beams[0][0]


# Usage
prompt_ids = tokenizer.encode("The cat sat on the", return_tensors="pt")
output_ids = beam_search_generate(model, prompt_ids, max_new_tokens=30, beam_width=5)
text = tokenizer.decode(output_ids[0])
```

The crucial detail is on line 33. We consider `beam_width × 2`
candidates for each sequence not just `beam_width`. This is because
some candidates share prefixes and would merge in the next step.
Considering twice as many candidates per source sequence gives the beam
more diversity to choose from.

## When to use each strategy

```
Greedy (beam=1):     Translation. Code generation. Tasks where the
                     output should be the single most likely answer.
                     Fast but can get stuck in loops.

Beam search (3-5):   Summarization. Captioning. Speech recognition.
                     Tasks where accuracy matters over creativity.
                     Good for structured output with clear right answers.

Sampling (T=0.8):    Creative writing. Dialogue. Story generation.
                     Tasks where diversity and interest matter.
                     Different each time. Less repetitive than greedy.

Beam + sampling:     Some production systems combine beam search with
                     small sampling to add slight variety while
                     maintaining quality. Rarely needed.
```

Beam search is slower than greedy by roughly the beam width factor.
Beam width 5 means 5 forward passes per step instead of 1. With KV
caching this overhead can be reduced because the beams share the same
prefix and the keys and values for the shared prefix are cached.

## The repetition problem

Beam search can still produce repetitive output. The beam converges
to a single prefix and then loops. The repetition is more subtle than
in greedy decoding but still present.

Fixes include n-gram blocking which prevents the same n-gram from
appearing twice in a beam and repetition penalty which reduces the
probability of tokens that have already appeared. These are applied on
top of beam search and are standard in production systems.

## What you need to remember

Beam search maintains multiple candidate sequences and picks the best
one at the end. It is more accurate than greedy decoding and more
consistent than sampling. It uses log probabilities to avoid floating
point underflow and length normalization to avoid favoring short
sequences.

Beam width 5 is standard for most tasks. Higher beam widths are slower
and have diminishing returns. Lower beam widths approach greedy
behavior.

Beam search is best for tasks with a clear correct answer like
translation and summarization. It is not ideal for creative writing
where diversity matters more than accuracy.

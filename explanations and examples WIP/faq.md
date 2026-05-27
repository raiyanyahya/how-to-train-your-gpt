# FAQ and Troubleshooting

## Training

### Q: My loss is stuck at 10.8 after thousands of steps. What is wrong

10.8 is the loss of a random model predicting uniformly over the
vocabulary. It equals ln(50257). If your loss stays at 10.8 your model
is not learning. Possible causes.

The learning rate is too low to move the weights. Try increasing from
3e-4 to 1e-3 temporarily and see if the loss moves.

The optimizer is not stepping. Check that `optimizer.step()` is being
called and that `optimizer.zero_grad()` is called after.

The gradients are zero. A bug in the loss computation or the backward
pass. Check that `loss.backward()` is called and that gradients are
flowing. Print `model.layers[0].attention.qkv_proj.weight.grad` after
backward. It should be non zero.

The data is wrong. Maybe input and target are identical or the targets
are all the same token. Print a few samples.

### Q: My loss is NaN. What happened

NaN means not a number. It means a number overflowed or divided by zero.
This is almost always caused by the learning rate being too high or
gradient clipping being missing.

Fix: lower the learning rate by 10x. Add gradient clipping with max
norm 1.0. Check that your loss function is computing correctly. Print
the logits before loss. If they contain NaN the problem is in the model
forward pass. If they are fine the problem is in the loss computation.

### Q: My loss decreases for a while then suddenly spikes

This is gradient explosion. A rare batch of data causes very large
gradients that shoot the model weights into a region where the loss
is huge.

Fix: add gradient clipping. `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`. This should be called after `loss.backward()` and before `optimizer.step()`.

### Q: Training is extremely slow on CPU. What can I do

CPU training is 10x to 50x slower than GPU. Options:

Use the tiny config (d_model=256, 4 layers). It trains in minutes on
CPU. The small config (d_model=768, 12 layers) will take days.

Use gradient accumulation to simulate larger batches without more
memory. But this does not speed up training. It only lets you use a
bigger effective batch size.

Use a cloud GPU. Google Colab provides a free T4 GPU. Run the colab
notebook for one click training.

Use Apple MPS if you have a Mac. Main.py now auto detects MPS and
enables mixed precision.

### Q: Do I need to train the full 50,000 steps

No. You can stop early. 500 steps on the tiny config gives a loss
around 6 to 7. The model will not be good but it proves the code works.
For the small config 5,000 steps shows obvious learning. 50,000 steps
is for production quality. Stop whenever you are satisfied with the
generated text.

### Q: How do I know if my model is overfitting

If the training loss keeps decreasing but the model generates
repetitive or nonsensical text it is overfitting. The model memorized
the training data instead of learning general patterns.

Fix: increase dropout (try 0.2 or 0.3). Increase weight decay (try
0.2). Reduce the number of training steps. Use a larger and more
diverse dataset.

## Generation

### Q: The model generates gibberish

This is normal for a randomly initialized model or a model trained for
very few steps. Even 500 steps on the tiny config produces mostly
gibberish. The model needs thousands of steps to produce coherent text.

If the model was trained for many steps and still produces gibberish
check that the tokenizer is the same one used during training. Using
a different tokenizer for generation than training produces garbage
because the token IDs mean different things.

### Q: The model repeats the same phrase over and over

This is a common problem called repetitive degeneration. The model
learns that repeating itself is a safe prediction because repeated
patterns are common in text.

Fix: increase the temperature to 0.8 or 1.0. Use top_k sampling with
k=50. Use top_p sampling with p=0.9. These prevent the model from
always picking the most likely token which is often a repetition.

### Q: How do I make the model more creative

Increase temperature to 1.2 or 1.5. Remove or increase top_k to 100.
Set top_p to 0.95. The model will pick less likely tokens more often
producing more varied output.

### Q: How do I make the model more factual

Decrease temperature to 0.3 or 0.5. Set top_k to 20. Use top_p of 0.5
to 0.7. The model will stick to its most confident predictions. More
accurate but less interesting.

### Q: What is the `<|endoftext|>` token I see in my output

This is the end of text marker. The model was trained with this token
between documents. During generation the model sometimes predicts this
token meaning it thinks the text should end. You can filter it out or
stop generation when it appears.

## Architecture

### Q: Why RoPE instead of learned positional embeddings

Learned positional embeddings cannot handle sequences longer than the
training length. If trained on 1024 tokens the model cannot process
2048 tokens. RoPE captures relative position so it generalizes to any
length. RoPE also has no learned parameters. Free improvement.

### Q: Why RMSNorm instead of LayerNorm

RMSNorm is mathematically simpler and about 15 percent faster. It
removes the mean centering and bias which experiments showed are
unnecessary. Every modern model uses RMSNorm.

### Q: Why SwiGLU instead of ReLU or GELU

SwiGLU has a gating mechanism. It learns which information to pass and
which to block. ReLU and GELU treat every input the same way. The gate
gives SwiGLU more expressive power per parameter. At large scale this
translates to better performance.

### Q: Why weight tying

The embedding layer and output layer do inverse operations. Embeddings
map token IDs to vectors. The output layer maps vectors to token
probabilities. Sharing the matrix saves 30 percent of parameters and
improves training because each token embedding gets gradient signals
from both directions.

### Q: Does our model use Flash Attention

No. Flash Attention is an optimized CUDA kernel that speeds up
attention by 2x to 4x. It does not change the math. Our implementation
uses standard PyTorch operations which are slower but understandable.
For production use you would swap in Flash Attention.

### Q: Does our model use grouped query attention

No. Grouped query attention reduces the number of key and value heads
relative to query heads. This saves memory in the KV cache during
inference. Our model uses standard multi head attention where Q K and
V all have the same number of heads.

## Hardware

### Q: Can I train on my laptop

Yes. The tiny config (256 dims, 4 layers, 17M params) trains in 2 to 5
minutes on a modern laptop CPU. The small config (768 dims, 12 layers,
152M params) takes hours to days on CPU. A GPU makes a huge difference.

### Q: What GPU do I need

Tiny config: any GPU or CPU.
GPT-2 Small (152M): 4GB VRAM minimum. 8GB comfortable.
GPT-2 Medium (350M): 8GB VRAM minimum. 12GB comfortable.
GPT-3 1.3B: 12GB VRAM minimum. 16GB comfortable.
GPT-3 6.7B: 24GB VRAM minimum.
GPT-3 175B: 8x A100 80GB.

### Q: How much memory does my model use

Rule of thumb: each parameter uses 2 bytes (bfloat16) for weights plus
8 bytes (float32) for optimizer states during training. Total is about
10 bytes per parameter for training.

```
17M params:  170 MB training memory
152M params: 1.5 GB training memory
7B params:   70 GB training memory (needs multiple GPUs)
```

## Bugs

### Q: PyTorch 2.6+ fails to load my checkpoint

PyTorch 2.6 changed `torch.load` defaults to `weights_only=True`.
Pass `weights_only=False` to load checkpoints containing custom classes
like GPTConfig. Our code handles this. If you use an older notebook
that does not have the fix add `weights_only=False` to your load call.

### Q: I get a shape mismatch error in attention

The most common cause is the sequence length and number of heads being
swapped. Our attention expects input as [batch, seq_len, d_model]
which is reshaped internally to [batch, num_heads, seq_len, head_dim].
If your input has num_heads where seq_len should be the broadcast fails.

### Q: My loss prints as 0.0000

The loss is probably computed as 0 divided by something. Check that
logits and targets have the correct shapes and that cross_entropy is
called correctly. Cross entropy expects logits of shape [N, vocab_size]
and targets of shape [N] with integer class indices.

### Q: I get `RuntimeError: expected scalar type Float but found Half`

This happens when mixed precision is enabled but some operation expects
float32. Use `torch.amp.autocast(device_type, enabled=True)` for the
forward pass and keep the loss computation in float32. The autocast
context manager handles dtype conversion for most operations.

### Q: The attention weights are all NaN after a few training steps

This is usually caused by the attention scores becoming too large
before softmax. Check that you are dividing by `sqrt(head_dim)` in the
attention score computation. Without this division the scores can be
large enough that `exp(score)` overflows to infinity.

### Q: I ran import tiktoken and got an error

Install tiktoken: `pip install tiktoken`. This is the tokenizer used
by GPT-2 and GPT-3. It is written in Rust and is very fast. If you get
a compilation error on installation try `pip install tiktoken --no-binary tiktoken` or upgrade your pip.

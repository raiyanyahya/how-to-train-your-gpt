# Mixed Precision Training: Speed Without Sacrifice

## What is it

Mixed precision training uses two different number formats during
training. Most operations use bfloat16 a compact format that
takes less memory and runs faster. A few critical operations use
float32 the standard format that is more precise. The mix gives
you the speed of the compact format with the accuracy of the
standard one.

Think of it like carrying groceries. You can carry items in your
hands with full control over each one. That is float32. Or you
can put everything in a bag and carry the bag. That is bfloat16.
You lose some control over individual items but you can carry
twice as many in a single trip.

Mixed precision means using the bag for the heavy lifting but
taking items out to handle them carefully when precision matters.

## Where is it used

Mixed precision wraps the forward pass of the model. Every matrix
multiplication inside the attention and feed forward layers runs
in bfloat16. The loss is computed in float32. The weight updates
are stored in float32.

```python
with torch.amp.autocast('cuda', enabled=True):
    # Everything here runs in bfloat16
    logits, loss = model(input_ids, target_ids)

# The loss is float32 for accurate backward pass
loss.backward()
```

Without mixed precision training a large language model would
take twice as long and use twice as much GPU memory. With it you
can train bigger models on the same hardware.

## Why we need it

The bottleneck in neural network training is matrix
multiplication. The attention layer does Q times K transpose. The
feed forward layer does input times weight matrix. These
operations dominate the training time. If we can make them twice
as fast we cut training time in half.

Smaller number formats make matrix multiplication faster because
each number takes less memory. With float32 each number is four
bytes. With bfloat16 each number is two bytes. You can fit twice
as many numbers in the same memory. The GPU can process them in
wider batches. The result is roughly twice the speed.

But why not use float16 everywhere and get even more speed? The
problem is range. Float16 can only represent numbers up to about
sixty five thousand. During training intermediate values can
exceed this limit. The number overflows and becomes infinity. The
model produces garbage. This is why early attempts at half
precision training failed.

Bfloat16 fixes this by keeping the same range as float32. The
maximum possible value is about three point four times ten to
the thirty eighth power for both formats. Bfloat16 cannot
overflow. It just has less precision within that range. For
neural network training this tradeoff is perfect. We need the
range for intermediate values but we do not need seven decimal
digits of precision for every activation.

## When was it invented

Bfloat16 was created by Google in 2017 specifically for their
TPU hardware. It was designed from scratch for neural network
training. NVIDIA adopted it in their A100 GPUs in 2020. Now
every major GPU supports bfloat16 natively. Mixed precision
training using bfloat16 is the default for all production
language model training.

## The three number formats compared

```
Float32:    32 bits total
            1 bit for sign
            8 bits for exponent (range)
            23 bits for mantissa (precision)
            Range: up to ±3.4 × 10³⁸
            Precision: 7 decimal digits

Bfloat16:   16 bits total
            1 bit for sign
            8 bits for exponent (range)
            7 bits for mantissa (precision)
            Range: up to ±3.4 × 10³⁸ (same as float32!)
            Precision: 2 decimal digits

Float16:    16 bits total
            1 bit for sign
            5 bits for exponent (range)
            10 bits for mantissa (precision)
            Range: up to ±65504 (can overflow!)
            Precision: 3 decimal digits
```

Notice that bfloat16 and float32 have the same range. The only
difference is precision. Bfloat16 has seven bits for the mantissa
while float32 has twenty three. This means bfloat16 can represent
numbers with about two decimal digits of accuracy. Float32 can
represent seven. For neural network training two digits is enough.
The gradients and activations do not need extreme precision. They
just need consistent ballpark numbers.

Float16 has better precision than bfloat16 but terrible range. In
practice float16 overflows during long training runs. Bfloat16
does not. This is why bfloat16 won.

## How it works in practice

The training loop has three parts and each uses a different
precision strategy.

### Part 1: the forward pass

Most operations run in bfloat16 inside an autocast context.

```python
with torch.amp.autocast('cuda', enabled=True):
    logits, loss = model(input_ids, target_ids)
```

The autocast context manager automatically converts operations to
bfloat16 wherever it is safe. Matrix multiplications and
convolutions are converted. Normalization operations like RMSNorm
stay in float32 because they need more precision. The developer
does not need to manually specify which operations to convert.
Autocast handles it.

### Part 2: the backward pass

The loss is in float32 for accuracy. The backward pass computes
gradients. Some gradients are in float32 and some in bfloat16.
A gradient scaler watches for underflow.

The gradient scaler multiplies the loss by a large number before
the backward pass. This pushes small gradients into the
representable range of bfloat16. After the backward pass the
scaler divides the gradients back down before the weight update.
This prevents tiny gradients from becoming zero in bfloat16.

```python
scaler = torch.amp.GradScaler('cuda', enabled=True)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Part 3: the weight update

The master weights are always stored in float32. After the
backward pass the gradients are applied to the float32 weights.
This ensures that even if the forward pass computes in bfloat16
the weights themselves never lose precision over many updates.
This is called mixed precision because the forward pass is half
precision and the weight storage is full precision.

```python
# Master weights are float32 always
optimizer.step()  # Applies float32 gradients to float32 weights
```

## A tiny code example

```python
import torch

# Check if your GPU supports bfloat16
if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability()
    print(f"GPU compute capability: {capability}")
    bf16_ok = capability[0] >= 8
    print(f"Bfloat16 supported: {bf16_ok}")
else:
    print("No GPU available. Mixed precision requires CUDA.")
    print("CPU training runs in float32 only.")

# Simple speed comparison
if torch.cuda.is_available():
    size = 4096
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')

    # Float32
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    c32 = a @ b
    end.record()
    torch.cuda.synchronize()
    t32 = start.elapsed_time(end)

    # Bfloat16
    start.record()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        c16 = a @ b
    end.record()
    torch.cuda.synchronize()
    t16 = start.elapsed_time(end)

    print(f"\nMatrix multiply {size}x{size}:")
    print(f"Float32 time:  {t32:.1f} ms")
    print(f"Bfloat16 time: {t16:.1f} ms")
    print(f"Speedup:       {t32/t16:.1f}x")
```

## When not to use mixed precision

Some operations degrade with reduced precision. Normalization
layers like RMSNorm should stay in float32. The softmax in
attention should run in float32 for stability. The loss
computation must be in float32 for accurate gradients. Autocast
handles most of these automatically.

If you train on CPU mixed precision provides no benefit. CPU does
not have native bfloat16 support. The conversions would add
overhead without speedup. Our code checks for CUDA availability
and only enables mixed precision on GPU.

## What you need to remember

Mixed precision training runs most operations in bfloat16 for
speed while keeping critical values in float32 for accuracy. The
bfloat16 format has the same range as float32 so it never
overflows. It has less precision but neural networks do not need
seven decimal digits for every number.

The result is roughly twice the speed and half the memory usage
with no measurable loss in model quality. Every production
language model is trained with mixed precision. It is not an
optional optimization. It is the standard way to train.

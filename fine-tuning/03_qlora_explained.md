# QLoRA: Quantized LoRA

## The short answer

QLoRA is LoRA plus quantization. The base model is compressed to four
bits per weight before applying LoRA adapters. This reduces memory by
four to eight times compared to the full precision model. A seven
billion parameter model that normally needs fourteen gigabytes can run
in under four gigabytes. Fine-tuning becomes possible on a laptop.

## Where the quantization happens

Quantization applies to the frozen base model only. The LoRA adapter
matrices stay in full precision. During training the four bit weights
are temporarily dequantized back to sixteen bit for the forward and
backward passes. The dequantization happens inside the computation.
You do not see it. The model lives in four bits but computes in sixteen
bits.

```
Base model weights (frozen):
  Stored as 4-bit integers
  Dequantized to bfloat16 for computation
  Re-quantized after computation (gradients don't flow through them)

LoRA adapters (trained):
  Stored and computed in bfloat16
  Gradients flow through them normally
```

## Why four bits

The standard float32 representation uses 32 bits per weight. Bfloat16
uses 16 bits. Four bit quantization uses 4 bits. Each reduction halves
both memory usage and the time to read weights from memory.

```
Float32 model (7B params):    28 GB  (7B × 4 bytes)
Bfloat16 model (7B params):   14 GB  (7B × 2 bytes)
4-bit quantized (7B params):   3.5 GB (7B × 0.5 bytes)
Plus LoRA adapters:           ~0.1 GB (negligible)
```

The memory savings are dramatic. A model that would not fit on a 24
gigabyte RTX 3090 in bfloat16 fits easily in 4-bit with room to spare
for training batches.

## How the quantization works

Quantization maps floating point values to a small set of integer
values. The simplest quantization is rounding to the nearest integer.
But weights in a neural network are not uniformly distributed. Some
layers have weights clustered near zero. Others have weights spread
across a wide range.

QLoRA uses a technique called NormalFloat4. It assumes weights follow
a normal distribution and assigns more quantization levels near zero
where most weights live and fewer levels at the extremes where few
weights live.

```
Standard uniform quantization (2-bit example):
  0.00 ... 0.25 ... 0.50 ... 0.75 ... 1.00
  Each bin gets equal width.

NormalFloat quantization (2-bit example):
  0.00 ......... 0.10 . 0.40 ......... 1.00
  More bins near zero. Fewer bins at extremes.
```

The result is less quantization error for the same number of bits.
Most weights are near zero and get high precision. The few outlier
weights at the extremes get lower precision. This matches the actual
distribution of pretrained model weights.

## A tiny code example

Here is a simplified version of how QLoRA quantizes and dequantizes
weights during training.

```python
import torch

def quantize_4bit(weights, block_size=64):
    """
    Quantize a weight matrix to 4 bits.
    Works on blocks of weights for better accuracy.
    """
    out_features, in_features = weights.shape
    quantized = torch.zeros(out_features, in_features // 2, dtype=torch.uint8)

    for i in range(0, out_features):
        for j_start in range(0, in_features, block_size):
            j_end = min(j_start + block_size, in_features)
            block = weights[i, j_start:j_end]

            # Find min and max of this block
            w_min = block.min()
            w_max = block.max()

            # Quantize to 4 bits (16 levels)
            scale = (w_max - w_min) / 15.0
            zero_point = -w_min / scale

            quantized_block = torch.clamp(
                torch.round(block / scale + zero_point), 0, 15
            ).to(torch.uint8)

            # Pack two 4-bit values into one 8-bit byte
            for j in range(0, len(quantized_block), 2):
                byte_idx = j_start // 2 + j // 2
                high = quantized_block[j] << 4
                low = quantized_block[j + 1] if j + 1 < len(quantized_block) else 0
                quantized[i, byte_idx] = high | low

    return quantized


def dequantize_4bit(quantized, original_shape, block_size=64):
    """
    Reconstruct approximate float values from 4-bit quantized weights.
    This simplified version does not store scales per block.
    In real QLoRA the scale and zero point are stored alongside.
    """
    out_features, packed_in_features = quantized.shape
    in_features = packed_in_features * 2
    reconstructed = torch.zeros(out_features, min(in_features, original_shape[1]))

    for i in range(out_features):
        for j in range(0, min(in_features, original_shape[1]), 2):
            byte_idx = j // 2
            packed = quantized[i, byte_idx].item()
            high = (packed >> 4) & 0xF
            low = packed & 0xF
            if j < original_shape[1]:
                reconstructed[i, j] = float(high)
            if j + 1 < original_shape[1]:
                reconstructed[i, j + 1] = float(low)

    return reconstructed


# Example: quantize and dequantize a small weight matrix
weights = torch.randn(4, 16) * 0.5
quantized = quantize_4bit(weights, block_size=8)
reconstructed = dequantize_4bit(quantized, weights.shape, block_size=8)

# In real QLoRA the scale and offset per block are stored
# so reconstructed values are much closer to originals
print(f"Original shape: {weights.shape}")
print(f"Quantized shape: {quantized.shape}")
print(f"Compression ratio: {weights.numel() * 4 / quantized.numel():.1f}x")
```

In a real QLoRA implementation the scale and zero point are stored
alongside the quantized values for each block. During dequantization
the integer values are converted back using their block specific scale
and zero point. This gives much higher accuracy than the simplified
version above which loses the per block metadata.

## Double quantization

QLoRA applies a second round of quantization to the quantization
constants themselves. Each block of weights has a scaling factor that
converts between the integer representation and the floating point
representation. These scaling factors are stored as float32 by default.
Double quantization compresses them to eight bits.

The savings from double quantization are small. About 0.13 bits per
parameter on average. But for a seven billion parameter model that is
about 120 megabytes. Enough to matter when fitting a model into a GPU
with tight memory.

## Paged optimizers

Training with QLoRA uses an optimizer that manages GPU memory like an
operating system manages RAM. When the GPU runs out of memory the
optimizer pages some optimizer states to CPU RAM. When those states
are needed again they are paged back to GPU memory.

This prevents out-of-memory errors during training. Without paging a
memory spike from a large gradient calculation could crash training.
With paging the optimizer shuffles data between GPU and CPU to handle
the spike.

The technique is borrowed from operating systems where virtual memory
allows processes to use more memory than physically available by
swapping pages to disk. The same idea applied to GPU memory management
during training.

## The tradeoff

QLoRA is slightly slower than LoRA because of the dequantization
overhead. Every forward and backward pass must convert the base weights
from four bits to sixteen bits and back. This adds computation time.
But the memory savings usually outweigh the speed penalty. A model that
cannot be trained at all in bfloat16 can be trained successfully with
QLoRA.

The quality is slightly lower than LoRA. The quantization introduces
noise. Some information is lost in the conversion to four bits. But the
loss is small. For most tasks the difference between QLoRA and LoRA is
within a few percent on standard benchmarks.

## What you can run where

```
16 GB GPU (RTX 4080 laptop):
  LoRA: fine-tune up to ~7B models
  QLoRA: fine-tune up to ~13B models

24 GB GPU (RTX 3090/4090):
  LoRA: fine-tune up to ~13B models
  QLoRA: fine-tune up to ~34B models

12 GB GPU (RTX 3060/4060):
  LoRA: fine-tune up to ~3B models
  QLoRA: fine-tune up to ~7B models

8 GB GPU (RTX 2070):
  LoRA: fine-tune up to ~1B models
  QLoRA: fine-tune up to ~3B models
```

## A real QLoRA training snippet

Using the popular bitsandbytes library which implements QLoRA.

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",            # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16, # Compute in bfloat16
    bnb_4bit_use_double_quant=True,        # Enable double quantization
)

# Load a 7B model in 4-bit (~4 GB instead of ~14 GB)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# Add LoRA on top (using peft library)
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 8,388,608 || all params: 6,746,726,400 || trainable%: 0.12%

# Now train as usual. Only the LoRA params update.
# The 4-bit base model stays frozen.
```

## What you need to remember

QLoRA compresses the frozen base model to four bits. LoRA adapters
stay in full precision. The combination lets you fine-tune models that
would otherwise not fit in GPU memory. NormalFloat4 quantization gives
better accuracy than uniform quantization for the same bit budget.
Double quantization squeezes a little more memory from the quantization
constants. Paged optimizers prevent out-of-memory crashes. QLoRA is the
standard approach for fine-tuning on consumer hardware.

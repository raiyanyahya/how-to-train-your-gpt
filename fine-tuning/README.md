# Fine-Tuning: Making the Model Useful

This folder covers everything about adapting a pretrained language model
for a specific task. Pretraining teaches the model language. Fine-tuning
teaches it to be helpful.

## What's inside

| File | Topic |
|---|---|
| [01_what_is_finetuning.md](01_what_is_finetuning.md) | The concept. Why we fine-tune. Types: full, LoRA, QLoRA. |
| [02_lora_explained.md](02_lora_explained.md) | LoRA deep dive. Low-rank decomposition. The math in simple terms. |
| [03_qlora_explained.md](03_qlora_explained.md) | QLoRA. Quantization plus LoRA. Run on a laptop. |
| [04_data_preparation.md](04_data_preparation.md) | How to format data for instruction tuning. Chat templates. |
| [05_full_finetune.md](05_full_finetune.md) | Full fine-tuning. When and why not. |
| [06_dpo_explained.md](06_dpo_explained.md) | DPO. Preference optimization without RL. |
| [07_prompt_vs_finetune.md](07_prompt_vs_finetune.md) | When to prompt engineer vs when to fine-tune. |

## Notebook

[`notebooks/lora_finetune.ipynb`](notebooks/lora_finetune.ipynb). A
runnable notebook that fine-tunes a small model with LoRA on a toy
instruction dataset. Runs on a single consumer GPU.

## Reading order

Start with 01 for the big picture. Then 02 to understand LoRA which is
what almost everyone uses. 03 adds quantization for even smaller GPUs.
04 shows you how to prepare your data. 05 covers the full approach.
06 explains DPO the simpler alternative to RLHF. 07 helps you decide
between prompting and fine-tuning for your own use case.

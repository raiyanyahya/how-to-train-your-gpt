import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import tiktoken
from datasets import load_dataset
import matplotlib.pyplot as plt


# ===== ROTARY POSITION EMBEDDINGS (RoPE) =====
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=2048, theta=10000.0):
        super().__init__()
        assert d_model % 2 == 0
        dim_indices = torch.arange(0, d_model, 2).float()
        inv_freq = 1.0 / (theta ** (dim_indices / d_model))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    @staticmethod
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x, seq_len):
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        return (x * cos) + (self.rotate_half(x) * sin)


# ===== CAUSAL MASK =====
def create_causal_mask(seq_len, device):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.view(1, 1, seq_len, seq_len)


# ===== RMS NORMALIZATION =====
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# ===== SwiGLU FEED-FORWARD =====
class SwiGLU(nn.Module):
    def __init__(self, d_model, expansion_factor=4):
        super().__init__()
        hidden_dim = expansion_factor * d_model
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# ===== MULTI-HEAD ATTENTION =====
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rotary = RotaryPositionalEmbedding(self.head_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.rotary(q, seq_len)
        k = self.rotary(k, seq_len)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)
        return output


# ===== TRANSFORMER BLOCK =====
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)

    def forward(self, x, mask=None):
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ===== GPT CONFIGURATION =====
@dataclass
class GPTConfig:
    vocab_size: int = 50257
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 4
    max_seq_len: int = 128
    dropout: float = 0.1
    embd_dropout: float = 0.1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 50
    max_steps: int = 500
    batch_size: int = 4
    grad_accum_steps: int = 2
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8


# ===== GPT MODEL =====
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embd_dropout = nn.Dropout(config.embd_dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.num_heads, config.dropout)
            for _ in range(config.num_layers)
        ])
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids, targets=None):
        batch_size, seq_len = input_ids.shape
        x = self.token_embedding(input_ids)
        x = self.embd_dropout(x)
        mask = create_causal_mask(seq_len, input_ids.device)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            logits_flat = logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size)
            targets_flat = targets[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            if input_ids.shape[1] > self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]
            logits, _ = self.forward(input_ids)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


# ===== TOKENIZER =====
@dataclass
class TokenizerConfig:
    name: str = "gpt2"
    vocab_size: int = 50257


class SimpleTokenizer:
    def __init__(self, config=None):
        self.config = config or TokenizerConfig()
        self.enc = tiktoken.get_encoding(self.config.name)
        self.eos_token = "<|endoftext|>"
        self.eos_token_id = self.enc.encode(
            self.eos_token, allowed_special={self.eos_token}
        )[0]

    def encode(self, text):
        return self.enc.encode(text, allowed_special={self.eos_token})

    def decode(self, ids):
        return self.enc.decode(ids)

    @property
    def vocab_size(self):
        return self.config.vocab_size


# ===== DATASET =====
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_seq_len=128):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
            all_tokens.append(tokenizer.eos_token_id)
        self.tokens = torch.tensor(all_tokens, dtype=torch.long)

    def __len__(self):
        return (len(self.tokens) - 1) // self.max_seq_len

    def __getitem__(self, idx):
        start = idx * self.max_seq_len
        end = start + self.max_seq_len
        return self.tokens[start:end], self.tokens[start + 1 : end + 1]


def load_training_data(max_samples=None):
    print("Loading dataset: wikitext-103-raw-v1...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    texts = [item["text"] for item in dataset if item["text"].strip()]
    if max_samples:
        texts = texts[:max_samples]
    print(f"Loaded {len(texts):,} documents")
    return texts


# ===== LEARNING RATE SCHEDULER =====
class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, max_steps, max_lr=3e-4, min_lr=1e-5):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0

    def get_lr(self):
        step = self.current_step
        if step < self.warmup_steps:
            return self.max_lr * step / self.warmup_steps
        if step < self.max_steps:
            progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
        return self.min_lr

    def step(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.get_lr()
        self.current_step += 1


# ===== OPTIMIZER =====
def create_optimizer(model, config):
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() <= 1 or 'norm' in name.lower() or 'bias' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=config.learning_rate, betas=config.betas, eps=config.eps)


# ===== TRAINING =====
def train(model, train_dataset, config, device, steps=500):
    model = model.to(device)
    model.train()
    dataloader = DataLoader(train_dataset, batch_size=config.batch_size,
                            shuffle=True, drop_last=True)
    optimizer = create_optimizer(model, config)
    scheduler = CosineWarmupScheduler(optimizer, config.warmup_steps,
                                       steps, max_lr=config.learning_rate)
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None
    step = 0
    loss_history = []
    start = time.time()

    while step < steps:
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            if step >= steps:
                break
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                _, loss = model(input_ids, target_ids)
            loss = loss / config.grad_accum_steps
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (batch_idx + 1) % config.grad_accum_steps == 0:
                if use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if use_amp and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                step += 1
                if step % 50 == 0 or step == 1:
                    elapsed = time.time() - start
                    print(f"Step {step:>4d} | Loss: {loss.item() * config.grad_accum_steps:.4f} | Time: {elapsed:.0f}s")
                    loss_history.append((step, loss.item() * config.grad_accum_steps))
    print(f"\nDone. {time.time() - start:.0f}s total.")
    return loss_history


def plot_loss(loss_history, save_path="loss_curve.png"):
    steps, losses = zip(*loss_history)
    plt.figure(figsize=(10, 4))
    plt.plot(steps, losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"Loss curve saved to {save_path}")


# ===== MAIN =====
def main():
    print("How to Train Your GPT\n")

    # TINY MODEL (works on CPU, ~2-5 minutes)
    config = GPTConfig(
        d_model=256, num_heads=4, num_layers=4, max_seq_len=128,
        batch_size=4, grad_accum_steps=2, max_steps=500,
        warmup_steps=50, learning_rate=3e-4,
    )

    # SMALL MODEL (GPT-2 scale, needs GPU)
    # config = GPTConfig(
    #     d_model=768, num_heads=12, num_layers=12, max_seq_len=1024,
    #     batch_size=4, grad_accum_steps=8, max_steps=50000,
    #     warmup_steps=2000, learning_rate=3e-4,
    # )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    tokenizer = SimpleTokenizer()
    print("Loading training data...")
    texts = load_training_data(max_samples=5000)
    train_dataset = TextDataset(texts, tokenizer, max_seq_len=config.max_seq_len)

    print("Creating model...")
    model = GPT(config)
    print(f"Parameters: {model.get_num_params():,}")

    print("\n" + "=" * 50)
    print("TRAINING")
    print("=" * 50)
    loss_history = train(model, train_dataset, config, device, steps=config.max_steps)
    plot_loss(loss_history)

    print("\n" + "=" * 50)
    print("GENERATING")
    print("=" * 50 + "\n")

    prompts = [
        "The history of artificial intelligence",
        "In the beginning the universe",
        "The most important scientific discovery",
    ]

    for prompt in prompts:
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        output_ids = model.generate(input_ids, max_new_tokens=50, temperature=0.8, top_k=50)
        text = tokenizer.decode(output_ids[0].tolist())
        print(f"Prompt: {prompt}")
        print(f"Output: {text}")
        print("-" * 50)
        print()

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
    }, "checkpoints/model.pt")
    print("Model saved to checkpoints/model.pt")


if __name__ == "__main__":
    main()

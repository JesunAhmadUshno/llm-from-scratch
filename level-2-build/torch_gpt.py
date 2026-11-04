"""PyTorch track: the Level 1 TinyGPT architecture in PyTorch.

Same model as level-1-learn/gpt_numpy.py (token and position
embeddings, causal multi-head attention, LayerNorm, GELU MLP,
residual blocks, language model head), written the framework way:
autograd, nn.Module, Adam from torch.optim. Checkpointing saves the
model, the optimizer, and the step count, and the training driver
restarts from that checkpoint mid-run.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (BATCH_SIZE, CONFIG, LEARNING_RATE, RESTART_AT, SEED,
                    TOTAL_STEPS, get_batch, mean_last, split_data)


class Block(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg["n_embd"])
        self.attn = nn.MultiheadAttention(cfg["n_embd"], cfg["n_head"],
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(cfg["n_embd"])
        self.mlp = nn.Sequential(
            nn.Linear(cfg["n_embd"], 4 * cfg["n_embd"]), nn.GELU(),
            nn.Linear(4 * cfg["n_embd"], cfg["n_embd"]))

    def forward(self, x):
        seq = x.shape[1]
        mask = torch.triu(
            torch.full((seq, seq), float("-inf"), device=x.device),
            diagonal=1)
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x


class TorchGPT(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab_size, cfg["n_embd"])
        self.pos_emb = nn.Embedding(cfg["block_size"], cfg["n_embd"])
        self.blocks = nn.ModuleList(
            [Block(cfg, vocab_size) for _ in range(cfg["n_layer"])])
        self.ln_f = nn.LayerNorm(cfg["n_embd"])
        self.head = nn.Linear(cfg["n_embd"], vocab_size)

    def forward(self, idx):
        pos = torch.arange(idx.shape[1], device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    def loss_on(self, x, y):
        logits = self.forward(x)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               y.reshape(-1))

    @torch.no_grad()
    def generate(self, ids, max_new, greedy=True, temperature=1.0, top_k=None):
        self.eval()
        idx = torch.tensor([ids], dtype=torch.long)
        for _ in range(max_new):
            window = idx[:, -self.cfg["block_size"]:]
            logits = self.forward(window)[0, -1]
            if greedy:
                nxt = int(torch.argmax(logits))
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                if top_k:
                    vals, _ = torch.topk(probs, top_k)
                    probs = torch.where(probs < vals[-1], 0.0, probs)
                    probs = probs / probs.sum()
                nxt = int(torch.multinomial(probs, 1))
            idx = torch.cat([idx, torch.tensor([[nxt]])], dim=1)
        return idx[0].tolist()


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(path, model, optimizer, step):
    torch.save({"model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "step": step}, path)


def load_checkpoint(path, model, optimizer):
    blob = torch.load(path, weights_only=False)
    model.load_state_dict(blob["model"])
    optimizer.load_state_dict(blob["optimizer"])
    return int(blob["step"])


@torch.no_grad()
def evaluate(model, data, batches=10):
    model.eval()
    rng = np.random.default_rng(999)
    losses = []
    for _ in range(batches):
        x, y = get_batch(data, rng)
        losses.append(float(model.loss_on(torch.tensor(x),
                                          torch.tensor(y))))
    model.train()
    return float(np.mean(losses))


def train_torch(text, tokenizer, checkpoint_path, steps=TOTAL_STEPS,
                restart_at=RESTART_AT, seed=SEED, log_every=50):
    torch.manual_seed(seed)
    train_data, val_data = split_data(text, tokenizer)
    cfg = dict(CONFIG, vocab_size=tokenizer.vocab_size)
    model = TorchGPT(CONFIG, tokenizer.vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    rng = np.random.default_rng(seed)
    losses = []
    restarted = False
    val_before = evaluate(model, val_data)
    start = time.perf_counter()
    step = 0
    while step < steps:
        x, y = get_batch(train_data, rng)
        loss = model.loss_on(torch.tensor(x), torch.tensor(y))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
        step += 1
        if step == restart_at and restart_at < steps:
            save_checkpoint(checkpoint_path, model, optimizer, step)
            model = TorchGPT(CONFIG, tokenizer.vocab_size)
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=LEARNING_RATE)
            step = load_checkpoint(checkpoint_path, model, optimizer)
            restarted = True
    elapsed = time.perf_counter() - start
    val_after = evaluate(model, val_data)
    result = {
        "framework": "pytorch " + torch.__version__,
        "params": count_params(model),
        "steps": steps,
        "restarted_at": restart_at if restarted else None,
        "loss_step_1": round(losses[0], 4),
        "loss_mean_last_20": round(mean_last(losses), 4),
        "val_loss_before": round(val_before, 4),
        "val_loss_after": round(val_after, 4),
        "seconds": round(elapsed, 2),
        "tokens_per_second": round(
            steps * BATCH_SIZE * CONFIG["block_size"] / elapsed, 1),
        "loss_curve": {str(s): round(losses[s - 1], 4)
                       for s in range(log_every, steps + 1, log_every)},
    }
    prompt = tokenizer.encode("HAMLET.\n")
    sample_ids = model.generate(prompt, 40, greedy=True)
    result["sample_greedy"] = tokenizer.decode(sample_ids)
    return result, model

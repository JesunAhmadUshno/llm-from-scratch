"""Level 1, Learn: a tiny GPT in pure NumPy. No framework, nothing hidden.

Every piece a GPT needs is here in plain array code, so a reader can
follow each operation and check the maths by hand or by finite
differences (the test suite does exactly that):

- a character tokenizer (encode, decode, roundtrip),
- token and position embeddings,
- causal multi-head self-attention with an explicit future mask,
- transformer blocks (LayerNorm, attention, GELU MLP, residuals),
- a language model head and mean cross-entropy loss,
- full manual backpropagation through every layer,
- an Adam optimiser and a training loop,
- sampling with greedy, temperature, and top-k controls.

Framework ports (the same model in PyTorch, then in JAX with Flax,
per PROPOSAL.md) are Build level work: they re-express this exact
architecture and compare numbers. Nothing here depends on a GPU.
"""

import math

import numpy as np

# ---------------------------------------------------------------- tokenizer

class CharTokenizer:
    """Character level tokenizer built from a corpus string."""

    def __init__(self, corpus):
        chars = sorted(set(corpus))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


# ---------------------------------------------------------------- helpers

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


_GELU_C = math.sqrt(2.0 / math.pi)
_GELU_A = 0.044715


def gelu(x):
    """GELU, tanh approximation (the GPT-2 form). Fully vectorised."""
    return 0.5 * x * (1.0 + np.tanh(_GELU_C * (x + _GELU_A * x ** 3)))


def gelu_grad(x):
    u = _GELU_C * (x + _GELU_A * x ** 3)
    t = np.tanh(u)
    du = _GELU_C * (1.0 + 3.0 * _GELU_A * x ** 2)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du


def layernorm_forward(x, g, b, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    xhat = (x - mean) / std
    return xhat * g + b, (x, xhat, mean, std, g)


def layernorm_backward(dout, cache):
    x, xhat, mean, std, g = cache
    n = x.shape[-1]
    dg = np.sum(dout * xhat, axis=tuple(range(dout.ndim - 1)))
    db = np.sum(dout, axis=tuple(range(dout.ndim - 1)))
    dxhat = dout * g
    inv = 1.0 / std
    dvar = np.sum(dxhat * (x - mean) * -0.5 * inv ** 3, axis=-1, keepdims=True)
    dmean = np.sum(dxhat * -inv, axis=-1, keepdims=True) + dvar * np.mean(
        -2.0 * (x - mean), axis=-1, keepdims=True
    )
    dx = dxhat * inv + dvar * 2.0 * (x - mean) / n + dmean / n
    return dx, dg, db


# ---------------------------------------------------------------- model

class TinyGPT:
    """Decoder-only transformer. Config is a plain dict."""

    def __init__(self, cfg, seed=0):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        V, T, C, L = cfg["vocab_size"], cfg["block_size"], cfg["n_embd"], cfg["n_layer"]
        p = {}

        def w(*shape, scale=0.02):
            return rng.normal(0.0, scale, size=shape)

        p["wte"] = w(V, C, scale=0.1)
        p["wpe"] = w(T, C, scale=0.1)
        for i in range(L):
            k = f"l{i}."
            p[k + "ln1_g"] = np.ones(C)
            p[k + "ln1_b"] = np.zeros(C)
            p[k + "wqkv"] = w(C, 3 * C)
            p[k + "bqkv"] = np.zeros(3 * C)
            p[k + "wo"] = w(C, C)
            p[k + "bo"] = np.zeros(C)
            p[k + "ln2_g"] = np.ones(C)
            p[k + "ln2_b"] = np.zeros(C)
            p[k + "w1"] = w(C, 4 * C)
            p[k + "b1"] = np.zeros(4 * C)
            p[k + "w2"] = w(4 * C, C)
            p[k + "b2"] = np.zeros(C)
        p["lnf_g"] = np.ones(C)
        p["lnf_b"] = np.zeros(C)
        p["lm_head"] = w(V, C, scale=0.1)
        self.params = p

    # ------------------------------------------------------------ forward
    def forward(self, idx):
        """idx: (B, T) int array. Returns logits (B, T, V) and a cache."""
        p, cfg = self.params, self.cfg
        B, T = idx.shape
        C, H = cfg["n_embd"], cfg["n_head"]
        D = C // H
        cache = {"idx": idx, "blocks": []}
        x = p["wte"][idx] + p["wpe"][:T][None, :, :]
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        for i in range(cfg["n_layer"]):
            k = f"l{i}."
            bc = {}
            h, bc["ln1"] = layernorm_forward(x, p[k + "ln1_g"], p[k + "ln1_b"])
            bc["h1"] = h
            qkv = h @ p[k + "wqkv"] + p[k + "bqkv"]
            q, kk, v = np.split(qkv, 3, axis=-1)
            def split_heads(t):
                return t.reshape(B, T, H, D).transpose(0, 2, 1, 3)
            q, kk, v = split_heads(q), split_heads(kk), split_heads(v)
            att = (q @ kk.transpose(0, 1, 3, 2)) / math.sqrt(D)
            att = np.where(mask[None, None, :, :], -1e10, att)
            probs = softmax(att, axis=-1)
            heads = probs @ v
            concat = heads.transpose(0, 2, 1, 3).reshape(B, T, C)
            attn_out = concat @ p[k + "wo"] + p[k + "bo"]
            x = x + attn_out
            bc.update({"q": q, "k": kk, "v": v, "probs": probs,
                       "concat": concat, "x_mid": x})
            h2, bc["ln2"] = layernorm_forward(x, p[k + "ln2_g"], p[k + "ln2_b"])
            bc["h2"] = h2
            hidden_pre = h2 @ p[k + "w1"] + p[k + "b1"]
            hidden = gelu(hidden_pre)
            bc["hidden_pre"] = hidden_pre
            bc["hidden"] = hidden
            x = x + (hidden @ p[k + "w2"] + p[k + "b2"])
            cache["blocks"].append(bc)
        xf, cache["lnf"] = layernorm_forward(x, p["lnf_g"], p["lnf_b"])
        cache["xf"] = xf
        logits = xf @ p["lm_head"].T
        return logits, cache

    def loss(self, idx, targets):
        logits, _ = self.forward(idx)
        return cross_entropy(logits, targets)

    # ----------------------------------------------------------- backward
    def backward(self, cache, dlogits):
        """Full manual backprop. Returns a grads dict matching params."""
        p, cfg = self.params, self.cfg
        idx = cache["idx"]
        B, T = idx.shape
        C, H = cfg["n_embd"], cfg["n_head"]
        D = C // H
        grads = {k: np.zeros_like(v) for k, v in p.items()}
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)

        grads["lm_head"] = np.einsum("btv,btc->vc", dlogits, cache["xf"])
        dxf = dlogits @ p["lm_head"]
        dx, dg, db = layernorm_backward(dxf, cache["lnf"])
        grads["lnf_g"], grads["lnf_b"] = dg, db

        for i in reversed(range(cfg["n_layer"])):
            k = f"l{i}."
            bc = cache["blocks"][i]
            # MLP branch: x_out = x_mid + mlp(ln2(x_mid))
            dhidden = dx @ p[k + "w2"].T
            grads[k + "w2"] = np.einsum("btc,btd->cd", bc["hidden"], dx)
            grads[k + "b2"] = dx.sum(axis=(0, 1))
            dpre = dhidden * gelu_grad(bc["hidden_pre"])
            grads[k + "w1"] = np.einsum("btc,btd->cd", bc["h2"], dpre)
            grads[k + "b1"] = dpre.sum(axis=(0, 1))
            dh2 = dpre @ p[k + "w1"].T
            dx_ln2, dg2, db2 = layernorm_backward(dh2, bc["ln2"])
            grads[k + "ln2_g"], grads[k + "ln2_b"] = dg2, db2
            dx = dx + dx_ln2
            # attention branch: x_mid = x_in + attn(ln1(x_in))
            dconcat = dx @ p[k + "wo"].T
            grads[k + "wo"] = np.einsum("btc,btd->cd", bc["concat"], dx)
            grads[k + "bo"] = dx.sum(axis=(0, 1))
            dheads = dconcat.reshape(B, T, H, D).transpose(0, 2, 1, 3)
            probs, q, kk, v = bc["probs"], bc["q"], bc["k"], bc["v"]
            dv = probs.transpose(0, 1, 3, 2) @ dheads
            dprobs = dheads @ v.transpose(0, 1, 3, 2)
            datt = probs * (dprobs - np.sum(dprobs * probs, axis=-1, keepdims=True))
            datt = np.where(mask[None, None, :, :], 0.0, datt)
            scale = 1.0 / math.sqrt(D)
            dq = (datt @ kk) * scale
            dkk = (datt.transpose(0, 1, 3, 2) @ q) * scale
            def merge_heads(t):
                return t.transpose(0, 2, 1, 3).reshape(B, T, C)
            dqkv = np.concatenate(
                [merge_heads(dq), merge_heads(dkk), merge_heads(dv)], axis=-1
            )
            grads[k + "wqkv"] = np.einsum("btc,btd->cd", bc["h1"], dqkv)
            grads[k + "bqkv"] = dqkv.sum(axis=(0, 1))
            dh1 = dqkv @ p[k + "wqkv"].T
            dx_ln1, dg1, db1 = layernorm_backward(dh1, bc["ln1"])
            grads[k + "ln1_g"], grads[k + "ln1_b"] = dg1, db1
            dx = dx + dx_ln1

        np.add.at(grads["wte"], idx, dx)
        grads["wpe"][:T] += dx.sum(axis=0)
        return grads


def cross_entropy(logits, targets):
    """Mean next-token cross entropy. logits (B,T,V), targets (B,T)."""
    probs = softmax(logits, axis=-1)
    B, T, _ = logits.shape
    picked = probs[np.arange(B)[:, None], np.arange(T)[None, :], targets]
    return float(-np.log(np.maximum(picked, 1e-12)).mean())


def loss_and_grads(model, idx, targets):
    logits, cache = model.forward(idx)
    B, T, V = logits.shape
    probs = softmax(logits, axis=-1)
    loss = cross_entropy(logits, targets)
    dlogits = probs.copy()
    dlogits[np.arange(B)[:, None], np.arange(T)[None, :], targets] -= 1.0
    dlogits /= B * T
    return loss, model.backward(cache, dlogits)


# ---------------------------------------------------------------- optimiser

class Adam:
    def __init__(self, params, lr=3e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# ---------------------------------------------------------------- sampling

def generate(model, prompt_ids, max_new, temperature=1.0, top_k=None,
             rng=None, greedy=False):
    """Autoregressive sampling with greedy, temperature, and top-k."""
    rng = rng or np.random.default_rng(0)
    ids = list(prompt_ids)
    block = model.cfg["block_size"]
    for _ in range(max_new):
        window = np.array([ids[-block:]], dtype=np.int64)
        logits, _ = model.forward(window)
        last = logits[0, -1].astype(np.float64)
        if greedy or temperature == 0.0:
            nxt = int(np.argmax(last))
        else:
            scaled = last / temperature
            if top_k is not None and 0 < top_k < len(scaled):
                cutoff = np.sort(scaled)[-top_k]
                scaled = np.where(scaled < cutoff, -1e10, scaled)
            probs = softmax(scaled)
            nxt = int(rng.choice(len(probs), p=probs))
        ids.append(nxt)
    return ids


def get_batch(data, block_size, batch_size, rng):
    starts = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[s:s + block_size] for s in starts])
    y = np.stack([data[s + 1:s + block_size + 1] for s in starts])
    return x.astype(np.int64), y.astype(np.int64)

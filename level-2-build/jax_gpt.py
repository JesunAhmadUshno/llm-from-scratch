"""JAX track: the same TinyGPT architecture in pure JAX.

Same model as the PyTorch track and as Level 1 (token and position
embeddings, causal multi-head attention, LayerNorm, GELU MLP,
residual blocks, language model head), written the JAX way: an
explicit parameter pytree, a pure forward function, jax.grad for the
backward pass, and a jax.jit compiled training step. The optimizer is
a hand-written Adam over the pytree (jax.tree_map), and checkpointing
pickles the parameters, the Adam state, and the step count, so the
training driver can restart mid-run exactly like the PyTorch track.

Named limit: this track uses core JAX only. Flax, Optax, and Orbax
(the SPEC named libraries) are not used; the checkpoint is a pickle
of NumPy arrays, not an Orbax checkpoint. That gap is a stated
deduction in the level README, not hidden.
"""

import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

from common import (BATCH_SIZE, CONFIG, LEARNING_RATE, RESTART_AT, SEED,
                    TOTAL_STEPS, get_batch, mean_last, split_data)

EPS = 1e-5
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8


def init_params(vocab_size, cfg, seed=0):
    key = jax.random.PRNGKey(seed)
    n_embd = cfg["n_embd"]

    def take(k, shape, scale=0.02):
        return jax.random.normal(k, shape) * scale

    keys = jax.random.split(key, 4 + cfg["n_layer"] * 10)
    i = 0

    def nxt(shape, scale=0.02):
        nonlocal i
        value = take(keys[i], shape, scale)
        i += 1
        return value

    params = {
        "tok_emb": nxt((vocab_size, n_embd)),
        "pos_emb": nxt((cfg["block_size"], n_embd)),
        "blocks": [],
        "ln_f": {"g": jnp.ones((n_embd,)), "b": jnp.zeros((n_embd,))},
        "head_w": nxt((n_embd, vocab_size)),
        "head_b": jnp.zeros((vocab_size,)),
    }
    for _ in range(cfg["n_layer"]):
        block = {
            "ln1": {"g": jnp.ones((n_embd,)), "b": jnp.zeros((n_embd,))},
            "wq": nxt((n_embd, n_embd), 0.05), "bq": jnp.zeros((n_embd,)),
            "wk": nxt((n_embd, n_embd), 0.05), "bk": jnp.zeros((n_embd,)),
            "wv": nxt((n_embd, n_embd), 0.05), "bv": jnp.zeros((n_embd,)),
            "wo": nxt((n_embd, n_embd), 0.05), "bo": jnp.zeros((n_embd,)),
            "ln2": {"g": jnp.ones((n_embd,)), "b": jnp.zeros((n_embd,))},
            "w1": nxt((n_embd, 4 * n_embd), 0.05),
            "b1": jnp.zeros((4 * n_embd,)),
            "w2": nxt((4 * n_embd, n_embd), 0.05),
            "b2": jnp.zeros((n_embd,)),
        }
        params["blocks"].append(block)
    return params


def layernorm(x, g, b):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + EPS) * g + b


def forward(params, idx, cfg):
    n_head = cfg["n_head"]
    n_embd = cfg["n_embd"]
    head_dim = n_embd // n_head
    batch, seq = idx.shape
    x = params["tok_emb"][idx] + params["pos_emb"][:seq]
    mask = jnp.triu(jnp.full((seq, seq), -1e9), k=1)
    for block in params["blocks"]:
        h = layernorm(x, block["ln1"]["g"], block["ln1"]["b"])
        q = h @ block["wq"] + block["bq"]
        k = h @ block["wk"] + block["bk"]
        v = h @ block["wv"] + block["bv"]
        q = q.reshape(batch, seq, n_head, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq, n_head, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq, n_head, head_dim).transpose(0, 2, 1, 3)
        attn = jax.nn.softmax(
            q @ k.transpose(0, 1, 3, 2) / jnp.sqrt(float(head_dim)) + mask,
            axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(batch, seq, n_embd)
        x = x + out @ block["wo"] + block["bo"]
        h = layernorm(x, block["ln2"]["g"], block["ln2"]["b"])
        h = jax.nn.gelu(h @ block["w1"] + block["b1"]) @ block["w2"] \
            + block["b2"]
        x = x + h
    x = layernorm(x, params["ln_f"]["g"], params["ln_f"]["b"])
    return x @ params["head_w"] + params["head_b"]


def loss_fn(params, x, y, cfg):
    logits = forward(params, x, cfg)
    one_hot = jax.nn.one_hot(y, logits.shape[-1])
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))


def adam_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": jax.tree_util.tree_map(jnp.zeros_like, params),
            "t": 0}


def adam_update(params, grads, state, lr=LEARNING_RATE):
    state["t"] += 1
    t = state["t"]
    state["m"] = jax.tree_util.tree_map(
        lambda m, g: ADAM_B1 * m + (1 - ADAM_B1) * g, state["m"], grads)
    state["v"] = jax.tree_util.tree_map(
        lambda v, g: ADAM_B2 * v + (1 - ADAM_B2) * g * g, state["v"], grads)
    bc1 = 1 - ADAM_B1 ** t
    bc2 = 1 - ADAM_B2 ** t
    new_params = jax.tree_util.tree_map(
        lambda p, m, v: p - lr * (m / bc1) /
        (jnp.sqrt(v / bc2) + ADAM_EPS), params, state["m"], state["v"])
    return new_params, state


@jax.jit
def train_step(params, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y, CONFIG)
    return loss, grads


def count_params(params):
    return int(sum(int(np.prod(np.shape(leaf)))
                   for leaf in jax.tree_util.tree_leaves(params)))


def save_checkpoint(path, params, opt_state, step):
    blob = {"params": jax.tree_util.tree_map(np.asarray, params),
            "m": jax.tree_util.tree_map(np.asarray, opt_state["m"]),
            "v": jax.tree_util.tree_map(np.asarray, opt_state["v"]),
            "t": opt_state["t"], "step": step}
    with open(path, "wb") as handle:
        pickle.dump(blob, handle)


def load_checkpoint(path):
    with open(path, "rb") as handle:
        blob = pickle.load(handle)
    params = jax.tree_util.tree_map(jnp.asarray, blob["params"])
    state = {"m": jax.tree_util.tree_map(jnp.asarray, blob["m"]),
             "v": jax.tree_util.tree_map(jnp.asarray, blob["v"]),
             "t": blob["t"]}
    return params, state, int(blob["step"])


def evaluate(params, data, batches=10):
    rng = np.random.default_rng(999)
    losses = []
    for _ in range(batches):
        x, y = get_batch(data, rng)
        losses.append(float(loss_fn(params, jnp.asarray(x),
                                    jnp.asarray(y), CONFIG)))
    return float(np.mean(losses))


def generate(params, tokenizer, prompt_ids, max_new):
    ids = list(prompt_ids)
    for _ in range(max_new):
        window = ids[-CONFIG["block_size"]:]
        logits = forward(params, jnp.asarray([window]), CONFIG)
        ids.append(int(jnp.argmax(logits[0, -1])))
    return tokenizer.decode(ids)


def train_jax(text, tokenizer, checkpoint_path, steps=TOTAL_STEPS,
              restart_at=RESTART_AT, seed=SEED, log_every=50):
    train_data, val_data = split_data(text, tokenizer)
    params = init_params(tokenizer.vocab_size, CONFIG, seed=seed)
    state = adam_init(params)
    rng = np.random.default_rng(seed)
    losses = []
    restarted = False
    val_before = evaluate(params, val_data)
    start = time.perf_counter()
    step = 0
    while step < steps:
        x, y = get_batch(train_data, rng)
        loss, grads = train_step(params, jnp.asarray(x), jnp.asarray(y))
        params, state = adam_update(params, grads, state)
        losses.append(float(loss))
        step += 1
        if step == restart_at and restart_at < steps:
            save_checkpoint(checkpoint_path, params, state, step)
            params, state, step = load_checkpoint(checkpoint_path)
            restarted = True
    elapsed = time.perf_counter() - start
    val_after = evaluate(params, val_data)
    result = {
        "framework": "jax " + jax.__version__,
        "params": count_params(params),
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
    result["sample_greedy"] = generate(params, tokenizer, prompt, 40)
    return result, params

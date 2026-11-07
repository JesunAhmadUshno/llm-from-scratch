// A JavaScript port of level-1-learn/gpt_numpy.py TinyGPT: token and position
// embeddings, causal multi-head attention, LayerNorm, GELU MLP, a language-model
// head, full hand-written backward pass, and Adam. Same layout and init scales
// as the NumPy file; the RNG is a seeded mulberry32, so runs are repeatable here
// but not bit-identical to NumPy. Float64 throughout, so the finite-difference
// gradient check can meet the repo's 1e-4 bar.

export function rng(seed) {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  next.normal = () => Math.sqrt(-2 * Math.log(1 - next())) * Math.cos(2 * Math.PI * next());
  next.int = (n) => Math.floor(next() * n);
  return next;
}

export function tokenizer(text) {
  // Code points, not UTF-16 units: corpus.txt has a curly apostrophe and an accented letter.
  const chars = [...new Set(Array.from(text))].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  const stoi = new Map(chars.map((c, i) => [c, i]));
  return {
    chars, vocabSize: chars.length,
    encode: (s) => Array.from(s).map((c) => stoi.get(c)),
    decode: (ids) => ids.map((i) => chars[i]).join(''),
    has: (c) => stoi.has(c),
  };
}

export const paramCount = ({ V, T, C, L, headBias = false }) => V * C + T * C + L * (12 * C * C + 13 * C) + 2 * C + C * V + (headBias ? V : 0);

export function init(cfg, seed = 7) {
  const { vocab: V, block: T, embd: C, layers: L } = cfg;
  const r = rng(seed);
  const p = {};
  const w = (n, s = 0.02) => Float64Array.from({ length: n }, () => r.normal() * s);
  const ones = (n) => new Float64Array(n).fill(1);
  const zeros = (n) => new Float64Array(n);
  p.wte = w(V * C, 0.1);
  p.wpe = w(T * C, 0.1);
  for (let i = 0; i < L; i++) {
    const k = `l${i}.`;
    p[k + 'ln1_g'] = ones(C); p[k + 'ln1_b'] = zeros(C);
    p[k + 'wqkv'] = w(C * 3 * C); p[k + 'bqkv'] = zeros(3 * C);
    p[k + 'wo'] = w(C * C); p[k + 'bo'] = zeros(C);
    p[k + 'ln2_g'] = ones(C); p[k + 'ln2_b'] = zeros(C);
    p[k + 'w1'] = w(C * 4 * C); p[k + 'b1'] = zeros(4 * C);
    p[k + 'w2'] = w(4 * C * C); p[k + 'b2'] = zeros(C);
  }
  p.lnf_g = ones(C); p.lnf_b = zeros(C);
  p.lm_head = w(V * C, 0.1);
  return { cfg: { ...cfg }, p };
}

// out[n,m] = a[n,k] @ w[k,m] + b[m]
function mm(a, w, b, n, k, m) {
  const out = new Float64Array(n * m);
  for (let i = 0; i < n; i++) {
    const o = i * m;
    if (b) for (let j = 0; j < m; j++) out[o + j] = b[j];
    for (let t = 0; t < k; t++) {
      const av = a[i * k + t], wo = t * m;
      if (av !== 0) for (let j = 0; j < m; j++) out[o + j] += av * w[wo + j];
    }
  }
  return out;
}
// Given dout[n,m] for out = a @ w + b: accumulate dw, db and return da.
function mmBack(dout, a, w, n, k, m, dw, db) {
  const da = new Float64Array(n * k);
  for (let i = 0; i < n; i++) {
    for (let t = 0; t < k; t++) {
      let s = 0;
      const av = a[i * k + t], wo = t * m, o = i * m;
      for (let j = 0; j < m; j++) { s += dout[o + j] * w[wo + j]; dw[wo + j] += av * dout[o + j]; }
      da[i * k + t] = s;
    }
    if (db) for (let j = 0; j < m; j++) db[j] += dout[i * m + j];
  }
  return da;
}
function lnFwd(x, g, b, n, C) {
  const out = new Float64Array(n * C), xhat = new Float64Array(n * C), rstd = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let mu = 0; for (let j = 0; j < C; j++) mu += x[i * C + j]; mu /= C;
    let v = 0; for (let j = 0; j < C; j++) { const d = x[i * C + j] - mu; v += d * d; } v /= C;
    const r = 1 / Math.sqrt(v + 1e-5); rstd[i] = r;
    for (let j = 0; j < C; j++) { const h = (x[i * C + j] - mu) * r; xhat[i * C + j] = h; out[i * C + j] = h * g[j] + b[j]; }
  }
  return { out, xhat, rstd };
}
function lnBack(dout, c, g, n, C, dg, db) {
  const dx = new Float64Array(n * C);
  for (let i = 0; i < n; i++) {
    let m1 = 0, m2 = 0;
    for (let j = 0; j < C; j++) {
      const d = dout[i * C + j], h = c.xhat[i * C + j];
      dg[j] += d * h; db[j] += d;
      const dh = d * g[j]; m1 += dh; m2 += dh * h;
    }
    m1 /= C; m2 /= C;
    for (let j = 0; j < C; j++) dx[i * C + j] = c.rstd[i] * (dout[i * C + j] * g[j] - m1 - c.xhat[i * C + j] * m2);
  }
  return dx;
}
const GC = Math.sqrt(2 / Math.PI), GA = 0.044715;
const gelu = (x) => 0.5 * x * (1 + Math.tanh(GC * (x + GA * x * x * x)));
const geluGrad = (x) => { const t = Math.tanh(GC * (x + GA * x * x * x)); return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * GC * (1 + 3 * GA * x * x); };

// Forward one sequence of ids (length T <= block). Returns logits [T*V] and a cache.
export function forward(model, ids, keepProbs = false) {
  const { p, cfg } = model;
  const { vocab: V, embd: C, heads: H, layers: L } = cfg;
  const T = ids.length, D = C / H, sc = 1 / Math.sqrt(D);
  let x = new Float64Array(T * C);
  for (let t = 0; t < T; t++) for (let j = 0; j < C; j++) x[t * C + j] = p.wte[ids[t] * C + j] + p.wpe[t * C + j];
  const blocks = [];
  for (let l = 0; l < L; l++) {
    const k = `l${l}.`, bc = {};
    bc.ln1 = lnFwd(x, p[k + 'ln1_g'], p[k + 'ln1_b'], T, C);
    const qkv = mm(bc.ln1.out, p[k + 'wqkv'], p[k + 'bqkv'], T, C, 3 * C);
    const probs = new Float64Array(H * T * T), concat = new Float64Array(T * C);
    for (let h = 0; h < H; h++) {
      for (let i = 0; i < T; i++) {
        const row = h * T * T + i * T;
        let mx = -Infinity;
        for (let j = 0; j <= i; j++) {
          let s = 0;
          for (let d = 0; d < D; d++) s += qkv[i * 3 * C + h * D + d] * qkv[j * 3 * C + C + h * D + d];
          probs[row + j] = s * sc; if (probs[row + j] > mx) mx = probs[row + j];
        }
        let z = 0;
        for (let j = 0; j <= i; j++) { probs[row + j] = Math.exp(probs[row + j] - mx); z += probs[row + j]; }
        for (let j = 0; j <= i; j++) {
          probs[row + j] /= z;
          for (let d = 0; d < D; d++) concat[i * C + h * D + d] += probs[row + j] * qkv[j * 3 * C + 2 * C + h * D + d];
        }
      }
    }
    const attn = mm(concat, p[k + 'wo'], p[k + 'bo'], T, C, C);
    const xMid = new Float64Array(T * C);
    for (let i = 0; i < T * C; i++) xMid[i] = x[i] + attn[i];
    bc.ln2 = lnFwd(xMid, p[k + 'ln2_g'], p[k + 'ln2_b'], T, C);
    const pre = mm(bc.ln2.out, p[k + 'w1'], p[k + 'b1'], T, C, 4 * C);
    const hid = pre.map(gelu);
    const mlp = mm(hid, p[k + 'w2'], p[k + 'b2'], T, 4 * C, C);
    const xOut = new Float64Array(T * C);
    for (let i = 0; i < T * C; i++) xOut[i] = xMid[i] + mlp[i];
    Object.assign(bc, { qkv, probs, concat, pre, hid });
    blocks.push(keepProbs ? { probs } : bc);
    x = xOut;
  }
  const lnf = lnFwd(x, p.lnf_g, p.lnf_b, T, C);
  const logits = new Float64Array(T * V);
  for (let t = 0; t < T; t++) for (let v = 0; v < V; v++) {
    let s = 0; for (let j = 0; j < C; j++) s += lnf.out[t * C + j] * p.lm_head[v * C + j];
    logits[t * V + v] = s;
  }
  return { logits, cache: { ids, blocks, lnf } };
}

// Mean cross-entropy over a batch of [x, y] sequences, plus gradients (when wanted).
export function lossAndGrads(model, xs, ys, wantGrads = true) {
  const { p, cfg } = model;
  const { vocab: V, embd: C, heads: H, layers: L } = cfg;
  const D = C / H, sc = 1 / Math.sqrt(D);
  const grads = wantGrads ? Object.fromEntries(Object.entries(p).map(([k, v]) => [k, new Float64Array(v.length)])) : null;
  const N = xs.length * xs[0].length;
  let loss = 0;
  for (let b = 0; b < xs.length; b++) {
    const ids = xs[b], T = ids.length;
    const { logits, cache } = forward(model, ids);
    const dlogits = new Float64Array(T * V);
    for (let t = 0; t < T; t++) {
      let mx = -Infinity; for (let v = 0; v < V; v++) mx = Math.max(mx, logits[t * V + v]);
      let z = 0; for (let v = 0; v < V; v++) z += Math.exp(logits[t * V + v] - mx);
      const y = ys[b][t];
      loss -= logits[t * V + y] - mx - Math.log(z);
      if (wantGrads) {
        for (let v = 0; v < V; v++) dlogits[t * V + v] = Math.exp(logits[t * V + v] - mx) / z / N;
        dlogits[t * V + y] -= 1 / N;
      }
    }
    if (!wantGrads) continue;
    const g = grads, xf = cache.lnf.out;
    let dxf = new Float64Array(T * C);
    for (let t = 0; t < T; t++) for (let v = 0; v < V; v++) {
      const d = dlogits[t * V + v]; if (d === 0) continue;
      for (let j = 0; j < C; j++) { g.lm_head[v * C + j] += d * xf[t * C + j]; dxf[t * C + j] += d * p.lm_head[v * C + j]; }
    }
    let dx = lnBack(dxf, cache.lnf, p.lnf_g, T, C, g.lnf_g, g.lnf_b);
    for (let l = L - 1; l >= 0; l--) {
      const k = `l${l}.`, bc = cache.blocks[l];
      const dhid = mmBack(dx, bc.hid, p[k + 'w2'], T, 4 * C, C, g[k + 'w2'], g[k + 'b2']);
      for (let i = 0; i < dhid.length; i++) dhid[i] *= geluGrad(bc.pre[i]);
      const dh2 = mmBack(dhid, bc.ln2.out, p[k + 'w1'], T, C, 4 * C, g[k + 'w1'], g[k + 'b1']);
      const dln2 = lnBack(dh2, bc.ln2, p[k + 'ln2_g'], T, C, g[k + 'ln2_g'], g[k + 'ln2_b']);
      for (let i = 0; i < dx.length; i++) dx[i] += dln2[i];
      const dconcat = mmBack(dx, bc.concat, p[k + 'wo'], T, C, C, g[k + 'wo'], g[k + 'bo']);
      const dqkv = new Float64Array(T * 3 * C), q = bc.qkv;
      for (let h = 0; h < H; h++) for (let i = 0; i < T; i++) {
        const row = h * T * T + i * T;
        const dp = new Float64Array(i + 1);
        let dot = 0;
        for (let j = 0; j <= i; j++) {
          let s = 0;
          for (let d = 0; d < D; d++) {
            const dh = dconcat[i * C + h * D + d];
            s += dh * q[j * 3 * C + 2 * C + h * D + d];
            dqkv[j * 3 * C + 2 * C + h * D + d] += bc.probs[row + j] * dh; // dv
          }
          dp[j] = s; dot += s * bc.probs[row + j];
        }
        for (let j = 0; j <= i; j++) {
          const da = bc.probs[row + j] * (dp[j] - dot) * sc;
          for (let d = 0; d < D; d++) {
            dqkv[i * 3 * C + h * D + d] += da * q[j * 3 * C + C + h * D + d]; // dq
            dqkv[j * 3 * C + C + h * D + d] += da * q[i * 3 * C + h * D + d]; // dk
          }
        }
      }
      const dh1 = mmBack(dqkv, bc.ln1.out, p[k + 'wqkv'], T, C, 3 * C, g[k + 'wqkv'], g[k + 'bqkv']);
      const dln1 = lnBack(dh1, bc.ln1, p[k + 'ln1_g'], T, C, g[k + 'ln1_g'], g[k + 'ln1_b']);
      for (let i = 0; i < dx.length; i++) dx[i] += dln1[i];
    }
    for (let t = 0; t < T; t++) for (let j = 0; j < C; j++) { g.wte[ids[t] * C + j] += dx[t * C + j]; g.wpe[t * C + j] += dx[t * C + j]; }
  }
  return { loss: loss / N, grads };
}

export function adam(params, lr, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
  const m = {}, v = {};
  for (const k in params) { m[k] = new Float64Array(params[k].length); v[k] = new Float64Array(params[k].length); }
  let t = 0;
  return (grads) => {
    t++;
    const c1 = 1 - b1 ** t, c2 = 1 - b2 ** t;
    for (const k in params) {
      const P = params[k], G = grads[k], M = m[k], S = v[k];
      for (let i = 0; i < P.length; i++) {
        M[i] = b1 * M[i] + (1 - b1) * G[i];
        S[i] = b2 * S[i] + (1 - b2) * G[i] * G[i];
        P[i] -= lr * (M[i] / c1) / (Math.sqrt(S[i] / c2) + eps);
      }
    }
  };
}

export function getBatch(data, block, batch, r) {
  const xs = [], ys = [];
  for (let b = 0; b < batch; b++) {
    const s = r.int(data.length - block - 1);
    xs.push(data.slice(s, s + block)); ys.push(data.slice(s + 1, s + block + 1));
  }
  return [xs, ys];
}

export function generate(model, prompt, maxNew, { greedy = true, temperature = 1, topK = 0, seed = 3 } = {}) {
  const r = rng(seed), ids = [...prompt], { vocab: V, block } = model.cfg;
  for (let n = 0; n < maxNew; n++) {
    const win = ids.slice(-block);
    const { logits } = forward(model, win, true);
    const last = Array.from(logits.subarray((win.length - 1) * V, win.length * V));
    if (greedy || temperature === 0) { ids.push(last.indexOf(Math.max(...last))); continue; }
    let s = last.map((x) => x / temperature);
    if (topK > 0 && topK < V) { const cut = [...s].sort((a, b) => b - a)[topK - 1]; s = s.map((x) => (x < cut ? -1e10 : x)); }
    const mx = Math.max(...s), e = s.map((x) => Math.exp(x - mx)), z = e.reduce((a, b) => a + b, 0);
    let u = r() * z, pick = V - 1;
    for (let i = 0; i < V; i++) { u -= e[i]; if (u <= 0) { pick = i; break; } }
    ids.push(pick);
  }
  return ids;
}

// Central finite differences against the analytic backward, on sampled entries of
// every parameter tensor (mirrors tests/test_learn_gpt.py). Returns the max relative error.
export function gradCheck(model, xs, ys, perTensor = 3, seed = 0) {
  const { grads } = lossAndGrads(model, xs, ys);
  const r = rng(seed), eps = 1e-5, rows = [];
  let zeros = 0;
  for (const [k, arr] of Object.entries(model.p)) {
    let worst = 0;
    for (let n = 0; n < perTensor; n++) {
      const i = r.int(arr.length), o = arr[i];
      arr[i] = o + eps; const plus = lossAndGrads(model, xs, ys, false).loss;
      arr[i] = o - eps; const minus = lossAndGrads(model, xs, ys, false).loss;
      arr[i] = o;
      const num = (plus - minus) / (2 * eps), ana = grads[k][i];
      // Both effectively zero (e.g. the key bias, whose gradient is exactly 0 because softmax
      // ignores a constant added to a whole score row): the ratio would only measure float noise.
      if (Math.abs(num) + Math.abs(ana) < 1e-7) { zeros++; continue; }
      worst = Math.max(worst, Math.abs(num - ana) / (Math.abs(num) + Math.abs(ana)));
    }
    rows.push([k, worst]);
  }
  return { maxRel: Math.max(...rows.map((x) => x[1])), checked: rows.length * perTensor, zeros, rows };
}

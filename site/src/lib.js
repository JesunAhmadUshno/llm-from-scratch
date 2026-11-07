// Pure helpers shared by the page and the build-time check (src/check.mjs).
// Everything here takes repo file contents as strings or parsed JSON.

// Rows of the first markdown table that follows a line containing `heading`.
export function tableAfter(md, heading) {
  const lines = md.split('\n');
  const at = lines.findIndex((l) => l.includes(heading));
  if (at < 0) throw new Error(`README table anchor not found: ${heading}`);
  const rows = [];
  let started = false;
  for (const l of lines.slice(at)) {
    if (l.trim().startsWith('|')) { started = true; rows.push(l.split('|').slice(1, -1).map((c) => c.trim())); }
    else if (started) break;
  }
  if (rows.length < 3) throw new Error(`README table under "${heading}" has no data rows`);
  return rows.slice(2); // drop header and --- rule
}
export const num = (s) => Number(String(s).replace(/,/g, ''));

export function learnTable(md) {
  return tableAfter(md, '| Step | Loss |').map(([s, l]) => ({ step: num(s), loss: num(l) }));
}
export function buildCurve(md) {
  return tableAfter(md, 'Loss curve (training loss at the step)').map(([s, a, b]) => ({ step: num(s), torch: num(a), jax: num(b) }));
}
export function buildMeasures(md) {
  return Object.fromEntries(tableAfter(md, 'Method: identical config').map(([k, a, b]) => [k, { torch: a, jax: b }]));
}
// Deep README: generation and prefill tables follow the "Config A:" / "Config B:" lines.
export function deepTables(md) {
  const out = {};
  for (const c of ['A', 'B']) {
    const sec = md.slice(md.indexOf(`Config ${c}: `));
    const gen = tableAfter(sec, `Config ${c}: `);
    const pre = tableAfter(sec, '| Prefill path');
    out[c] = { gen, pre };
  }
  return out;
}
export function wcLines(src) { return src.endsWith('\n') ? src.split('\n').length - 1 : src.split('\n').length; }

export function speedups(deep) {
  return deep.configs.map((c) => {
    const g = Object.fromEntries(c.generation.map((r) => [r.mode, r]));
    return g.cached.tokens_per_second / g.naive.tokens_per_second;
  });
}

// common.split_data semantics: first 90% of ids train, the rest validation.
export function split(ids, valFraction = 0.1) {
  const cut = Math.floor(ids.length * (1 - valFraction));
  return [ids.slice(0, cut), ids.slice(cut)];
}
// Unigram and add-k bigram cross-entropy (nats per char) fitted on train, scored on val.
export function ngramBaselines(train, val, V, k = 1) {
  const uni = new Float64Array(V).fill(k);
  for (const t of train) uni[t]++;
  const uz = uni.reduce((a, b) => a + b, 0);
  let u = 0;
  for (const t of val) u -= Math.log(uni[t] / uz);
  const bi = new Float64Array(V * V).fill(k), row = new Float64Array(V).fill(k * V);
  for (let i = 1; i < train.length; i++) { bi[train[i - 1] * V + train[i]]++; row[train[i - 1]]++; }
  let b = 0;
  for (let i = 1; i < val.length; i++) b -= Math.log(bi[val[i - 1] * V + val[i]] / row[val[i - 1]]);
  return { uniform: Math.log(V), unigram: u / val.length, bigram: b / (val.length - 1) };
}

// KV cache: per token, each layer stores one K and one V vector of n_embd floats.
export const cacheFloatsPerToken = (L, C) => 2 * L * C;
export const cacheBytes = (L, C, tokens, bytesPerFloat = 4) => cacheFloatsPerToken(L, C) * tokens * bytesPerFloat;
// Analytic attention dot products to generate n tokens after a prompt of p tokens (a model, not a measurement).
export function attentionOps(p, n) {
  let naive = 0, cached = 0;
  for (let t = p; t < p + n; t++) { naive += (t * (t + 1)) / 2; cached += t; }
  cached += (p * (p + 1)) / 2; // the prefill pass
  return { naive, cached };
}

// The Learn teaching corpus, read out of level-1-learn/train_tiny.py: the quoted phrases joined, times the repeat count.
export function learnCorpus(src) {
  const m = src.match(/CORPUS = \(\n([\s\S]*?)\) \* (\d+)/);
  if (!m) throw new Error('CORPUS block not found in train_tiny.py');
  return [...m[1].matchAll(/"([^"]*)"/g)].map((x) => x[1]).join('').repeat(+m[2]);
}
// The "(date, machine, versions)" text of a README's "## Measured results (...)" heading.
export function measuredOn(md) {
  const m = md.match(/^## Measured results \((.*)\)$/m);
  if (!m) throw new Error('Measured results heading not found');
  return m[1];
}

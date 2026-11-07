// Known-answer check, run by `npm run build` before vite. Fails the build if the
// page's in-browser computations stop reproducing the repo's recorded numbers,
// or if a README table the page parses goes missing or drifts from its JSON.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import * as L from './lib.js';
import * as G from './gpt.js';

const read = (p) => fs.readFileSync(fileURLToPath(new URL(`../../${p}`, import.meta.url)), 'utf8');
const json = (p) => JSON.parse(read(p));
const deep = json('level-3-deep/results_deep.json');
const learn = json('level-1-learn/results_learn.json');
const build = json('level-2-build/results_build.json');
const corpus = read('level-2-build/corpus.txt');
const tiny = read('level-1-learn/train_tiny.py');

// 1. Parameter-count formula reproduces all four recorded counts.
assert.equal(G.paramCount({ V: 20, T: 32, C: 64, L: 2 }), 104704);
assert.equal(G.paramCount({ V: 68, T: 32, C: 64, L: 2, headBias: true }), 110916);
assert.equal(G.paramCount({ V: 68, T: 128, C: 64, L: 2, headBias: true }), deep.configs[0].params);
assert.equal(G.paramCount({ V: 68, T: 128, C: 128, L: 4, headBias: true }), deep.configs[1].params);
assert.equal(learn.params, 104704);
assert.equal(build.pytorch.params, 110916);
assert.equal(deep.configs[1].params, 827204);

// 2. Speedups recompute from results_deep.json (2.34x and 3.69x quoted in the Deep README).
const [sA, sB] = L.speedups(deep);
assert.equal(sA.toFixed(2), '2.33'); // 664.0 / 284.4 = 2.3347. The Deep README prose says 2.34; the recorded numbers give 2.33 by either ratio.
assert.equal((deep.configs[0].generation[0].seconds_mean / deep.configs[0].generation[1].seconds_mean).toFixed(2), '2.33');
assert.equal(sB.toFixed(2), '3.69');

// 3. Deep README tables match results_deep.json, row by row.
const dt = L.deepTables(read('level-3-deep/README.md'));
deep.configs.forEach((c, i) => {
  const t = dt[i ? 'B' : 'A'];
  const g = Object.fromEntries(c.generation.map((r) => [r.mode, r]));
  const p = Object.fromEntries(c.prefill.map((r) => [r.path, r]));
  assert.deepEqual(t.gen[0].slice(1).map(L.num), [g.naive.seconds_mean, g.cached.seconds_mean]);
  assert.deepEqual(t.gen[1].slice(1).map(L.num), [g.naive.tokens_per_second, g.cached.tokens_per_second]);
  assert.deepEqual(t.pre.map((r) => r.slice(1).map(L.num)), [[p.hand.seconds_mean, p.hand.tokens_per_second], [p.sdpa.seconds_mean, p.sdpa.tokens_per_second]]);
});

// 4. Learn README table equals the rerun's results_learn.json (seeded run reproduced exactly).
const lt = L.learnTable(read('level-1-learn/README.md'));
assert.equal(lt.length, 7);
for (const r of lt) assert.equal(learn.losses[r.step - 1], r.loss, `learn step ${r.step}`);

// 5. Build README tables: present, complete, and PyTorch reproduced exactly by the rerun.
const bc = L.buildCurve(read('level-2-build/README.md'));
assert.deepEqual(bc.map((r) => r.step), [50, 100, 150, 200, 250, 300]);
for (const r of bc) assert.equal(build.pytorch.losses[r.step - 1], r.torch, `torch step ${r.step}`);
const bm = L.buildMeasures(read('level-2-build/README.md'));
assert.equal(L.num(bm['Validation loss, after training'].torch), build.pytorch.val_loss_after);
for (const k of ['Tokens per second', 'Validation loss, after training', 'Parameters', 'Train loss, step 1']) assert.ok(bm[k], `build row ${k}`);

// 6. Tokenizer: 68-char vocab from corpus.txt, by code point; 20 chars for the Learn corpus.
const tok = G.tokenizer(corpus);
assert.equal(tok.vocabSize, deep.vocab_size);
assert.equal(tok.vocabSize, build.vocab_size);
assert.equal(tok.decode(tok.encode(corpus)), corpus);
const learnText = L.learnCorpus(tiny);
assert.equal(G.tokenizer(learnText).vocabSize, learn.config.vocab_size);

for (const lv of ['level-1-learn', 'level-2-build', 'level-3-deep']) assert.match(L.measuredOn(read(`${lv}/README.md`)), /^2026-11-04, Muse VM/);

// 7. KV cache floats per token (Deep README: 256 for Config A, 1,024 for Config B).
assert.equal(L.cacheFloatsPerToken(2, 64), 256);
assert.equal(L.cacheFloatsPerToken(4, 128), 1024);

// 8. The JS model: parameter count, gradient check at the repo's 1e-4 bar, causality, and learning.
const lt2 = G.tokenizer(learnText), data = lt2.encode(learnText);
const m = G.init({ vocab: lt2.vocabSize, block: 6, embd: 12, heads: 2, layers: 2 }, 1);
assert.equal(Object.values(m.p).reduce((a, v) => a + v.length, 0), G.paramCount({ V: lt2.vocabSize, T: 6, C: 12, L: 2 }));
const [xs, ys] = G.getBatch(data, 6, 2, G.rng(5));
const gc = G.gradCheck(m, xs, ys);
assert.ok(gc.maxRel < 1e-4, `grad check ${gc.maxRel}`);
const a = G.forward(m, [1, 2, 3, 4]).logits, b = G.forward(m, [1, 2, 3, 9]).logits;
assert.deepEqual(a.subarray(0, 3 * lt2.vocabSize), b.subarray(0, 3 * lt2.vocabSize));
const m2 = G.init({ vocab: lt2.vocabSize, block: 16, embd: 32, heads: 4, layers: 1 }, 7), step = G.adam(m2.p, 5e-3), r = G.rng(42);
const losses = [];
for (let i = 0; i < 60; i++) { const [x, y] = G.getBatch(data, 16, 8, r); const o = G.lossAndGrads(m2, x, y); step(o.grads); losses.push(o.loss); }
assert.ok(losses.at(-1) < losses[0] * 0.75, `loss ${losses[0]} -> ${losses.at(-1)}`);
const [xt, yt] = G.getBatch(data, 16, 2, G.rng(5));
assert.ok(G.gradCheck(m2, xt, yt).maxRel < 1e-4, 'grad check after training');

// 9. N-gram baselines are finite and ordered uniform > unigram > bigram.
const [tr, va] = L.split(tok.encode(corpus));
const ng = L.ngramBaselines(tr, va, tok.vocabSize);
assert.ok(ng.uniform > ng.unigram && ng.unigram > ng.bigram && ng.bigram > 1);

console.log('check ok', { speedA: +sA.toFixed(4), speedB: +sB.toFixed(4), gradMaxRel: gc.maxRel, vocab: tok.vocabSize, ngram: ng, jsLoss: [+losses[0].toFixed(3), +losses.at(-1).toFixed(3)] });

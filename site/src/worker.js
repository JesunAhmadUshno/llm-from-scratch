// Runs the JS TinyGPT off the main thread so the page never blocks.
// Messages in: init, train, stop, gradcheck, sample, attention. Training yields
// between steps so a stop message is handled promptly.
import * as G from './gpt.js';

let model = null, tok = null, data = null, stopFlag = false;

const handlers = {
  init({ text, cfg, seed }) {
    tok = G.tokenizer(text);
    data = tok.encode(text);
    model = G.init({ ...cfg, vocab: tok.vocabSize }, seed);
    postMessage({ type: 'ready', params: Object.values(model.p).reduce((a, v) => a + v.length, 0), vocab: tok.vocabSize });
  },
  async train({ steps, batch, lr, seed }) {
    stopFlag = false;
    const step = G.adam(model.p, lr), r = G.rng(seed);
    const t0 = performance.now();
    for (let i = 1; i <= steps && !stopFlag; i++) {
      const [xs, ys] = G.getBatch(data, model.cfg.block, batch, r);
      const { loss, grads } = G.lossAndGrads(model, xs, ys);
      step(grads);
      postMessage({ type: 'loss', step: i, loss, ms: performance.now() - t0 });
      await new Promise((res) => setTimeout(res, 0));
    }
    postMessage({ type: 'trained', stopped: stopFlag });
  },
  stop() { stopFlag = true; },
  gradcheck() {
    const [xs, ys] = G.getBatch(data, model.cfg.block, 2, G.rng(5));
    const t0 = performance.now();
    const res = G.gradCheck(model, xs, ys);
    postMessage({ type: 'gradcheck', ...res, ms: performance.now() - t0 });
  },
  sample({ prompt, n, greedy, temperature, topK, seed }) {
    const ids = tok.encode([...prompt].filter((c) => tok.has(c)).join('') || tok.chars[0]);
    postMessage({ type: 'sample', text: tok.decode(G.generate(model, ids, n, { greedy, temperature, topK, seed })) });
  },
  attention({ prompt }) {
    const chars = [...prompt].filter((c) => tok.has(c)).slice(-model.cfg.block);
    if (!chars.length) return postMessage({ type: 'attention', chars: [], layers: [] });
    const { cache } = G.forward(model, tok.encode(chars.join('')), true);
    postMessage({ type: 'attention', chars, heads: model.cfg.heads, layers: cache.blocks.map((b) => b.probs) });
  },
};

onmessage = (e) => handlers[e.data.type](e.data);

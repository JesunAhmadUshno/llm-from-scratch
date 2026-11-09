import { useMemo, useState, useEffect, useRef, useId, useCallback } from 'react';
import deep from '../../level-3-deep/results_deep.json';
import var1 from '../../level-3-deep/results_deep_variance_t1.json';
import var20 from '../../level-3-deep/results_deep_variance_t20.json';
import learn from '../../level-1-learn/results_learn.json';
import build from '../../level-2-build/results_build.json';
import learnMd from '../../level-1-learn/README.md?raw';
import buildMd from '../../level-2-build/README.md?raw';
import deepMd from '../../level-3-deep/README.md?raw';
import reportMd from '../../REPORT.md?raw';
import corpus from '../../level-2-build/corpus.txt?raw';
import trainTiny from '../../level-1-learn/train_tiny.py?raw';
import torchSrc from '../../level-2-build/torch_gpt.py?raw';
import jaxSrc from '../../level-2-build/jax_gpt.py?raw';
import commonSrc from '../../level-2-build/common.py?raw';
import tLearn from '../../tests/test_learn_gpt.py?raw';
import tBuild from '../../tests/test_build_frameworks.py?raw';
import tDeep from '../../tests/test_deep_kv_cache.py?raw';
import ciYml from '../../.github/workflows/ci.yml?raw';
import { paramCount, tokenizer } from './gpt.js';
import * as L from './lib.js';

const REPO = 'https://github.com/JesunAhmadUshno/llm-from-scratch';
const blob = (p) => `${REPO}/blob/main/${p}`;
const fmt = (n, d = 0) => Number(n).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
const bytes = (b) => (b >= 1e9 ? `${fmt(b / 1e9, 2)} GB` : b >= 1e6 ? `${fmt(b / 1e6, 1)} MB` : b >= 1e3 ? `${fmt(b / 1e3, 1)} KB` : `${fmt(b)} B`);

// ---- everything below is read from repo files at build time
const LEARN_TEXT = L.learnCorpus(trainTiny);
const learnRows = L.learnTable(learnMd);
const buildRows = L.buildCurve(buildMd);
const bm = L.buildMeasures(buildMd);
const deepT = L.deepTables(deepMd);
const [speedA, speedB] = L.speedups(deep);
const nTests = [tLearn, tBuild, tDeep].reduce((a, s) => a + (s.match(/def test_/g) || []).length, 0);
const restartAt = +commonSrc.match(/RESTART_AT = (\d+)/)[1];
const corpusTok = tokenizer(corpus);
const [trainIds, valIds] = L.split(corpusTok.encode(corpus));
const ngram = L.ngramBaselines(trainIds, valIds, corpusTok.vocabSize);
const pyVersions = ciYml.match(/python-version: \[(.*)\]/)[1].replace(/"/g, '').split(',').map((s) => s.trim());
// The v1.1 backlog is the numbered list under REPORT.md's "## Future work" heading (formerly ROADMAP.md).
const futureWork = reportMd.split(/^## Future work.*$/m)[1]?.split(/^## /m)[0] ?? '';
const backlog = [...futureWork.matchAll(/^\d+\. (.+?):/gm)].map((m) => m[1]);
const quotedLines = deepMd.match(/torch_gpt\.py is (\d+) lines; jax_gpt\.py is (\d+) lines[\s\S]*?(\d+) shared lines/).slice(1).map(Number);
const learnOn = L.measuredOn(learnMd), buildOn = L.measuredOn(buildMd), deepOn = L.measuredOn(deepMd);
const learnSamples = [...learnMd.matchAll(/^- (Before training, greedy|After training, greedy|After training, temperature 0\.8 with top-k 10): "(.*)"$/gm)].map((m) => [m[1], m[2]]);
const firstRun = deepMd.match(/faster at Config B \(([\d.]+) against ([\d.]+) seconds\)/).slice(1);
const recordedTag = `recorded ${learnOn.split(',').slice(0, 2).join(',')} CPU`;

const SECTIONS = [['top', 'Overview', '◎'], ['learn', 'Learn', '∂'], ['tokens', 'Tokenizer', 'Aa'], ['arch', 'Architecture', '▦'], ['build', 'Build', '⇄'], ['deep', 'Deep', '⚡'], ['notes', 'Notes', '✎'], ['reproduce', 'Reproduce', '❯_']];

function useReveal() {
  useEffect(() => {
    const els = document.querySelectorAll('.reveal');
    if (!('IntersectionObserver' in window)) { els.forEach((e) => e.classList.add('in')); return; }
    const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), { threshold: 0.08 });
    els.forEach((e) => io.observe(e));
    return () => io.disconnect();
  }, []);
}

function MenuBar() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => { const t = setInterval(() => setNow(new Date()), 30000); return () => clearInterval(t); }, []);
  return (
    <div className="menubar" role="banner">
      <span className="apple" aria-hidden="true">◆</span>
      <a className="app" href="#top">LLM From Scratch</a>
      <a className="m" href={blob('level-1-learn/README.md')}>Learn</a>
      <a className="m" href={blob('level-2-build/README.md')}>Build</a>
      <a className="m" href={blob('level-3-deep/README.md')}>Deep</a>
      <span className="spacer" />
      <span className="status"><i /> {nTests} tests in CI</span>
      <a href={REPO}>GitHub ↗</a>
      <span className="clock">{now.toLocaleString('en-US', { weekday: 'short', hour: 'numeric', minute: '2-digit' })}</span>
    </div>
  );
}

function Dock() {
  const [active, setActive] = useState('top');
  useEffect(() => {
    const io = new IntersectionObserver((es) => es.forEach((e) => e.isIntersecting && setActive(e.target.id)), { rootMargin: '-45% 0px -50% 0px' });
    SECTIONS.forEach(([id]) => { const el = document.getElementById(id); if (el) io.observe(el); });
    return () => io.disconnect();
  }, []);
  return (
    <nav className="dock" aria-label="Sections">
      {SECTIONS.map(([id, label, icon]) => (
        <a key={id} href={`#${id}`} className={active === id ? 'on' : ''} aria-current={active === id ? 'true' : undefined}>
          <span className="ico" aria-hidden="true">{icon}</span><span className="lbl">{label}</span>
        </a>
      ))}
    </nav>
  );
}

function Range({ label, value, min, max, step, onChange, show, disabled }) {
  const id = useId();
  return (
    <div className="field">
      <label htmlFor={id}>{label} <span>{show}</span></label>
      <input id={id} type="range" min={min} max={max} step={step} value={value} disabled={disabled} onChange={(e) => onChange(+e.target.value)} />
    </div>
  );
}

function Seg({ label, value, options, onChange }) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map(([k, l]) => <button key={k} aria-pressed={value === k} onClick={() => onChange(k)}>{l}</button>)}
    </div>
  );
}

// Hand-written SVG line chart. series: [{ pts: [[x, y]], color, dash, width, label }]
function LineChart({ series, xMax, yMax, label, markers = [], xLabel = 'step', yLabel = 'loss (nats/char)', W = 520, H = 250 }) {
  const pad = { l: 40, r: 10, t: 12, b: 28 };
  const sx = (x) => pad.l + (x / xMax) * (W - pad.l - pad.r);
  const sy = (y) => pad.t + (1 - Math.min(y, yMax) / yMax) * (H - pad.t - pad.b);
  const yt = Array.from({ length: 5 }, (_, i) => (yMax / 4) * i);
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>
      {yt.map((t) => <g key={t}><line className="axis" x1={pad.l} x2={W - pad.r} y1={sy(t)} y2={sy(t)} /><text x={pad.l - 6} y={sy(t) + 4} textAnchor="end">{fmt(t, 1)}</text></g>)}
      {markers.map((m) => <g key={m.x}><line x1={sx(m.x)} x2={sx(m.x)} y1={pad.t} y2={H - pad.b} stroke="var(--warn)" strokeDasharray="4 4" /><text x={sx(m.x) + 4} y={pad.t + 10} style={{ fill: 'var(--warn)' }}>{m.label}</text></g>)}
      {series.map((s) => s.pts.length > 0 && (
        <g key={s.label}>
          <polyline fill="none" stroke={s.color} strokeWidth={s.width ?? 2} strokeDasharray={s.dash} strokeLinejoin="round" points={s.pts.map(([x, y]) => `${sx(x)},${sy(y)}`).join(' ')} />
          {s.dots && s.pts.map(([x, y]) => <circle key={x} cx={sx(x)} cy={sy(y)} r="3.5" fill={s.color} stroke="var(--surface-strong)" strokeWidth="1.5" />)}
        </g>
      ))}
      <text x={pad.l} y={H - 6}>{xLabel} 0</text><text x={W - pad.r} y={H - 6} textAnchor="end">{xLabel} {fmt(xMax)} · y: {yLabel}</text>
    </svg>
  );
}
const Legend = ({ items }) => <div className="legend">{items.map(([c, l, dash]) => <span key={l}><i style={{ background: dash ? 'transparent' : c, border: dash ? `2px dashed ${c}` : 0, boxSizing: 'border-box' }} />{l}</span>)}</div>;

function Bars({ rows, max, unit = '' }) {
  return (
    <div className="bars">
      {rows.map((b) => (
        <div className="bar-row" key={b.k}>
          <div className="lab"><span>{b.badge && <span className={`badge ${b.badge}`}>{b.badge === 'rec' ? 'recorded' : b.badge}</span>}{b.k}</span><b>{b.show ?? `${fmt(b.v, b.d ?? 1)}${unit}`}</b></div>
          <div className="track"><div className="fill" style={{ width: `${Math.min(100, (b.v / max) * 100)}%`, background: b.c }} /></div>
        </div>
      ))}
    </div>
  );
}

// ---- the live model lives in a Web Worker shared by the trainer and the attention explorer
function useModel() {
  const worker = useRef(null);
  const [st, setSt] = useState({ status: 'loading', losses: [], params: 0, version: 0 });
  // Created on first use: child effects (the trainer's init) run before this hook's own effects.
  const get = useCallback(() => {
    if (worker.current) return worker.current;
    const w = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    worker.current = w;
    w.onmessage = ({ data: m }) => setSt((s) => {
      if (m.type === 'ready') return { ...s, status: 'ready', params: m.params, losses: [], gc: null, sample: null, version: s.version + 1 };
      if (m.type === 'loss') return { ...s, status: 'training', losses: [...s.losses, [m.step, m.loss]], ms: m.ms };
      if (m.type === 'trained') return { ...s, status: 'ready', stopped: m.stopped, version: s.version + 1 };
      if (m.type === 'gradcheck') return { ...s, status: 'ready', gc: m };
      if (m.type === 'sample') return { ...s, sample: m.text };
      if (m.type === 'attention') return { ...s, att: m };
      return s;
    });
    return w;
  }, []);
  useEffect(() => () => worker.current?.terminate(), []);
  const send = useCallback((msg) => {
    if (msg.type === 'gradcheck') setSt((s) => ({ ...s, status: 'checking' }));
    get().postMessage(msg);
  }, [get]);
  return [st, send];
}

export default function App() {
  useReveal();
  const [model, send] = useModel();
  return (
    <div className="wrap">
      <MenuBar />
      <div className="glow" aria-hidden="true" />
      <Hero />
      <Learn model={model} send={send} />
      <Tokenizer />
      <Architecture model={model} send={send} />
      <Build />
      <Deep />
      <Notes />
      <Reproduce />
      <Dock />
      <footer>
        Every number on this page is read from the repo at build time (<code>level-*/results_*.json</code>, the level READMEs, and the source files) or computed live in your browser. Recorded runs are labelled recorded. Code: see <a href={blob('LICENSE')} style={{ color: 'inherit' }}>LICENSE</a>.
      </footer>
    </div>
  );
}

function Hero() {
  const b = deep.configs[1];
  return (
    <header className="card hero window" id="top" data-title="README.md">
      <div className="head">
        <p className="eyebrow">Learn · Build · Deep · one model, three implementations</p>
        <span className={`pill ${b.greedy_cached_equals_naive ? 'good' : 'bad'}`}>● cached = naive, token for token</span>
      </div>
      <h1>One tiny GPT, built three times, where a KV cache made generation <em>{speedB.toFixed(1)}x faster</em> in the recorded run.</h1>
      <p className="sub">
        The same decoder-only transformer, written in NumPy with every gradient derived by hand, then ported to PyTorch and to JAX and trained on a Hamlet slice, then given a KV cache.
        The {speedB.toFixed(2)}x figure is {fmt(b.generation[1].tokens_per_second, 1)} / {fmt(b.generation[0].tokens_per_second, 1)} tokens per second from <a href={blob('level-3-deep/results_deep.json')}>results_deep.json</a> ({fmt(b.params)}-parameter config, one run).
        A five-trial rerun on a different CPU measured {var1.configs[1].cache_speedup.mean.toFixed(2)}x ± {var1.configs[1].cache_speedup.std.toFixed(2)} single-threaded, so treat the headline as one run's number, not a constant.
      </p>
      <div className="actions">
        <a className="btn primary" href="#learn">Train a GPT in your browser</a>
        <a className="btn ghost" href="#deep">See the KV-cache study</a>
        <span className="hint">Everything here runs client-side; nothing is sent anywhere.</span>
      </div>
      <div className="tiles t5">
        <div className="tile"><div className="num">{fmt(learn.params)}</div><div className="cap">params, NumPy Learn model</div></div>
        <div className="tile"><div className="num">{fmt(build.pytorch.params)}</div><div className="cap">params, PyTorch and JAX ports</div></div>
        <div className="tile"><div className="num">{fmt(b.params)}</div><div className="cap">params, Deep Config B</div></div>
        <div className="tile"><div className="num">{nTests}</div><div className="cap">unit tests, counted from <code>tests/</code></div></div>
        <div className="tile"><div className="num">{corpusTok.vocabSize}</div><div className="cap">character vocab of <code>corpus.txt</code></div></div>
      </div>
    </header>
  );
}

const PRESETS = { quick: { layers: 1, embd: 32, steps: 150, batch: 8, lr: 0.005 }, learn: { layers: 2, embd: 64, steps: 300, batch: 16, lr: 0.005 } };

function Learn({ model, send }) {
  const [cfg, setCfg] = useState(PRESETS.quick);
  const [mode, setMode] = useState('greedy');
  const busy = model.status !== 'ready';
  const set = (k) => (v) => setCfg((c) => ({ ...c, [k]: v }));
  const init = useCallback(() => send({ type: 'init', text: LEARN_TEXT, cfg: { block: 32, embd: cfg.embd, heads: 4, layers: cfg.layers }, seed: 7 }), [cfg.embd, cfg.layers, send]);
  useEffect(init, [init]);
  const train = () => { init(); send({ type: 'train', steps: cfg.steps, batch: cfg.batch, lr: cfg.lr, seed: 42 }); };
  const sample = () => send({ type: 'sample', prompt: 'to be or ', n: 40, greedy: mode === 'greedy', temperature: 0.8, topK: mode === 'topk' ? 10 : 0, seed: 3 });
  const isLearn = cfg.layers === 2 && cfg.embd === 64;
  const last = model.losses.at(-1);
  return (
    <section id="learn" className="card window reveal" data-title="gpt_numpy.py — hand-written backprop">
      <div className="head">
        <div>
          <p className="eyebrow">Level 1 · Learn</p>
          <h2>Every gradient written by hand, then checked against finite differences.</h2>
          <p className="sub">
            The NumPy model has no autograd: <a href={blob('level-1-learn/gpt_numpy.py')}>gpt_numpy.py</a> derives the backward pass for embeddings, causal attention, LayerNorm, GELU and the head.
            Its test compares those gradients with central finite differences and requires relative error below 1e-4.
            Below, a JavaScript port of the same model trains live in a Web Worker on the same teaching corpus ({LEARN_TEXT.length} characters, {fmt(learn.config.vocab_size)} distinct), next to the recorded NumPy curve.
          </p>
        </div>
        <span className="pill info">{learn.params === paramCount({ V: 20, T: 32, C: 64, L: 2 }) ? '104,704 params recomputed' : 'param mismatch'}</span>
      </div>
      <div className="grid2">
        <div className="stack">
          <div className="controls" style={{ gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', margin: 0 }}>
            <Range label="Layers" value={cfg.layers} min={1} max={3} step={1} onChange={set('layers')} show={cfg.layers} disabled={busy} />
            <Range label="Embedding size" value={cfg.embd} min={16} max={64} step={16} onChange={set('embd')} show={cfg.embd} disabled={busy} />
            <Range label="Learning rate" value={cfg.lr} min={0.001} max={0.02} step={0.001} onChange={set('lr')} show={cfg.lr.toFixed(3)} disabled={busy} />
            <Range label="Steps" value={cfg.steps} min={20} max={300} step={10} onChange={set('steps')} show={cfg.steps} disabled={busy} />
          </div>
          <div className="actions" style={{ marginTop: 0 }}>
            {model.status === 'training'
              ? <button className="btn primary" onClick={() => send({ type: 'stop' })}>Stop training</button>
              : <button className="btn primary" onClick={train} disabled={busy}>Train live model</button>}
            <button className="btn ghost" onClick={() => setCfg(isLearn ? PRESETS.quick : PRESETS.learn)} disabled={busy}>{isLearn ? 'Quick preset (1L/32d)' : `Learn config (2L/64d/${fmt(learn.params)} params)`}</button>
          </div>
          <dl className="kv">
            <dt>Live model</dt><dd>{fmt(model.params)} params · batch {cfg.batch} · block 32 · 4 heads · Adam</dd>
            <dt>Status</dt><dd aria-live="polite">{model.status === 'training' ? `step ${model.losses.length} of ${cfg.steps}, loss ${last?.[1].toFixed(4)}, ${(model.ms / 1000).toFixed(1)} s` : model.losses.length ? `${model.stopped ? 'stopped' : 'done'} at step ${model.losses.length}, loss ${last?.[1].toFixed(4)} in ${(model.ms / 1000).toFixed(1)} s` : 'untrained (press Train)'}</dd>
          </dl>
          <div className="tile">
            <div className="head" style={{ marginBottom: 8 }}><h3>Gradient check, live</h3><span className={`pill ${model.gc ? (model.gc.maxRel < 1e-4 ? 'good' : 'bad') : 'info'}`}>{model.gc ? (model.gc.maxRel < 1e-4 ? 'below 1e-4' : 'above 1e-4') : 'not run'}</span></div>
            <p className="hint" style={{ margin: 0 }}>Central differences (ε 1e-5) against the analytic backward on 3 sampled entries of every parameter tensor of the current live model, the same procedure as <code>test_analytic_gradients_match_finite_differences</code>.</p>
            <div className="actions" style={{ marginTop: 10 }}>
              <button className="btn ghost" onClick={() => send({ type: 'gradcheck' })} disabled={busy}>{model.status === 'checking' ? 'Checking…' : 'Run gradient check'}</button>
              {model.gc && <span className="hint">max relative error <b>{model.gc.maxRel.toExponential(2)}</b> over {model.gc.checked} entries{model.gc.zeros ? ` (${model.gc.zeros} where both gradients are below 1e-7, such as the key bias, whose true gradient is exactly zero, are counted as agreeing)` : ''}, {fmt(model.gc.ms)} ms</span>}
            </div>
          </div>
        </div>
        <div className="stack">
          <div>
            <h3>Training loss: live JS model against the recorded NumPy run</h3>
            <LineChart label="Training loss curve: recorded NumPy run and live in-browser run" xMax={300} yMax={3.6}
              series={[
                { label: 'rec', pts: learn.losses.map((y, i) => [i + 1, y]), color: 'var(--muted)', width: 1.4 },
                { label: 'readme', pts: learnRows.map((r) => [r.step, r.loss]), color: 'var(--muted)', width: 0, dots: true },
                { label: 'live', pts: model.losses, color: 'var(--primary)', width: 2.2 },
              ]} />
            <Legend items={[['var(--muted)', `NumPy, ${recordedTag} (dots: README table)`], ['var(--primary)', 'Live JS port, this browser']]} />
            <p className="hint">The JS RNG differs from NumPy's, so the live curve is comparable in shape, not bit-identical. The grey line is the full 300-step array from a {learn.run_date} rerun ({learn.machine.split(' ')[1]}, NumPy {learn.numpy}), which matched all 7 README rows exactly; the build fails if it stops matching.</p>
          </div>
          <div>
            <div className="head" style={{ marginBottom: 6 }}><h3>Sample from the prompt "to be or "</h3>
              <Seg label="Sampling mode" value={mode} onChange={setMode} options={[['greedy', 'Greedy'], ['topk', 'Temp 0.8, top-k 10'], ['temp', 'Temp 0.8']]} /></div>
            <button className="btn ghost" onClick={sample} disabled={busy}>Sample live model</button>
            {model.sample && <p className="sample"><span className="badge live">live</span><b>to be or </b>{model.sample.slice(9)}</p>}
            {learnSamples.map(([k, s]) => <p className="sample" key={k}><span className="badge rec">recorded</span>{k}: <b>{s.slice(0, 9)}</b>{s.slice(9)}</p>)}
            <p className="hint">The recorded model memorised a tiny repeated passage, as the Learn README says: proof the machinery works, not language ability.</p>
          </div>
        </div>
      </div>
    </section>
  );
}

function Tokenizer() {
  const [text, setText] = useState('HAMLET.\nTo be, or not to be — that’s it.');
  const chars = Array.from(text);
  const kept = chars.filter((c) => corpusTok.has(c));
  const ids = corpusTok.encode(kept.join(''));
  const oov = [...new Set(chars.filter((c) => !corpusTok.has(c)))];
  const show = (c) => (c === '\n' ? '↵' : c === ' ' ? '␠' : c);
  return (
    <section id="tokens" className="card window reveal" data-title="corpus.txt — CharTokenizer">
      <p className="eyebrow">Live tokenizer</p>
      <h2>{corpusTok.vocabSize} characters are the whole vocabulary.</h2>
      <p className="sub">
        Built in your browser from <a href={blob('level-2-build/corpus.txt')}>corpus.txt</a> ({fmt(Array.from(corpus).length)} characters of Hamlet, Project Gutenberg ebook 1524) exactly like <code>common.CharTokenizer</code>: sorted distinct characters, split by code point so the curly apostrophe and accented letter count once each.
        Type anything; characters the model has never seen are flagged and dropped.
      </p>
      <div className="grid2" style={{ marginTop: 16 }}>
        <div className="stack">
          <div className="field"><label htmlFor="tok-in">Text to encode <span>{chars.length} chars</span></label>
            <textarea id="tok-in" rows={4} value={text} onChange={(e) => setText(e.target.value)} /></div>
          <dl className="kv">
            <dt>Token ids</dt><dd className="mono">[{ids.slice(0, 60).join(', ')}{ids.length > 60 ? ', …' : ''}]</dd>
            <dt>Round trip</dt><dd>{corpusTok.decode(ids) === kept.join('') ? <span className="pill good">decode(encode(x)) = x</span> : <span className="pill bad">mismatch</span>}</dd>
            <dt>Out of vocab</dt><dd>{oov.length ? oov.map((c) => <span key={c} className="chip oov" style={{ marginRight: 4 }}>{show(c)}</span>) : <span className="pill good">none</span>}</dd>
          </dl>
        </div>
        <div>
          <h3>Your text as tokens</h3>
          <div className="chips" aria-label="Tokens">{chars.slice(0, 160).map((c, i) => <span key={i} className={`chip${corpusTok.has(c) ? '' : ' oov'}`}>{show(c)}<small>{corpusTok.has(c) ? corpusTok.encode(c)[0] : '?'}</small></span>)}</div>
          <h3 style={{ marginTop: 16 }}>The full vocabulary</h3>
          <div className="chips">{corpusTok.chars.map((c, i) => <span key={i} className="chip">{show(c)}<small>{i}</small></span>)}</div>
        </div>
      </div>
    </section>
  );
}

const ARCH_PRESETS = [
  { k: 'Learn', V: learn.config.vocab_size, T: learn.config.block_size, C: learn.config.n_embd, L: learn.config.n_layer, headBias: false, rec: learn.params },
  { k: 'Build', V: build.vocab_size, T: 32, C: 64, L: 2, headBias: true, rec: build.pytorch.params },
  ...deep.configs.map((c, i) => ({ k: `Deep ${'AB'[i]}`, V: deep.vocab_size, T: c.config.block_size, C: c.config.n_embd, L: c.config.n_layer, headBias: true, rec: c.params })),
];

function Architecture({ model, send }) {
  const [a, setA] = useState(ARCH_PRESETS[0]);
  const set = (k) => (v) => setA((s) => ({ ...s, [k]: v, k: null }));
  const total = paramCount(a);
  const match = ARCH_PRESETS.find((p) => p.V === a.V && p.T === a.T && p.C === a.C && p.L === a.L && p.headBias === a.headBias);
  const parts = [
    { k: 'Token + position embeddings', v: a.V * a.C + a.T * a.C, c: 'var(--primary)' },
    { k: `${a.L} transformer block${a.L > 1 ? 's' : ''} (attention + MLP + 2 LayerNorms)`, v: a.L * (12 * a.C * a.C + 13 * a.C), c: 'var(--good)' },
    { k: 'Final LayerNorm + LM head', v: 2 * a.C + a.C * a.V + (a.headBias ? a.V : 0), c: 'var(--warn)' },
  ];
  const [prompt, setPrompt] = useState('to be or not to be');
  const [layer, setLayer] = useState(0), [head, setHead] = useState(0);
  useEffect(() => { if (model.status === 'ready') send({ type: 'attention', prompt }); }, [prompt, model.version, model.status, send]);
  const att = model.att;
  const nL = att?.layers?.length ?? 0, H = att?.heads ?? 4, T = att?.chars?.length ?? 0;
  const li = Math.min(layer, Math.max(0, nL - 1));
  const cell = T ? Math.min(22, 440 / T) : 20, off = 18;
  const show = (c) => (c === ' ' ? '␠' : c);
  return (
    <section id="arch" className="card window reveal" data-title="TinyGPT config — parameters · attention">
      <p className="eyebrow">Architecture</p>
      <h2>One formula reproduces all four recorded parameter counts.</h2>
      <p className="sub">
        V·C + T·C + L·(12C² + 13C) + 2C + C·V, plus V for the head bias on the PyTorch and JAX tracks. The build fails unless it returns {ARCH_PRESETS.map((p) => fmt(p.rec)).join(', ')} for the four configs on record.
      </p>
      <div className="grid2" style={{ marginTop: 16 }}>
        <div className="stack">
          <div className="seg" role="group" aria-label="Recorded configs">{ARCH_PRESETS.map((p) => <button key={p.k} aria-pressed={match?.k === p.k} onClick={() => setA(p)}>{p.k}</button>)}</div>
          <div className="controls" style={{ gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', margin: 0 }}>
            <Range label="Vocab V" value={a.V} min={2} max={256} step={1} onChange={set('V')} show={a.V} />
            <Range label="Block size T" value={a.T} min={8} max={1024} step={8} onChange={set('T')} show={a.T} />
            <Range label="Embedding C" value={a.C} min={16} max={768} step={16} onChange={set('C')} show={a.C} />
            <Range label="Layers L" value={a.L} min={1} max={12} step={1} onChange={set('L')} show={a.L} />
          </div>
          <label className="hint" style={{ display: 'flex', gap: 8, alignItems: 'center' }}><input type="checkbox" checked={a.headBias} onChange={(e) => setA((s) => ({ ...s, headBias: e.target.checked }))} /> LM head has a bias (PyTorch and JAX tracks)</label>
          <div className="tile"><div className="num">{fmt(total)}</div><div className="cap">{match ? <>parameters · <span className="ok">matches the recorded {match.k} count</span></> : 'parameters (custom config)'}</div></div>
          <Bars rows={parts.map((p) => ({ ...p, show: `${fmt(p.v)} · ${((p.v / total) * 100).toFixed(0)}%` }))} max={total} />
        </div>
        <div className="stack">
          <div className="head" style={{ marginBottom: 0 }}>
            <h3><span className="badge live">live</span>Attention of the in-browser model</h3>
            <span className="hint">{model.losses.length ? `after ${model.losses.length} training steps` : 'untrained weights: train it above'}</span>
          </div>
          <div className="field"><label htmlFor="att-in">Prompt <span>{T} of 32 positions</span></label><input id="att-in" type="text" value={prompt} maxLength={32} onChange={(e) => setPrompt(e.target.value.toLowerCase())} /></div>
          <div className="actions" style={{ marginTop: 0 }}>
            <Seg label="Layer" value={li} onChange={setLayer} options={Array.from({ length: nL }, (_, i) => [i, `Layer ${i + 1}`])} />
            <Seg label="Head" value={head} onChange={setHead} options={Array.from({ length: H }, (_, i) => [i, `H${i + 1}`])} />
          </div>
          {T > 0 ? (
            <div className="heat-wrap">
              <svg className="chart" viewBox={`0 0 ${off + T * cell + 4} ${off + T * cell + 4}`} role="img" aria-label={`Attention weights, layer ${li + 1}, head ${head + 1}; future positions masked`} style={{ maxWidth: 460 }}>
                {att.chars.map((c, i) => <g key={i}><text x={off - 4} y={off + i * cell + cell * 0.7} textAnchor="end">{show(c)}</text><text x={off + i * cell + cell / 2} y={off - 5} textAnchor="middle">{show(c)}</text></g>)}
                {att.chars.map((_, i) => att.chars.map((__, j) => {
                  const pv = att.layers[li][head * T * T + i * T + j];
                  return j > i
                    ? <rect key={`${i}-${j}`} x={off + j * cell} y={off + i * cell} width={cell - 1} height={cell - 1} fill="var(--grid)" />
                    : <rect key={`${i}-${j}`} x={off + j * cell} y={off + i * cell} width={cell - 1} height={cell - 1} rx="2" fill="var(--primary)" opacity={0.08 + 0.92 * pv}><title>{`query "${show(att.chars[i])}" (pos ${i}) → key "${show(att.chars[j])}" (pos ${j}): ${pv.toFixed(3)}`}</title></rect>;
                }))}
              </svg>
            </div>
          ) : <p className="hint">Type a prompt using the Learn corpus characters (lowercase letters and spaces).</p>}
          <p className="hint">Rows are queries, columns are keys. Grey cells are the future: the causal mask gives them exactly zero weight, which is what the repo's causality test proves for the NumPy model.</p>
        </div>
      </div>
    </section>
  );
}

function Build() {
  const [view, setView] = useState('recorded');
  const step1 = { torch: L.num(bm['Train loss, step 1'].torch), jax: L.num(bm['Train loss, step 1'].jax) };
  const rec = (k) => [[1, step1[k]], ...buildRows.map((r) => [r.step, r[k]])];
  const torchC = 'var(--primary)', jaxC = 'var(--bad)';
  const series = view === 'recorded'
    ? [{ label: 't', pts: rec('torch'), color: torchC, dots: true }, { label: 'j', pts: rec('jax'), color: jaxC, dots: true }]
    : [{ label: 't', pts: build.pytorch.losses.map((y, i) => [i + 1, y]), color: torchC, width: 1.3 }, { label: 'j', pts: build.jax.losses.map((y, i) => [i + 1, y]), color: jaxC, width: 1.3 }];
  const val = { torch: L.num(bm['Validation loss, after training'].torch), jax: L.num(bm['Validation loss, after training'].jax) };
  const tps = { torch: L.num(bm['Tokens per second'].torch), jax: L.num(bm['Tokens per second'].jax) };
  return (
    <section id="build" className="card window reveal" data-title="torch_gpt.py vs jax_gpt.py — same model, same data">
      <div className="head">
        <div>
          <p className="eyebrow">Level 2 · Build</p>
          <h2>Same model, same Hamlet slice, two frameworks, both restarted from a checkpoint.</h2>
          <p className="sub">
            Both tracks train {fmt(build.pytorch.params)} parameters for 300 steps on the first {fmt(build.corpus_chars)} characters of Hamlet, save a checkpoint at step {restartAt}, rebuild the model from it, and finish.
            Read the gap honestly: init and batch order are seeded per track, so the loss difference is a recipe difference, not a framework result; and JAX tokens per second include JIT compilation.
          </p>
        </div>
        <Seg label="Curve source" value={view} onChange={setView} options={[['recorded', 'README record'], ['rerun', `Rerun ${build.run_date}`]]} />
      </div>
      <div className="grid2">
        <div>
          <h3>Training loss {view === 'recorded' ? `(${recordedTag}, logged steps)` : `(all 300 steps, rerun ${build.run_date})`}</h3>
          <LineChart label="PyTorch and JAX training loss with checkpoint restart" xMax={300} yMax={4.8} markers={[{ x: restartAt, label: `restart @ ${restartAt}` }]} series={series} />
          <Legend items={[[torchC, 'PyTorch'], [jaxC, 'JAX'], ['var(--warn)', 'checkpoint restart', true]]} />
          <p className="hint">{view === 'recorded'
            ? `Step 1 plus steps 50 to 300, parsed from the Build README tables at build time (${buildOn}).`
            : `From results_build.json, rerun on ${build.machine} with ${build.pytorch.framework} and ${build.jax.framework}. PyTorch reproduced the README losses exactly; JAX drifted (val ${build.jax.val_loss_after} against ${val.jax}).`}</p>
        </div>
        <div className="stack">
          <div>
            <h3>Validation loss after training, against baselines (nats per char, lower is better)</h3>
            <div style={{ marginTop: 10 }}>
              <Bars max={4.5} rows={[
                { k: `Uniform, ln ${corpusTok.vocabSize}`, v: ngram.uniform, d: 4, c: 'var(--grid)', badge: 'live' },
                { k: 'Unigram (add-1)', v: ngram.unigram, d: 4, c: 'var(--muted)', badge: 'live' },
                { k: 'Bigram (add-1)', v: ngram.bigram, d: 4, c: 'var(--warn)', badge: 'live' },
                { k: 'JAX TinyGPT', v: val.jax, d: 4, c: jaxC, badge: 'rec' },
                { k: 'PyTorch TinyGPT', v: val.torch, d: 4, c: torchC, badge: 'rec' },
              ]} />
            </div>
            <p className="hint">Baselines are fitted on the same 90% train split of corpus.txt and scored on the full 10% validation split in your browser; the model numbers are the README's mean over 10 fixed-seed validation batches, so the estimators differ slightly. After 300 steps, JAX sits {(ngram.bigram - val.jax).toFixed(3)} below a bigram model and PyTorch {(ngram.bigram - val.torch).toFixed(3)} below.</p>
          </div>
          <div>
            <h3>Training tokens per second ({recordedTag})</h3>
            <div style={{ marginTop: 10 }}>
              <Bars max={Math.max(tps.torch, tps.jax) * 1.1} rows={[{ k: 'PyTorch', v: tps.torch, c: torchC, badge: 'rec' }, { k: 'JAX (includes JIT compile)', v: tps.jax, c: jaxC, badge: 'rec' }]} />
            </div>
            <p className="hint">The {build.run_date} rerun on a different machine gave PyTorch {fmt(build.pytorch.tokens_per_second, 1)} and JAX {fmt(build.jax.tokens_per_second, 1)}: the ordering flipped, so no framework speed ranking is claimed from one CPU run.</p>
          </div>
        </div>
      </div>
    </section>
  );
}

function Deep() {
  const [ci, setCi] = useState(1);
  const c = deep.configs[ci];
  const g = Object.fromEntries(c.generation.map((r) => [r.mode, r]));
  const p = Object.fromEntries(c.prefill.map((r) => [r.path, r]));
  const [ctx, setCtx] = useState(Math.log10(100000));
  const [kvc, setKvc] = useState(1);
  const kc = deep.configs[kvc].config;
  const tokens = Math.round(10 ** ctx);
  const [gen, setGen] = useState(64);
  const ops = L.attentionOps(c.prompt_tokens, gen);
  const v = [var1, var20];
  return (
    <section id="deep" className="card window reveal" data-title="kv_cache.py — results_deep.json">
      <div className="head">
        <div>
          <p className="eyebrow">Level 3 · Deep</p>
          <h2>The KV cache gives the same tokens, faster, but how much faster depends on the run.</h2>
          <p className="sub">
            Naive generation recomputes the whole window every step; the cached mode keeps each layer's keys and values and computes only the new token.
            Greedy outputs were token-identical over {c.equivalence_tokens_checked} tokens on both configs (seeded untrained weights, one prompt, batch 1, CPU). Bars are the recorded run ({deepOn}).
          </p>
        </div>
        <Seg label="Deep config" value={ci} onChange={setCi} options={deep.configs.map((x, i) => [i, `Config ${'AB'[i]} · ${fmt(x.params)}`])} />
      </div>
      <div className="grid2">
        <div className="stack">
          <div>
            <h3>Generation, 64 tokens, tokens per second</h3>
            <div style={{ marginTop: 10 }}>
              <Bars max={Math.max(g.naive.tokens_per_second, g.cached.tokens_per_second) * 1.1} rows={[
                { k: 'Naive (recompute window)', v: g.naive.tokens_per_second, c: 'var(--muted)', badge: 'rec' },
                { k: 'KV cached', v: g.cached.tokens_per_second, c: 'var(--primary)', badge: 'rec' },
              ]} />
            </div>
            <p className="hint">Speedup {(ci ? speedB : speedA).toFixed(2)}x, mean of {g.naive.repeats} generations per mode. (The README prose rounds Config A to 2.34; the recorded numbers give {speedA.toFixed(2)}.)</p>
          </div>
          <div>
            <h3>Prefill, batch {p.hand.batch_size} × {p.hand.seq_len}, seconds (lower is faster)</h3>
            <div style={{ marginTop: 10 }}>
              <Bars max={Math.max(p.hand.seconds_mean, p.sdpa.seconds_mean) * 1.1} rows={[
                { k: 'Hand-written attention', v: p.hand.seconds_mean, d: 4, c: 'var(--good)', badge: 'rec' },
                { k: 'SDPA (fused)', v: p.sdpa.seconds_mean, d: 4, c: 'var(--warn)', badge: 'rec' },
              ]} unit=" s" />
            </div>
            <p className="hint">The README's first run had SDPA faster at Config B ({firstRun[0]} against {firstRun[1]} s): the winner flips between runs, so no stable prefill winner is claimed.</p>
          </div>
        </div>
        <div className="stack">
          <div>
            <h3>Does the speedup replicate? Five-trial reruns ({var1.run_date}, {var1.machine})</h3>
            <div style={{ overflowX: 'auto' }}>
              <table>
                <thead><tr><th>Run</th><th>Config A speedup</th><th>Config B speedup</th><th>Hand prefill faster</th></tr></thead>
                <tbody>
                  <tr><td><span className="badge rec">recorded</span>1 run, README</td><td>{speedA.toFixed(2)}x</td><td>{speedB.toFixed(2)}x</td><td>2 of 2 configs</td></tr>
                  {v.map((r) => (
                    <tr key={r.torch_threads}><td>{r.trials} trials, {r.torch_threads} thread{r.torch_threads > 1 ? 's' : ''}</td>
                      {r.configs.map((x, i) => <td key={i} className={x.cache_speedup.mean < 1 ? 'no' : undefined}>{x.cache_speedup.mean.toFixed(2)}x ± {x.cache_speedup.std.toFixed(2)}<div className="hint">range {x.cache_speedup.min.toFixed(2)} to {x.cache_speedup.max.toFixed(2)}</div></td>)}
                      <td>{r.configs.map((x) => `${x.prefill_hand_faster_trials}/${r.trials}`).join(' · ')}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="hint">Output was token-identical in every trial. Single-threaded, the speedup is small and stable; with {var20.torch_threads} threads the tiny tensors drown in threading overhead and Config B caching was slower than naive. The cache saves attention work, but at this size per-call overhead decides the wall clock.</p>
          </div>
          <div className="tile">
            <h3><span className="badge live">live</span>KV-cache memory calculator</h3>
            <div className="actions" style={{ marginTop: 8 }}><Seg label="Cache config" value={kvc} onChange={setKvc} options={deep.configs.map((x, i) => [i, `Config ${'AB'[i]}: ${x.config.n_layer}L × ${x.config.n_embd}`])} /></div>
            <Range label="Context length (tokens)" value={ctx} min={Math.log10(128)} max={6} step={0.01} onChange={setCtx} show={fmt(tokens)} />
            <dl className="kv">
              <dt>Floats per token</dt><dd>2 × {kc.n_layer} layers × {kc.n_embd} = {fmt(L.cacheFloatsPerToken(kc.n_layer, kc.n_embd))}</dd>
              <dt>fp32 cache, one sequence</dt><dd><b>{bytes(L.cacheBytes(kc.n_layer, kc.n_embd, tokens))}</b></dd>
            </dl>
            <p className="hint" style={{ marginBottom: 0 }}>Set Config B to 100,000 tokens to check the README's "about 400 MB" failure-mode estimate. The code itself caps context at block {kc.block_size} and raises past it.</p>
          </div>
          <div className="tile">
            <h3><span className="badge model">model</span>Attention dot products, naive vs cached</h3>
            <Range label="Tokens generated" value={gen} min={1} max={c.config.block_size - c.prompt_tokens} step={1} onChange={setGen} show={gen} />
            <Bars max={ops.naive} rows={[{ k: 'Naive', v: ops.naive, d: 0, c: 'var(--muted)' }, { k: 'Cached (incl. prefill)', v: ops.cached, d: 0, c: 'var(--primary)' }]} />
            <p className="hint" style={{ marginBottom: 0 }}>An analytic count of query-key pairs per head and layer after the {c.prompt_tokens}-token prompt ({(ops.naive / ops.cached).toFixed(1)}x fewer when cached). A model, not a measurement: projections, MLP and head still run per token, which is why measured speedups are far smaller.</p>
          </div>
        </div>
      </div>
    </section>
  );
}

function Notes() {
  const counts = [['torch_gpt.py', L.wcLines(torchSrc), quotedLines[0]], ['jax_gpt.py', L.wcLines(jaxSrc), quotedLines[1]], ['common.py (shared)', L.wcLines(commonSrc), quotedLines[2]]];
  const max = Math.max(...counts.map((r) => r[1]));
  return (
    <section id="notes" className="card window reveal" data-title="NOTES.md — comparison, failure modes, limits">
      <p className="eyebrow">Comparison and limits</p>
      <h2>What the PyTorch and JAX ports taught, and what is not done.</h2>
      <div className="grid2" style={{ marginTop: 16 }}>
        <div className="stack">
          <div>
            <h3>Code size, recounted from the files at build time</h3>
            <div style={{ marginTop: 10 }}><Bars max={max * 1.1} rows={counts.map(([k, n, q]) => ({ k, v: n, d: 0, show: `${n} lines`, c: k.startsWith('jax') ? 'var(--bad)' : k.startsWith('torch') ? 'var(--primary)' : 'var(--muted)' }))} /></div>
            <p className="hint">The Deep README quotes {quotedLines.join(' / ')} lines (wc -l at the time); the files have been edited since, so the bars show today's count. JAX is longer because the parameter pytree, Adam and the checkpoint are written out by hand.</p>
          </div>
          <div>
            <h3>Debugging log (from the Deep README)</h3>
            <ol className="list">
              <li><b>JAX tracer error.</b> A jitted loop over a traced length raised ConcretizationTypeError; fix: static argument or lax.scan with a mask.</li>
              <li><b>SDPA causal alignment.</b> <code>is_causal</code> assumes equal query and key lengths; the cached step passes causal=False.</li>
              <li><b>Step mask shape.</b> The square mask cannot add to (1, cache) scores; applied only when query length &gt; 1.</li>
              <li><b>Cache concat axis.</b> After the head split, append on dim 2, not dim 1; a shape test guards it.</li>
            </ol>
          </div>
        </div>
        <div className="stack">
          <div>
            <h3>Failure modes and threats to validity</h3>
            <ul className="list">
              <li><b>Toy scale.</b> 300 steps, about {fmt(build.pytorch.params / 1000)}k params, one config per level, one seed per track. No scaling sweep backs "the cache win grows with model size": it rests on two configs.</li>
              <li><b>Confounded framework comparison.</b> Per-track seeds and JIT-inclusive timing; memory is one combined-process RSS, not per track.</li>
              <li><b>Unstable timings.</b> A rerun on a different machine and torch build gave different speedups and prefill winner (table above); the recorded numbers have not been re-measured under the same conditions.</li>
              <li><b>Deep study scope.</b> Untrained weights, greedy only, batch 1, one prompt, no padding, no eviction: the cache raises past block size by design.</li>
              <li><b>Learn memorises.</b> The Learn corpus is a 6x-repeated passage; its low loss is memorisation, and it has no validation split.</li>
              <li><b>TPU note is reasoning only.</b> No TPU and no GPU run exists; no TPU or GPU number is claimed.</li>
            </ul>
          </div>
          <div>
            <h3>v1.1 backlog <span className="pill warn" style={{ marginLeft: 6 }}>not done</span></h3>
            <ol className="list">{backlog.map((b) => <li key={b}>{b}</li>)}</ol>
          </div>
        </div>
      </div>
    </section>
  );
}

function Reproduce() {
  const docs = [
    ['README.md', 'README', 'Question, headline, verdict, reproduce'],
    ['PROPOSAL.md', 'Proposal', 'Scope and prior work (not pre-registered)'],
    ['level-1-learn/README.md', 'Learn', 'NumPy model and gradient proof'],
    ['level-2-build/README.md', 'Build', 'PyTorch and JAX side by side'],
    ['level-3-deep/README.md', 'Deep', 'KV cache, SDPA, comparison'],
    ['REPORT.md', 'Report', 'Results, misses, review findings, v1.1 backlog'],
  ];
  return (
    <section id="reproduce" className="card window reveal" data-title="Terminal — reproduce">
      <p className="eyebrow">Reproduce it</p>
      <h2>Every recorded number comes from a script you can rerun on a CPU.</h2>
      <ol className="steps" style={{ marginTop: 16 }}>
        <li><span className="n">1</span><div><b>Install</b><div className="hint"><code>pip install numpy && pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install jax</code></div></div></li>
        <li><span className="n">2</span><div><b>Tests ({nTests})</b><div className="hint"><code>python -m unittest discover -s tests -p "test_*.py"</code></div></div></li>
        <li><span className="n">3</span><div><b>Learn</b> (writes results_learn.json, about {Math.round(+learnMd.match(/about (\d+) seconds/)[1])} s on the recorded VM)<div className="hint"><code>python level-1-learn/train_tiny.py</code></div></div></li>
        <li><span className="n">4</span><div><b>Build</b> (writes results_build.json)<div className="hint"><code>python level-2-build/run_build.py</code></div></div></li>
        <li><span className="n">5</span><div><b>Deep</b>, plus the variance pass<div className="hint"><code>python level-3-deep/run_deep.py && OMP_NUM_THREADS=1 python level-3-deep/run_deep.py --variance 5</code></div></div></li>
      </ol>
      <dl className="kv" style={{ marginTop: 16 }}>
        <dt>CI matrix</dt><dd>Python {pyVersions.join(' and ')}, every push to main</dd>
        <dt>Learn record</dt><dd>{learnOn}</dd>
        <dt>Build record</dt><dd>{buildOn}</dd>
        <dt>Deep record</dt><dd>{deepOn}</dd>
        <dt>Reruns</dt><dd>{learn.run_date}, {learn.machine}, Python {learn.python}, NumPy {learn.numpy}, {build.pytorch.framework}, {build.jax.framework}</dd>
      </dl>
      <div className="docs" style={{ marginTop: 20 }}>
        {docs.map(([p, t, d]) => <a key={p} href={blob(p)}><b>{t}</b><small>{d}</small></a>)}
      </div>
    </section>
  );
}

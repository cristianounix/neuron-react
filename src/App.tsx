import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { NeuronDiagram } from './components/NeuronDiagram';
import { DecisionPlot } from './components/DecisionPlot';
import { ErrorChart, type ErrorSample } from './components/ErrorChart';
import { DATASETS, generateDataset, mulberry32, type DatasetKey } from './data';
import {
  ACTIVATIONS,
  WEIGHT_LIMIT,
  accuracy,
  meanSquaredError,
  trainStep,
  type ActivationKey,
  type StepResult,
  type Weights,
} from './neuron';
import { prefersReducedMotion, useTween } from './hooks/useAnimation';

const FORWARD_MS = 900;
const UPDATE_MS = 700;
const AUTO_PAUSE_MS = 250;
const MAX_HISTORY = 500;

type LastResult = StepResult & { iteration: number; pointId: number; trained: boolean };

const fmt = (v: number, d = 2) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);

function randomWeights(seed: number): Weights {
  const rand = mulberry32(seed);
  const r = () => Math.round((rand() * 4 - 2) * 100) / 100;
  return { w1: r(), w2: r(), b: r() };
}

function Slider(props: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  disabled?: boolean;
}) {
  const { label, value, min, max, step, onChange, disabled } = props;
  return (
    <label className="slider">
      <span className="slider-label">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <span className="slider-value">{value.toFixed(2)}</span>
    </label>
  );
}

export default function App() {
  const [datasetKey, setDatasetKey] = useState<DatasetKey>('blobs');
  const [seed, setSeed] = useState(7);
  const points = useMemo(() => generateDataset(datasetKey, seed), [datasetKey, seed]);

  const [weights, setWeights] = useState<Weights>({ w1: 0.8, w2: -1.2, b: 0.3 });
  const [activation, setActivation] = useState<ActivationKey>('sigmoid');
  const [learningRate, setLearningRate] = useState(0.3);
  const [inputs, setInputs] = useState({ x1: 0.3, x2: -0.4 });
  const [learn, setLearn] = useState(true);
  const [auto, setAuto] = useState(false);

  const [cursor, setCursor] = useState(0);
  const [iteration, setIteration] = useState(0);
  const [run, setRun] = useState<{ id: number; pointId: number } | null>(null);
  const [last, setLast] = useState<LastResult | null>(null);
  const [errorHistory, setErrorHistory] = useState<ErrorSample[]>([]);

  const weightsRef = useRef(weights);
  weightsRef.current = weights;
  const timer = useRef<number>();

  const tweenedWeights = useTween(weights, UPDATE_MS);
  const forwardMs = prefersReducedMotion() ? 0 : FORWARD_MS;
  const acc = accuracy(points, weights);
  const loss = meanSquaredError(points, weights, activation);
  const busy = run !== null;

  const predict = useCallback(() => {
    if (busy || points.length === 0) return;
    const point = points[cursor % points.length];
    const id = iteration + 1;
    setInputs({ x1: point.x1, x2: point.x2 });
    setRun({ id, pointId: point.id });
    setLast(null);

    // Once the signal has travelled through the neuron, reveal the prediction and learn from it.
    timer.current = window.setTimeout(() => {
      const step = trainStep(weightsRef.current, activation, point, learningRate);
      setLast({ ...step, iteration: id, pointId: point.id, trained: learn });
      const before = weightsRef.current;
      const after = learn ? step.next : before;
      if (learn) setWeights(after);
      const sample: ErrorSample = { iteration: id, loss: meanSquaredError(points, after, activation), correct: step.correct };
      setErrorHistory((h) =>
        (h.length
          ? [...h, sample]
          : [{ iteration: id - 1, loss: meanSquaredError(points, before, activation), correct: null }, sample]
        ).slice(-MAX_HISTORY),
      );
      setIteration(id);
      setCursor((c) => c + 1);
      setRun(null);
    }, forwardMs);
  }, [busy, points, cursor, iteration, activation, learningRate, learn, forwardMs]);

  // Auto-train: keep pressing "Predict" after each boundary animation settles.
  useEffect(() => {
    if (!auto || busy) return;
    const t = window.setTimeout(predict, last ? UPDATE_MS + AUTO_PAUSE_MS : 0);
    return () => window.clearTimeout(t);
  }, [auto, busy, predict, last]);

  useEffect(() => () => window.clearTimeout(timer.current), []);

  const cancelRun = () => {
    window.clearTimeout(timer.current);
    setRun(null);
  };

  const resetProgress = () => {
    cancelRun();
    setCursor(0);
    setIteration(0);
    setLast(null);
    setErrorHistory([]);
  };

  const changeDataset = (key: DatasetKey) => {
    setDatasetKey(key);
    resetProgress();
  };

  const newData = () => {
    setSeed((s) => s + 1);
    resetProgress();
  };

  const resetWeights = () => {
    cancelRun();
    setWeights(randomWeights(Date.now() & 0xffff));
    resetProgress();
  };

  const setWeight = (k: keyof Weights) => (v: number) => setWeights((w) => ({ ...w, [k]: v }));

  const highlight = run
    ? { pointId: run.pointId, runId: run.id, correct: null }
    : last
      ? { pointId: last.pointId, runId: last.iteration, correct: last.correct }
      : null;

  const epoch = points.length ? Math.floor(iteration / points.length) + 1 : 1;
  const lastPoint = last ? points.find((p) => p.id === last.pointId) : undefined;

  return (
    <div className="app">
      <header className="app-header">
        <h1>Neuron React</h1>
        <p>
          An artificial neuron multiplies each input by a weight, adds a bias, and passes the sum through an activation
          function. Press <strong>Predict</strong> to feed the next sample through the neuron and let it learn.
        </p>
      </header>

      <main className="layout">
        {/* Left: neuron structure + error over time */}
        <div className="column">
          <section className="card">
            <div className="card-header">
              <h2>Neuron</h2>
              <code className="formula">y = f(w₁·x₁ + w₂·x₂ + b)</code>
            </div>

            <NeuronDiagram
              inputs={inputs}
              weights={weights}
              activation={activation}
              runId={run?.id ?? null}
              forwardMs={forwardMs}
            />

            <div className="controls">
              <fieldset>
                <legend>Input signals</legend>
                <Slider label="x₁" value={inputs.x1} min={-1} max={1} step={0.01} disabled={busy} onChange={(v) => setInputs((i) => ({ ...i, x1: v }))} />
                <Slider label="x₂" value={inputs.x2} min={-1} max={1} step={0.01} disabled={busy} onChange={(v) => setInputs((i) => ({ ...i, x2: v }))} />
              </fieldset>

              <fieldset>
                <legend>Weights &amp; bias</legend>
                <Slider label="w₁" value={weights.w1} min={-WEIGHT_LIMIT} max={WEIGHT_LIMIT} step={0.01} onChange={setWeight('w1')} />
                <Slider label="w₂" value={weights.w2} min={-WEIGHT_LIMIT} max={WEIGHT_LIMIT} step={0.01} onChange={setWeight('w2')} />
                <Slider label="b" value={weights.b} min={-WEIGHT_LIMIT} max={WEIGHT_LIMIT} step={0.01} onChange={setWeight('b')} />
              </fieldset>

              <fieldset>
                <legend>Activation function</legend>
                <div className="segmented" role="radiogroup" aria-label="Activation function">
                  {(Object.keys(ACTIVATIONS) as ActivationKey[]).map((k) => (
                    <button
                      key={k}
                      role="radio"
                      aria-checked={activation === k}
                      className={activation === k ? 'active' : ''}
                      onClick={() => {
                        setActivation(k);
                        setErrorHistory([]); // the loss scale differs per activation
                      }}
                    >
                      {ACTIVATIONS[k].label}
                    </button>
                  ))}
                </div>
                <p className="hint">{ACTIVATIONS[activation].formula}</p>
              </fieldset>

              <fieldset>
                <legend>Learning</legend>
                <Slider label="η" value={learningRate} min={0.01} max={1} step={0.01} onChange={setLearningRate} />
                <label className="checkbox">
                  <input type="checkbox" checked={learn} onChange={(e) => setLearn(e.target.checked)} />
                  Update weights after each prediction
                </label>
              </fieldset>
            </div>
          </section>

          <section className="card">
            <div className="card-header">
              <h2>Error</h2>
              <span className="formula">
                current MSE <strong>{loss.toFixed(3)}</strong>
              </span>
            </div>
            <ErrorChart history={errorHistory} />
            <p className="hint">Mean squared error over all samples after each iteration — lower is better.</p>
          </section>
        </div>

        {/* Right: data + decision boundary */}
        <section className="card">
          <div className="card-header">
            <h2>Data &amp; decision boundary</h2>
          </div>

          <div className="toolbar">
            <button className="btn primary" onClick={predict} disabled={busy || auto}>
              {busy ? 'Predicting…' : 'Predict'}
            </button>
            <button className={`btn ${auto ? 'active' : ''}`} onClick={() => setAuto((a) => !a)} aria-pressed={auto}>
              {auto ? '❚❚ Pause' : '▶ Auto'}
            </button>
            <select value={datasetKey} onChange={(e) => changeDataset(e.target.value as DatasetKey)} aria-label="Dataset">
              {(Object.keys(DATASETS) as DatasetKey[]).map((k) => (
                <option key={k} value={k}>
                  {DATASETS[k].label}
                </option>
              ))}
            </select>
            <button className="btn" onClick={newData}>
              New data
            </button>
            <button className="btn" onClick={resetWeights}>
              Reset weights
            </button>
          </div>
          <p className="hint">{DATASETS[datasetKey].hint}</p>

          <div className="stats">
            <div className="stat">
              <span className="stat-label">Accuracy</span>
              <span className="stat-value">{Math.round(acc * 100)}%</span>
            </div>
            <div className="stat">
              <span className="stat-label">Iteration</span>
              <span className="stat-value">{iteration}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Epoch</span>
              <span className="stat-value">{epoch}</span>
            </div>
          </div>

          <DecisionPlot
            points={points}
            weights={tweenedWeights}
            activation={activation}
            probe={inputs}
            highlight={highlight}
          />

          <ul className="legend">
            <li>
              <span className="swatch class-1" /> Class 1
            </li>
            <li>
              <span className="swatch class-0" /> Class 0
            </li>
            <li>
              <span className="legend-line" /> Decision boundary (z = 0)
            </li>
            <li>
              <span className="legend-miss" /> Misclassified
            </li>
            <li>
              <span className="legend-probe" /> Current inputs
            </li>
          </ul>

          <div className={`result ${last ? (last.correct ? 'ok' : 'bad') : ''}`} aria-live="polite">
            {busy && <span>Feeding sample through the neuron…</span>}
            {!busy && !last && <span>Press Predict to run the next sample.</span>}
            {!busy && last && lastPoint && (
              <>
                <div className="result-title">
                  {last.correct ? '✓ Correct' : '✗ Wrong'} — sample #{lastPoint.id} (x₁ {lastPoint.x1.toFixed(2)}, x₂{' '}
                  {lastPoint.x2.toFixed(2)}), target class {lastPoint.label}
                </div>
                <div>
                  z = {last.z.toFixed(2)} → y = f(z) = {last.y.toFixed(2)} → predicted class {last.predicted}
                </div>
                <div className="result-update">
                  {last.trained ? (
                    <>
                      error = {fmt(last.error)} · Δw₁ {fmt(last.delta.w1, 3)} · Δw₂ {fmt(last.delta.w2, 3)} · Δb{' '}
                      {fmt(last.delta.b, 3)}
                    </>
                  ) : (
                    'Weights unchanged (learning is off).'
                  )}
                </div>
              </>
            )}
          </div>

          <details className="data-table">
            <summary>Show data table</summary>
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>x₁</th>
                  <th>x₂</th>
                  <th>class</th>
                </tr>
              </thead>
              <tbody>
                {points.map((p) => (
                  <tr key={p.id}>
                    <td>{p.id}</td>
                    <td>{p.x1.toFixed(2)}</td>
                    <td>{p.x2.toFixed(2)}</td>
                    <td>{p.label}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        </section>
      </main>
    </div>
  );
}

import type { Point } from './data';

export type Weights = { w1: number; w2: number; b: number };

export type ActivationKey = 'step' | 'sigmoid' | 'tanh' | 'relu';

type Activation = {
  label: string;
  formula: string;
  fn: (z: number) => number;
  /** Output range used to draw the mini curve inside the neuron. */
  range: [number, number];
  /** Map a 0/1 label to the target value this activation should produce. */
  target: (label: 0 | 1) => number;
  /** Map an output to a 0..1 "confidence for class 1" used to shade the plot. */
  confidence: (z: number, y: number) => number;
};

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export const ACTIVATIONS: Record<ActivationKey, Activation> = {
  step: {
    label: 'Step',
    formula: 'f(z) = 1 if z ≥ 0, else 0',
    fn: (z) => (z >= 0 ? 1 : 0),
    range: [0, 1],
    target: (l) => l,
    confidence: (_z, y) => y,
  },
  sigmoid: {
    label: 'Sigmoid',
    formula: 'f(z) = 1 / (1 + e⁻ᶻ)',
    fn: sigmoid,
    range: [0, 1],
    target: (l) => l,
    confidence: (_z, y) => y,
  },
  tanh: {
    label: 'Tanh',
    formula: 'f(z) = tanh(z)',
    fn: Math.tanh,
    range: [-1, 1],
    target: (l) => (l === 1 ? 1 : -1),
    confidence: (_z, y) => (y + 1) / 2,
  },
  relu: {
    label: 'ReLU',
    formula: 'f(z) = max(0, z)',
    fn: (z) => Math.max(0, z),
    range: [0, 4],
    target: (l) => l,
    confidence: (z) => 0.5 + clamp(z, -1, 1) / 2,
  },
};

export const WEIGHT_LIMIT = 5;

export function weightedSum(w: Weights, x1: number, x2: number) {
  return w.w1 * x1 + w.w2 * x2 + w.b;
}

/** Every activation here is monotonic with its decision threshold at z = 0. */
export function forward(w: Weights, act: ActivationKey, x1: number, x2: number) {
  const z = weightedSum(w, x1, x2);
  const y = ACTIVATIONS[act].fn(z);
  const predicted: 0 | 1 = z >= 0 ? 1 : 0;
  return { z, y, predicted };
}

export type StepResult = ReturnType<typeof forward> & {
  target: number;
  error: number;
  correct: boolean;
  delta: Weights;
  next: Weights;
};

/** One delta-rule update: w ← w + η · (target − y) · x */
export function trainStep(w: Weights, act: ActivationKey, p: Point, lr: number): StepResult {
  const out = forward(w, act, p.x1, p.x2);
  const target = ACTIVATIONS[act].target(p.label);
  const error = target - out.y;
  const delta = { w1: lr * error * p.x1, w2: lr * error * p.x2, b: lr * error };
  const next = {
    w1: clamp(w.w1 + delta.w1, -WEIGHT_LIMIT, WEIGHT_LIMIT),
    w2: clamp(w.w2 + delta.w2, -WEIGHT_LIMIT, WEIGHT_LIMIT),
    b: clamp(w.b + delta.b, -WEIGHT_LIMIT, WEIGHT_LIMIT),
  };
  return { ...out, target, error, correct: out.predicted === p.label, delta, next };
}

export function accuracy(points: Point[], w: Weights) {
  if (points.length === 0) return 0;
  const hits = points.filter((p) => (weightedSum(w, p.x1, p.x2) >= 0 ? 1 : 0) === p.label).length;
  return hits / points.length;
}

/** The line w1·x1 + w2·x2 + b = 0 clipped to the [-1, 1] square. */
export function boundarySegment(w: Weights): [[number, number], [number, number]] | null {
  const pts: [number, number][] = [];
  const add = (x: number, y: number) => {
    if (x < -1 - 1e-9 || x > 1 + 1e-9 || y < -1 - 1e-9 || y > 1 + 1e-9) return;
    if (pts.some(([px, py]) => Math.abs(px - x) < 1e-6 && Math.abs(py - y) < 1e-6)) return;
    pts.push([x, y]);
  };
  if (Math.abs(w.w2) > 1e-9) {
    add(-1, (-w.b + w.w1) / w.w2);
    add(1, (-w.b - w.w1) / w.w2);
  }
  if (Math.abs(w.w1) > 1e-9) {
    add((-w.b + w.w2) / w.w1, -1);
    add((-w.b - w.w2) / w.w1, 1);
  }
  return pts.length >= 2 ? [pts[0], pts[1]] : null;
}

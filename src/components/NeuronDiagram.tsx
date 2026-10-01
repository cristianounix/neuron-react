import { ACTIVATIONS, forward, type ActivationKey, type Weights } from '../neuron';
import { useProgress } from '../hooks/useAnimation';

type Props = {
  inputs: { x1: number; x2: number };
  weights: Weights;
  activation: ActivationKey;
  runId: number | null;
  forwardMs: number;
};

type Vec = { x: number; y: number };

const INPUT_R = 26;
const SUM = { x: 280, y: 175, r: 36 };
const ACT_BOX = { x: 372, y: 145, w: 76, h: 60 };
const OUT = { x: 530, y: 175, r: 30 };
const INPUTS = [
  { key: 'x1', label: 'x₁', pos: { x: 70, y: 65 } },
  { key: 'x2', label: 'x₂', pos: { x: 70, y: 175 } },
  { key: 'b', label: '1', pos: { x: 70, y: 285 } },
] as const;

const lerp = (a: Vec, b: Vec, t: number): Vec => ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t });
const clamp01 = (t: number) => Math.min(1, Math.max(0, t));
const fmt = (v: number, d = 2) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);

/** Start and end points of a straight edge between two circles. */
function edge(from: Vec, fromR: number, to: Vec, toR: number): [Vec, Vec] {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const len = Math.hypot(dx, dy);
  const ux = dx / len;
  const uy = dy / len;
  return [
    { x: from.x + ux * fromR, y: from.y + uy * fromR },
    { x: to.x - ux * toR, y: to.y - uy * toR },
  ];
}

const Pulse = ({ at }: { at: Vec }) => <circle cx={at.x} cy={at.y} r={6} className="nd-pulse" />;

function ActivationCurve({ activation, z }: { activation: ActivationKey; z: number }) {
  const { fn, range } = ACTIVATIONS[activation];
  const pad = 9;
  const sx = (v: number) => ACT_BOX.x + pad + ((v + 4) / 8) * (ACT_BOX.w - pad * 2);
  const sy = (v: number) =>
    ACT_BOX.y + ACT_BOX.h - pad - ((Math.min(v, range[1]) - range[0]) / (range[1] - range[0])) * (ACT_BOX.h - pad * 2);
  const path = Array.from({ length: 81 }, (_, i) => {
    const v = -4 + i * 0.1;
    return `${i === 0 ? 'M' : 'L'}${sx(v).toFixed(1)},${sy(fn(v)).toFixed(1)}`;
  }).join(' ');
  const zc = Math.max(-4, Math.min(4, z));

  return (
    <g>
      <line x1={sx(0)} x2={sx(0)} y1={ACT_BOX.y + 6} y2={ACT_BOX.y + ACT_BOX.h - 6} className="nd-axis" />
      <path d={path} className="nd-curve" />
      <circle cx={sx(zc)} cy={sy(fn(zc))} r={4} className="nd-curve-dot" />
    </g>
  );
}

export function NeuronDiagram({ inputs, weights, activation, runId, forwardMs }: Props) {
  const t = useProgress(runId, forwardMs);
  const running = runId !== null && t < 1;
  const { z, y, predicted } = forward(weights, activation, inputs.x1, inputs.x2);

  const values = { x1: inputs.x1, x2: inputs.x2, b: 1 };
  const weightOf = { x1: weights.w1, x2: weights.w2, b: weights.b };
  const weightLabel = { x1: 'w₁', x2: 'w₂', b: 'b' };

  // Timeline of the forward-pass animation (fractions of forwardMs).
  const inPhase = clamp01(t / 0.45);
  const sumGlow = running && t >= 0.4 && t < 0.65;
  const toAct = clamp01((t - 0.55) / 0.2);
  const toOut = clamp01((t - 0.75) / 0.2);
  const outGlow = running && t >= 0.9;

  const actLeft = { x: ACT_BOX.x, y: SUM.y };
  const actRight = { x: ACT_BOX.x + ACT_BOX.w, y: SUM.y };
  const [sumOutA, sumOutB] = [{ x: SUM.x + SUM.r, y: SUM.y }, actLeft];
  const [actOutA, actOutB] = [actRight, { x: OUT.x - OUT.r, y: OUT.y }];

  return (
    <svg viewBox="0 0 600 330" className="neuron-diagram" role="img" aria-label="Artificial neuron diagram">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" className="nd-arrowhead" />
        </marker>
      </defs>

      {/* Edges: thickness follows |w|, dashed when the weight is negative */}
      {INPUTS.map(({ key, pos }) => {
        const w = weightOf[key];
        const [a, b] = edge(pos, INPUT_R, SUM, SUM.r);
        const mid = lerp(a, b, 0.5);
        return (
          <g key={key}>
            <line
              x1={a.x}
              y1={a.y}
              x2={b.x}
              y2={b.y}
              className={`nd-edge ${w < 0 ? 'negative' : ''}`}
              strokeWidth={1 + Math.min(Math.abs(w), 5) * 0.9}
              markerEnd="url(#arrow)"
            />
            <g transform={`translate(${mid.x}, ${mid.y})`}>
              <rect x={-36} y={-12} width={72} height={24} rx={12} className="nd-pill" />
              <text className="nd-pill-text" textAnchor="middle" dy="0.35em">
                {weightLabel[key]} {fmt(w)}
              </text>
            </g>
          </g>
        );
      })}

      <line x1={sumOutA.x} y1={sumOutA.y} x2={sumOutB.x - 2} y2={sumOutB.y} className="nd-edge" markerEnd="url(#arrow)" />
      <line x1={actOutA.x} y1={actOutA.y} x2={actOutB.x - 2} y2={actOutB.y} className="nd-edge" markerEnd="url(#arrow)" />

      {/* Input nodes */}
      {INPUTS.map(({ key, label, pos }) => (
        <g key={key} transform={`translate(${pos.x}, ${pos.y})`}>
          <circle r={INPUT_R} className={`nd-node ${key === 'b' ? 'bias' : ''}`} />
          <text className="nd-node-label" textAnchor="middle" dy={key === 'b' ? '0.35em' : '-0.15em'}>
            {label}
          </text>
          {key !== 'b' && (
            <text className="nd-node-value" textAnchor="middle" dy="1.15em">
              {values[key].toFixed(2)}
            </text>
          )}
          <text className="nd-caption" textAnchor="middle" y={INPUT_R + 16}>
            {key === 'b' ? 'bias input' : `input ${key === 'x1' ? '1' : '2'}`}
          </text>
        </g>
      ))}

      {/* Summation */}
      <g transform={`translate(${SUM.x}, ${SUM.y})`}>
        <circle r={SUM.r} className={`nd-node sum ${sumGlow ? 'glow' : ''}`} />
        <text className="nd-sigma" textAnchor="middle" dy="-0.05em">
          Σ
        </text>
        <text className="nd-node-value" textAnchor="middle" dy="1.5em">
          z = {z.toFixed(2)}
        </text>
        <text className="nd-caption" textAnchor="middle" y={SUM.r + 18}>
          weighted sum
        </text>
      </g>

      {/* Activation */}
      <g>
        <rect
          x={ACT_BOX.x}
          y={ACT_BOX.y}
          width={ACT_BOX.w}
          height={ACT_BOX.h}
          rx={10}
          className={`nd-node act ${toAct >= 1 && toOut < 1 && running ? 'glow' : ''}`}
        />
        <ActivationCurve activation={activation} z={z} />
        <text className="nd-caption" textAnchor="middle" x={ACT_BOX.x + ACT_BOX.w / 2} y={ACT_BOX.y + ACT_BOX.h + 18}>
          {ACTIVATIONS[activation].label.toLowerCase()} f(z)
        </text>
      </g>

      {/* Output */}
      <g transform={`translate(${OUT.x}, ${OUT.y})`}>
        <circle r={OUT.r} className={`nd-node out class-${predicted} ${outGlow ? 'glow' : ''}`} />
        <text className="nd-node-label" textAnchor="middle" dy="-0.15em">
          y
        </text>
        <text className="nd-node-value" textAnchor="middle" dy="1.15em">
          {y.toFixed(2)}
        </text>
        <text className="nd-caption" textAnchor="middle" y={OUT.r + 18}>
          output → class {predicted}
        </text>
      </g>

      {/* Signal pulses travelling through the neuron */}
      {running && (
        <g className="nd-pulses">
          {inPhase < 1 &&
            INPUTS.map(({ key, pos }) => {
              const [a, b] = edge(pos, INPUT_R, SUM, SUM.r);
              return <Pulse key={key} at={lerp(a, b, inPhase)} />;
            })}
          {toAct > 0 && toAct < 1 && (
            <Pulse at={lerp(sumOutA, sumOutB, toAct)} />
          )}
          {toOut > 0 && toOut < 1 && (
            <Pulse at={lerp(actOutA, actOutB, toOut)} />
          )}
        </g>
      )}
    </svg>
  );
}

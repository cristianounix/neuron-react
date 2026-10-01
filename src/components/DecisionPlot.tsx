import { useMemo, useRef, useState } from 'react';
import type { Point } from '../data';
import { ACTIVATIONS, boundarySegment, forward, type ActivationKey, type Weights } from '../neuron';

type Highlight = { pointId: number; runId: number; correct: boolean | null };

type Props = {
  points: Point[];
  weights: Weights;
  activation: ActivationKey;
  probe: { x1: number; x2: number };
  highlight: Highlight | null;
};

const SIZE = 500;
const M = { top: 16, right: 16, bottom: 44, left: 48 };
const INNER = SIZE - M.left - M.right; // square plot area
const HEAT_RES = 48;
const TICKS = [-1, -0.5, 0, 0.5, 1];

const sx = (v: number) => M.left + ((v + 1) / 2) * INNER;
const sy = (v: number) => M.top + ((1 - v) / 2) * INNER;

// Class colours (validated categorical slots 1 & 2), used for the background wash.
const CLASS_RGB = { 1: [42, 120, 214], 0: [235, 104, 52] } as const;

function useHeatmap(weights: Weights, activation: ActivationKey) {
  const canvas = useRef<HTMLCanvasElement | null>(null);
  return useMemo(() => {
    if (typeof document === 'undefined') return '';
    if (!canvas.current) {
      canvas.current = document.createElement('canvas');
      canvas.current.width = HEAT_RES;
      canvas.current.height = HEAT_RES;
    }
    const ctx = canvas.current.getContext('2d');
    if (!ctx) return '';
    const img = ctx.createImageData(HEAT_RES, HEAT_RES);
    const { confidence } = ACTIVATIONS[activation];
    for (let row = 0; row < HEAT_RES; row++) {
      const x2 = 1 - ((row + 0.5) / HEAT_RES) * 2;
      for (let col = 0; col < HEAT_RES; col++) {
        const x1 = ((col + 0.5) / HEAT_RES) * 2 - 1;
        const { z, y } = forward(weights, activation, x1, x2);
        const c = confidence(z, y);
        const [r, g, b] = CLASS_RGB[c >= 0.5 ? 1 : 0];
        const i = (row * HEAT_RES + col) * 4;
        img.data[i] = r;
        img.data[i + 1] = g;
        img.data[i + 2] = b;
        img.data[i + 3] = Math.round((0.06 + Math.abs(c - 0.5) * 2 * 0.22) * 255);
      }
    }
    ctx.putImageData(img, 0, 0);
    return canvas.current.toDataURL();
  }, [weights, activation]);
}

function Marker({ p, size = 9 }: { p: Point; size?: number }) {
  return p.label === 1 ? (
    <circle cx={sx(p.x1)} cy={sy(p.x2)} r={size / 2 + 1} className="dp-point class-1" />
  ) : (
    <rect
      x={sx(p.x1) - size / 2}
      y={sy(p.x2) - size / 2}
      width={size}
      height={size}
      rx={2}
      className="dp-point class-0"
    />
  );
}

export function DecisionPlot({ points, weights, activation, probe, highlight }: Props) {
  const heatmap = useHeatmap(weights, activation);
  const segment = boundarySegment(weights);
  const [hover, setHover] = useState<Point | null>(null);

  const misclassified = useMemo(
    () => new Set(points.filter((p) => forward(weights, activation, p.x1, p.x2).predicted !== p.label).map((p) => p.id)),
    [points, weights, activation],
  );

  const highlighted = highlight ? points.find((p) => p.id === highlight.pointId) : undefined;

  // Arrow from the boundary's midpoint along the weight vector (towards class 1).
  let normal: { x1: number; y1: number; x2: number; y2: number } | null = null;
  if (segment) {
    const len = Math.hypot(weights.w1, weights.w2);
    const mx = (segment[0][0] + segment[1][0]) / 2;
    const my = (segment[0][1] + segment[1][1]) / 2;
    normal = {
      x1: sx(mx),
      y1: sy(my),
      x2: sx(mx + (weights.w1 / len) * 0.18),
      y2: sy(my + (weights.w2 / len) * 0.18),
    };
  }

  const hoverOut = hover ? forward(weights, activation, hover.x1, hover.x2) : null;

  return (
    <div className="decision-plot">
      <svg viewBox={`0 0 ${SIZE} ${SIZE}`} role="img" aria-label="Scatter plot of the data and the neuron's decision boundary">
        <defs>
          <clipPath id="plot-area">
            <rect x={M.left} y={M.top} width={INNER} height={INNER} />
          </clipPath>
          <marker id="normal-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" className="dp-normal-head" />
          </marker>
        </defs>

        <rect x={M.left} y={M.top} width={INNER} height={INNER} className="dp-surface" />
        {heatmap && (
          <image href={heatmap} x={M.left} y={M.top} width={INNER} height={INNER} preserveAspectRatio="none" />
        )}

        {/* Grid + axes */}
        {TICKS.map((t) => (
          <g key={t}>
            <line x1={sx(t)} x2={sx(t)} y1={M.top} y2={M.top + INNER} className={t === 0 ? 'dp-axis-zero' : 'dp-grid'} />
            <line x1={M.left} x2={M.left + INNER} y1={sy(t)} y2={sy(t)} className={t === 0 ? 'dp-axis-zero' : 'dp-grid'} />
            <text x={sx(t)} y={M.top + INNER + 18} textAnchor="middle" className="dp-tick">
              {t}
            </text>
            <text x={M.left - 8} y={sy(t)} dy="0.35em" textAnchor="end" className="dp-tick">
              {t}
            </text>
          </g>
        ))}
        <text x={M.left + INNER / 2} y={SIZE - 6} textAnchor="middle" className="dp-axis-label">
          x₁
        </text>
        <text x={14} y={M.top + INNER / 2} textAnchor="middle" className="dp-axis-label" transform={`rotate(-90 14 ${M.top + INNER / 2})`}>
          x₂
        </text>

        <g clipPath="url(#plot-area)">
          {/* Probe: where the current inputs sit in the input space */}
          <g className="dp-probe" style={{ transform: `translate(${sx(probe.x1)}px, ${sy(probe.x2)}px)` }}>
            <line x1={-INNER * 2} x2={INNER * 2} y1={0} y2={0} />
            <line y1={-INNER * 2} y2={INNER * 2} x1={0} x2={0} />
          </g>

          {/* Decision boundary w1·x1 + w2·x2 + b = 0 */}
          {segment && (
            <line
              x1={sx(segment[0][0])}
              y1={sy(segment[0][1])}
              x2={sx(segment[1][0])}
              y2={sy(segment[1][1])}
              className="dp-boundary"
            />
          )}
          {normal && <line {...normal} className="dp-normal" markerEnd="url(#normal-arrow)" />}

          {points.map((p) => (
            <g key={p.id}>
              {misclassified.has(p.id) && <circle cx={sx(p.x1)} cy={sy(p.x2)} r={10} className="dp-miss" />}
              <Marker p={p} />
            </g>
          ))}

          {/* Predict animation: ripple on the sample being fed + verdict badge */}
          {highlighted && highlight && (
            <g key={highlight.runId}>
              <circle cx={sx(highlighted.x1)} cy={sy(highlighted.x2)} r={8} className="dp-ripple" />
              <circle cx={sx(highlighted.x1)} cy={sy(highlighted.x2)} r={8} className="dp-ripple delay" />
              <g className="dp-focus" style={{ transformOrigin: `${sx(highlighted.x1)}px ${sy(highlighted.x2)}px` }}>
                <Marker p={highlighted} size={13} />
              </g>
              {highlight.correct !== null && (
                <g
                  className="dp-verdict"
                  transform={`translate(${Math.min(Math.max(sx(highlighted.x1), M.left + 44), M.left + INNER - 44)}, ${
                    sy(highlighted.x2) < M.top + 40 ? sy(highlighted.x2) + 30 : sy(highlighted.x2) - 26
                  })`}
                >
                  <rect x={-42} y={-12} width={84} height={24} rx={12} className="dp-verdict-bg" />
                  <text textAnchor="middle" dy="0.35em" className="dp-verdict-text">
                    {highlight.correct ? '✓ correct' : '✗ wrong'}
                  </text>
                </g>
              )}
            </g>
          )}

          {/* Hover hit targets, larger than the marks */}
          {points.map((p) => (
            <circle
              key={`hit-${p.id}`}
              cx={sx(p.x1)}
              cy={sy(p.x2)}
              r={12}
              className="dp-hit"
              onMouseEnter={() => setHover(p)}
              onMouseLeave={() => setHover(null)}
            />
          ))}
        </g>
      </svg>

      {hover && hoverOut && (
        <div
          className="tooltip"
          style={{
            left: `${(sx(hover.x1) / SIZE) * 100}%`,
            top: `${(sy(hover.x2) / SIZE) * 100}%`,
          }}
        >
          <div className="tooltip-title">
            <span className={`swatch class-${hover.label}`} /> Sample #{hover.id} · class {hover.label}
          </div>
          <div className="tooltip-row">
            <span>x₁, x₂</span>
            <span>
              {hover.x1.toFixed(2)}, {hover.x2.toFixed(2)}
            </span>
          </div>
          <div className="tooltip-row">
            <span>z</span>
            <span>{hoverOut.z.toFixed(2)}</span>
          </div>
          <div className="tooltip-row">
            <span>y = f(z)</span>
            <span>{hoverOut.y.toFixed(2)}</span>
          </div>
          <div className="tooltip-row">
            <span>predicted</span>
            <span>
              class {hoverOut.predicted} {hoverOut.predicted === hover.label ? '✓' : '✗'}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

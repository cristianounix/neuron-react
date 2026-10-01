import { useState, type MouseEvent } from 'react';

export type ErrorSample = { iteration: number; loss: number; correct: boolean | null };

type Props = { history: ErrorSample[] };

const W = 600;
const H = 220;
const M = { top: 14, right: 64, bottom: 32, left: 48 };
const PW = W - M.left - M.right;
const PH = H - M.top - M.bottom;

/** Round a max value up to a "nice" axis bound and return evenly spaced ticks. */
function niceTicks(max: number, count = 4) {
  const raw = max / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const top = Math.ceil(max / step) * step;
  return Array.from({ length: Math.round(top / step) + 1 }, (_, i) => +(i * step).toFixed(10));
}

function iterationTicks(min: number, max: number, count = 6) {
  const span = Math.max(1, max - min);
  const step = Math.max(1, Math.ceil(span / count));
  const ticks: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max; v += step) ticks.push(v);
  return ticks;
}

export function ErrorChart({ history }: Props) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  if (history.length === 0) {
    return <div className="error-empty">Press Predict to start recording the error.</div>;
  }

  const first = history[0].iteration;
  const last = history[history.length - 1].iteration;
  const yTicks = niceTicks(Math.max(...history.map((h) => h.loss), 0.01));
  const yMax = yTicks[yTicks.length - 1];
  const xTicks = iterationTicks(first, last);

  const sx = (it: number) => M.left + (last === first ? PW / 2 : ((it - first) / (last - first)) * PW);
  const sy = (v: number) => M.top + PH - (v / yMax) * PH;

  const path = history.map((h, i) => `${i === 0 ? 'M' : 'L'}${sx(h.iteration).toFixed(1)},${sy(h.loss).toFixed(1)}`).join(' ');
  const area = `${path} L${sx(last).toFixed(1)},${sy(0)} L${sx(first).toFixed(1)},${sy(0)} Z`;
  const tail = history[history.length - 1];

  const onMove = (e: MouseEvent<SVGRectElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = first + ((e.clientX - rect.left) / rect.width) * (last - first);
    let best = 0;
    history.forEach((h, i) => {
      if (Math.abs(h.iteration - x) < Math.abs(history[best].iteration - x)) best = i;
    });
    setHoverIndex(best);
  };

  const hovered = hoverIndex !== null ? history[hoverIndex] : null;

  return (
    <div className="error-chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Mean squared error per iteration">
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={M.left} x2={M.left + PW} y1={sy(t)} y2={sy(t)} className={t === 0 ? 'ec-baseline' : 'ec-grid'} />
            <text x={M.left - 8} y={sy(t)} dy="0.35em" textAnchor="end" className="ec-tick">
              {t}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={t} x={sx(t)} y={H - 10} textAnchor="middle" className="ec-tick">
            {t}
          </text>
        ))}

        <path d={area} className="ec-area" />
        <path d={path} className="ec-line" />

        {/* Newest point, direct-labelled with its value */}
        <circle key={tail.iteration} cx={sx(tail.iteration)} cy={sy(tail.loss)} r={4.5} className="ec-dot ec-dot-new" />
        <text x={sx(tail.iteration) + 9} y={sy(tail.loss)} dy="0.35em" className="ec-value">
          {tail.loss.toFixed(3)}
        </text>

        {hovered && (
          <g>
            <line x1={sx(hovered.iteration)} x2={sx(hovered.iteration)} y1={M.top} y2={M.top + PH} className="ec-crosshair" />
            <circle cx={sx(hovered.iteration)} cy={sy(hovered.loss)} r={4.5} className="ec-dot" />
          </g>
        )}

        <rect
          x={M.left}
          y={M.top}
          width={PW}
          height={PH}
          className="ec-hit"
          onMouseMove={onMove}
          onMouseLeave={() => setHoverIndex(null)}
        />
      </svg>

      {hovered && (
        <div
          className="tooltip"
          style={{ left: `${(sx(hovered.iteration) / W) * 100}%`, top: `${(sy(hovered.loss) / H) * 100}%` }}
        >
          <div className="tooltip-title">Iteration {hovered.iteration}</div>
          <div className="tooltip-row">
            <span>MSE</span>
            <span>{hovered.loss.toFixed(4)}</span>
          </div>
          {hovered.correct !== null && (
            <div className="tooltip-row">
              <span>sample</span>
              <span>{hovered.correct ? '✓ correct' : '✗ wrong'}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

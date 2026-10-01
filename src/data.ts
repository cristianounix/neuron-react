export type Point = { id: number; x1: number; x2: number; label: 0 | 1 };

export type DatasetKey = 'blobs' | 'and' | 'or' | 'xor';

export const DATASETS: Record<DatasetKey, { label: string; hint: string }> = {
  blobs: { label: 'Two clusters', hint: 'Linearly separable — a single neuron can learn it.' },
  and: { label: 'AND gate', hint: 'Class 1 only when both inputs are high.' },
  or: { label: 'OR gate', hint: 'Class 1 when either input is high.' },
  xor: { label: 'XOR gate', hint: 'Not linearly separable — a single neuron cannot solve it.' },
};

/** Small seeded PRNG so a dataset is reproducible from its seed. */
export function mulberry32(seed: number) {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rand: () => number) {
  let u = 0;
  while (u === 0) u = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

const clampUnit = (v: number) => Math.max(-0.98, Math.min(0.98, v));

function shuffle<T>(arr: T[], rand: () => number) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

export function generateDataset(key: DatasetKey, seed: number): Point[] {
  const rand = mulberry32(seed);
  const raw: Omit<Point, 'id'>[] = [];

  if (key === 'blobs') {
    const angle = rand() * Math.PI * 2;
    const cx = Math.cos(angle) * 0.45;
    const cy = Math.sin(angle) * 0.45;
    for (let i = 0; i < 20; i++) {
      raw.push({ x1: clampUnit(cx + gaussian(rand) * 0.2), x2: clampUnit(cy + gaussian(rand) * 0.2), label: 1 });
      raw.push({ x1: clampUnit(-cx + gaussian(rand) * 0.2), x2: clampUnit(-cy + gaussian(rand) * 0.2), label: 0 });
    }
  } else {
    const gate = (a: boolean, b: boolean) =>
      key === 'and' ? a && b : key === 'or' ? a || b : a !== b;
    for (const a of [false, true]) {
      for (const b of [false, true]) {
        for (let i = 0; i < 8; i++) {
          raw.push({
            x1: clampUnit((a ? 0.6 : -0.6) + gaussian(rand) * 0.12),
            x2: clampUnit((b ? 0.6 : -0.6) + gaussian(rand) * 0.12),
            label: gate(a, b) ? 1 : 0,
          });
        }
      }
    }
  }

  return shuffle(raw, rand).map((p, id) => ({ ...p, id }));
}

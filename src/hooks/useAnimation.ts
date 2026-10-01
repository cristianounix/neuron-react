import { useEffect, useRef, useState } from 'react';

export const prefersReducedMotion = () =>
  typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const easeInOutCubic = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);

/** Smoothly interpolates every numeric field of `target` whenever it changes. */
export function useTween<T extends Record<string, number>>(target: T, duration: number): T {
  const [value, setValue] = useState(target);
  const current = useRef(target);
  const key = Object.values(target).join('|');

  useEffect(() => {
    const from = current.current;
    const ms = prefersReducedMotion() ? 0 : duration;
    const start = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const t = ms === 0 ? 1 : Math.min(1, (now - start) / ms);
      const e = easeInOutCubic(t);
      const next = {} as Record<string, number>;
      for (const k of Object.keys(target)) next[k] = from[k] + (target[k] - from[k]) * e;
      current.current = next as T;
      setValue(next as T);
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, duration]);

  return value;
}

/** Runs a 0 → 1 progress value for `duration` ms each time `runKey` changes. Returns 1 when idle. */
export function useProgress(runKey: number | null, duration: number) {
  const [t, setT] = useState(1);

  useEffect(() => {
    if (runKey === null) return;
    const ms = prefersReducedMotion() ? 0 : duration;
    const start = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const p = ms === 0 ? 1 : Math.min(1, (now - start) / ms);
      setT(p);
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    setT(0);
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [runKey, duration]);

  return runKey === null ? 1 : t;
}

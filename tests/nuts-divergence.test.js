/**
 * A divergent step must not poison the run.
 *
 * NUTS legitimately steps outside a parameter's support — past sigma = 0, say
 * — on its way to rejecting a trajectory. The Hamiltonian is non-finite there,
 * so e^{H0-H} is NaN. If that NaN reaches dual averaging it destroys the run:
 * hBar goes NaN, so does logStepSize, and stepSize = exp(NaN) = NaN
 * permanently. Every subsequent leapfrog then produces NaN positions, every
 * tree stops, the proposal is never accepted, and the chain freezes for the
 * rest of the sampling — returning a full-length trace of identical draws,
 * with acceptanceRate = NaN as the only symptom.
 *
 * A divergence has acceptance probability 0. That is what dual averaging must
 * see, so it shrinks the step size and the chain recovers.
 */

import { describe, expect, it, vi } from 'vitest';
import { Model } from '../src/model.js';
import { Normal } from '../src/distributions/normal.js';
import { HalfNormal } from '../src/distributions/halfnormal.js';
import { NUTS } from '../src/samplers/nuts.js';
import { setRandomSeed } from '../src/rng.js';

const Y = [28, 8, -3, 7, -1, 1, 18, 12];

const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const sd = (a) => { const m = mean(a); return Math.sqrt(mean(a.map((v) => (v - m) ** 2))); };

/** Location-scale model: the scale's support boundary is what gets stepped past. */
function run(priorScale, start, seed = 1) {
  const m = new Model();
  m.addVariable('mu', new Normal(0, 10));
  m.addVariable('s', new HalfNormal(priorScale));
  m.potential('y', (p) => {
    let acc = 0;
    for (const v of Y) {
      const z = (v - p.mu) / p.s;
      acc += -0.5 * z * z - Math.log(p.s);
    }
    return acc;
  });
  setRandomSeed(seed);
  return new NUTS({ stepSize: 0.05 }).sample(
    m, { mu: 0, s: start }, { nSamples: 600, nWarmup: 300 },
  );
}

describe('NUTS divergence handling', () => {
  // Whether a run hits the boundary is trajectory-dependent, not systematic:
  // a HalfNormal(10) started at 1 happened to survive, while the same prior
  // started at 0.5 or 5 froze. All three must work.
  it.each([
    ['HalfNormal(2), start 1', 2, 1],
    ['HalfNormal(10), start 1', 10, 1],
    ['HalfNormal(10), start 5', 10, 5],
    ['HalfNormal(10), start 0.5', 10, 0.5],
  ])('%s: the chain keeps moving', (_name, prior, start) => {
    const out = run(prior, start);
    expect(Number.isFinite(out.acceptanceRate)).toBe(true);
    expect(out.acceptanceRate).toBeGreaterThan(0.3);
    expect(Number.isFinite(out.stepSize)).toBe(true);
    // A frozen chain reports exactly zero spread, which is the tell.
    expect(sd(out.trace.mu)).toBeGreaterThan(0.1);
    expect(sd(out.trace.s)).toBeGreaterThan(0.1);
  });

  it('never reports a non-finite step size or acceptance rate', () => {
    for (const seed of [1, 7, 42, 1234]) {
      const out = run(10, 5, seed);
      expect(Number.isFinite(out.stepSize)).toBe(true);
      expect(Number.isFinite(out.acceptanceRate)).toBe(true);
    }
  });

  it('samples the location-scale posterior correctly', () => {
    // This used to pin the acceptance rate to 0.8624 and mu's spread to
    // 2.1112, to show the divergence guard was inert on a run that never
    // diverges. Those constants encoded one particular trajectory, and
    // sampling in the unconstrained parameterization legitimately changes it —
    // that is the point of the transform, not a regression. The invariant
    // worth asserting is statistical: the chain mixes and recovers the data.
    const out = run(2, 1);
    const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
    expect(out.acceptanceRate).toBeGreaterThan(0.6);
    expect(out.acceptanceRate).toBeLessThan(0.99);
    // Y has mean 8.75; the posterior mean should land near it.
    expect(Math.abs(mean(out.trace.mu) - 8.75)).toBeLessThan(6);
    expect(sd(out.trace.mu)).toBeGreaterThan(1);
  });

  it('produces draws inside the support', () => {
    const out = run(10, 5);
    expect(out.trace.s.every((v) => v > 0)).toBe(true);
    expect(out.trace.s.every(Number.isFinite)).toBe(true);
  });
});

describe('divergent transitions are counted and reported', () => {
  // The diagnostic R-hat and ESS cannot give. A funnel, a scale parameter
  // whose prior lets it approach zero while other parameters scale with it,
  // is where NUTS diverges; a well-behaved model should report none.
  it('a benign model reports zero', () => {
    const m = new Model();
    m.addVariable('mu', new Normal(0, 10));
    m.potential('y', (p) => Y.reduce((acc, v) => acc - 0.5 * (v - p.mu) ** 2, 0));
    setRandomSeed(3);
    const fit = new NUTS({ stepSize: 0.1 }).sample(m, { mu: 0 }, { nSamples: 300, nWarmup: 300 });
    expect(fit.divergences).toBe(0);
    expect(typeof fit.divergencesWarmup).toBe('number');
  });

  it('a centered funnel reports some, and warns once', () => {
    // Neal's funnel, centered: x ~ N(0, exp(v/2)) with v ~ N(0, 3). The
    // neck near v << 0 is where the step size fails. Sampled without the
    // transform on purpose (v is unbounded), with a step too coarse for the
    // neck, so divergences are certain.
    const m = new Model();
    m.addVariable('v', new Normal(0, 3));
    m.addVariable('x', new Normal(0, 1));
    m.potential('funnel', (p) => -0.5 * p.x * p.x * Math.exp(-p.v) - 0.5 * p.v);
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    setRandomSeed(3);
    const fit = new NUTS({ stepSize: 0.5, targetAcceptance: 0.6 }).sample(m, { v: 0, x: 0 }, { nSamples: 400, nWarmup: 200 });
    expect(fit.divergences).toBeGreaterThan(0);
    const reports = warn.mock.calls.filter(([msg]) => /divergent transition/.test(msg));
    expect(reports).toHaveLength(1);
    expect(reports[0][0]).toMatch(/after warmup/);
    warn.mockRestore();
  });

  it('a multi-chain run aggregates the counts and warns once', async () => {
    const m = new Model();
    m.addVariable('v', new Normal(0, 3));
    m.addVariable('x', new Normal(0, 1));
    m.potential('funnel', (p) => -0.5 * p.x * p.x * Math.exp(-p.v) - 0.5 * p.v);
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const fit = await new NUTS({ stepSize: 0.5, targetAcceptance: 0.6 }).sample(m, { v: 0, x: 0 },
      { chains: 2, nSamples: 400, nWarmup: 200, seed: 3, parallel: false });
    expect(fit.divergences).toHaveLength(2);
    expect(fit.divergences.reduce((a, b) => a + b, 0)).toBeGreaterThan(0);
    const reports = warn.mock.calls.filter(([msg]) => /divergent transition/.test(msg));
    expect(reports).toHaveLength(1);
    expect(reports[0][0]).toMatch(/across 2 chains/);
    warn.mockRestore();
  }, 30000);
});

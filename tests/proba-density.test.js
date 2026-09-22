/**
 * The observation models derive from proba's logDensity, and three
 * distributions mc did not have arrive with it. What is pinned: on plain
 * numbers logDensity is the sum of logProb, which is proba's scipy-validated
 * density; with a Var parameter its gradient matches finite differences of
 * that sum; and each new class round-trips its options-object constructor.
 */

import { describe, expect, it } from 'vitest';
import { exp, valueAndGrad } from '@tangent.to/grad';
import { Exponential, Normal, Poisson, StudentT } from '../src/distributions/index.js';
import { Model } from '../src/model.js';
import { NUTS } from '../src/samplers/nuts.js';

const total = (a) => (Array.isArray(a) ? a.reduce((x, y) => x + y, 0) : a);

const cases = {
  Exponential: { make: (p) => new Exponential(p.lambda), p: { lambda: 1.7 }, x: [0.2, 1.4, 3] },
  StudentT: { make: (p) => new StudentT(p.nu, p.mu, p.sigma), p: { nu: 3.5, mu: 0.4, sigma: 1.2 }, x: [-1, 0.5, 4] },
  Poisson: { make: (p) => new Poisson(p.lambda), p: { lambda: 2.3 }, x: [0, 2, 5] },
  Normal: { make: (p) => new Normal(p.mu, p.sigma), p: { mu: 0.4, sigma: 1.2 }, x: [-1, 0.5, 4] },
};

describe('logDensity is the sum of logProb, through proba', () => {
  for (const [name, { make, p, x }] of Object.entries(cases)) {
    it(name, () => {
      const d = make(p);
      expect(d.logDensity(x).data[0]).toBeCloseTo(total(d.logProb(x)), 10);
    });
  }
});

describe('logDensity differentiates in every parameter', () => {
  for (const [name, { make, p, x }] of Object.entries(cases)) {
    it(name, () => {
      const f = (q) => total(make(q).logProb(x));
      const g = valueAndGrad((q) => make(q).logDensity(x))(p).gradient;
      for (const k of Object.keys(p)) {
        const up = { ...p, [k]: p[k] + 1e-6 }, dn = { ...p, [k]: p[k] - 1e-6 };
        expect(g[k]).toBeCloseTo((f(up) - f(dn)) / 2e-6, 5);
      }
    });
  }
});

describe('the new classes', () => {
  it('take an options object with the same aliases as their siblings', () => {
    expect(new Exponential({ rate: 2 }).getParams()).toEqual({ lambda: 2 });
    expect(new StudentT({ df: 4, mean: 1, sd: 2 }).getParams()).toEqual({ nu: 4, mu: 1, sigma: 2 });
    expect(new Poisson({ rate: 3 }).getParams()).toEqual({ lambda: 3 });
    expect(new StudentT(4).name).toBe('StudentT');
  });

  it('a Poisson observation model on a log-rate samples', async () => {
    const counts = [2, 3, 1, 4, 2, 5, 3, 2];
    const m = new Model('counts');
    m.addVariable('logRate', new Normal(0, 2));
    m.observe('y', (v) => new Poisson(exp(v.logRate)), counts);
    const fit = await new NUTS({ stepSize: 0.1 }).sample(m, { logRate: 0 }, { nSamples: 60, nWarmup: 60, seed: 3, quiet: true });
    const draws = fit.trace.logRate;
    expect(draws.every(Number.isFinite)).toBe(true);
    // Posterior mean rate near the sample mean, 2.75.
    const rate = Math.exp(draws.reduce((a, b) => a + b, 0) / draws.length);
    expect(rate).toBeGreaterThan(2);
    expect(rate).toBeLessThan(3.6);
  });
});

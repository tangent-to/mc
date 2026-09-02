/**
 * Constrained-parameter transforms.
 *
 * A gradient sampler moves through ℝⁿ. Stepping a scale in (0, ∞) directly
 * walks it past zero, where the density is -Infinity and the gradient is
 * meaningless; the proposal is rejected, but the trajectory is wasted and
 * step-size adaptation is dragged down near the boundary. Stan and PyMC both
 * sample a transformed parameter instead, correcting the log-density by
 * log|dx/du| so the posterior is unchanged.
 */

import { describe, expect, it } from 'vitest';
import { makeTransform, supportOf, unconstrainedView } from '../src/transforms.js';
import { Model } from '../src/model.js';
import { Normal } from '../src/distributions/normal.js';
import { HalfNormal } from '../src/distributions/halfnormal.js';
import { Beta } from '../src/distributions/beta.js';
import { Uniform } from '../src/distributions/uniform.js';
import { Gamma } from '../src/distributions/gamma.js';
import { NUTS } from '../src/samplers/nuts.js';
import { setRandomSeed } from '../src/rng.js';

describe('supportOf', () => {
  it.each([
    ['HalfNormal', new HalfNormal(2), { lower: 0, upper: null }],
    ['Gamma', new Gamma(2, 3), { lower: 0, upper: null }],
    ['Beta', new Beta(2, 3), { lower: 0, upper: 1 }],
    ['Uniform', new Uniform(-1, 4), { lower: -1, upper: 4 }],
    ['Normal', new Normal(0, 1), { lower: null, upper: null }],
  ])('%s', (_n, dist, expected) => {
    expect(supportOf(dist)).toEqual(expected);
  });

  it('leaves an array-valued Uniform untransformed', () => {
    // Per-element bounds would need a per-element support; using one element's
    // for all of them would silently constrain the others wrongly.
    expect(supportOf(new Uniform([0, 1], [2, 3]))).toEqual({ lower: null, upper: null });
  });
});

describe('makeTransform', () => {
  const cases = [
    ['lower-bounded', { lower: 0, upper: null }, [0.01, 1, 7.5]],
    ['upper-bounded', { lower: null, upper: 5 }, [-3, 0, 4.9]],
    ['doubly bounded', { lower: -1, upper: 4 }, [-0.9, 1.5, 3.8]],
    ['unbounded', { lower: null, upper: null }, [-2, 0, 3]],
  ];

  it.each(cases)('%s: round-trips a value', (_n, support, xs) => {
    const T = makeTransform(support);
    for (const x of xs) expect(T.toConstrained(T.toUnconstrained(x))).toBeCloseTo(x, 10);
  });

  it.each(cases)('%s: logDetJacobian matches log|dx/du| numerically', (_n, support, xs) => {
    const T = makeTransform(support);
    for (const x of xs) {
      const u = T.toUnconstrained(x);
      const h = 1e-6;
      const numeric = Math.log(Math.abs((T.toConstrained(u + h) - T.toConstrained(u - h)) / (2 * h)));
      expect(T.logDetJacobian(u)).toBeCloseTo(numeric, 5);
    }
  });

  it.each(cases)('%s: dLogDetJacobian matches its own derivative', (_n, support, xs) => {
    const T = makeTransform(support);
    for (const x of xs) {
      const u = T.toUnconstrained(x);
      const h = 1e-6;
      const numeric = (T.logDetJacobian(u + h) - T.logDetJacobian(u - h)) / (2 * h);
      expect(T.dLogDetJacobian(u)).toBeCloseTo(numeric, 5);
    }
  });

  it('only the unbounded transform reports itself as the identity', () => {
    expect(makeTransform({ lower: null, upper: null }).isIdentity).toBe(true);
    expect(makeTransform({ lower: 0, upper: null }).isIdentity).toBe(false);
  });
});

describe('the model in unconstrained space', () => {
  const build = () => {
    const m = new Model();
    m.addVariable('mu', new Normal(0, 10));
    m.addVariable('sigma', new HalfNormal(2));
    m.addVariable('p', new Beta(2, 3));
    return m;
  };
  const X = { mu: 1.5, sigma: 2.0, p: 0.3 };

  it('round-trips parameters', () => {
    const m = build();
    const back = m.toConstrained(m.toUnconstrained(X));
    for (const k of Object.keys(X)) expect(back[k]).toBeCloseTo(X[k], 10);
  });

  it('gives gradients that match finite differences of the transformed density', () => {
    // The check that the Jacobian correction is both applied and differentiated.
    const m = build();
    const u = m.toUnconstrained(X);
    const { gradients } = m.logProbAndGradientUnconstrained(u);
    const f = (uu) => m.logProbAndGradientUnconstrained(uu).logProb;
    for (const k of Object.keys(u)) {
      const h = 1e-6;
      const fd = (f({ ...u, [k]: u[k] + h }) - f({ ...u, [k]: u[k] - h })) / (2 * h);
      expect(gradients[k]).toBeCloseTo(fd, 6);
    }
  });

  it('adds the log-Jacobian to the density, not nothing', () => {
    // Without the correction the sampler explores the transformed density
    // rather than the one the user wrote.
    const m = build();
    const u = m.toUnconstrained(X);
    const constrained = m.logProbAndGradient(X).logProb;
    const unconstrained = m.logProbAndGradientUnconstrained(u).logProb;
    // log|dsigma/du| = u_sigma, plus the Beta term.
    expect(unconstrained).not.toBeCloseTo(constrained, 6);
    expect(unconstrained - constrained).toBeCloseTo(
      makeTransform({ lower: 0, upper: null }).logDetJacobian(u.sigma)
        + makeTransform({ lower: 0, upper: 1 }).logDetJacobian(u.p),
      10,
    );
  });

  it('reports whether anything is constrained at all', () => {
    expect(build().hasConstrainedVariables()).toBe(true);
    const plain = new Model();
    plain.addVariable('a', new Normal(0, 1));
    plain.addVariable('b', new Normal(0, 1));
    expect(plain.hasConstrainedVariables()).toBe(false);
  });
});

describe('unconstrainedView', () => {
  it('returns the model itself when nothing is bounded', () => {
    const m = new Model();
    m.addVariable('a', new Normal(0, 1));
    expect(unconstrainedView(m)).toBe(m);
  });

  it('forwards the rest of the model API through the prototype chain', () => {
    // Listing methods by hand broke the moment a sampler reached for one that
    // was not on the list.
    const m = new Model();
    m.addVariable('s', new HalfNormal(1));
    const view = unconstrainedView(m);
    expect(view).not.toBe(m);
    expect(typeof view.getFreeVariableNames).toBe('function');
    expect(view.getFreeVariableNames()).toEqual(m.getFreeVariableNames());
    expect(view.variables).toBe(m.variables);
  });
});

describe('NUTS through the transform', () => {
  const Y = [4.1, 5.3, 3.8, 6.0, 4.7, 5.1, 4.4, 5.8, 4.9, 5.2];
  const fit = (seed) => {
    const m = new Model();
    m.addVariable('mu', new Normal(0, 10));
    m.addVariable('sigma', new HalfNormal(5));
    m.potential('y', (p) => {
      let acc = 0;
      for (const v of Y) {
        const z = (v - p.mu) / p.sigma;
        acc += -0.5 * z * z - Math.log(p.sigma);
      }
      return acc;
    });
    setRandomSeed(seed);
    return new NUTS({ stepSize: 0.05 }).sample(
      m, { mu: 0, sigma: 1 }, { nSamples: 1500, nWarmup: 800 },
    );
  };

  it('records draws in the model\'s own units, all inside the support', () => {
    const out = fit(3);
    expect(out.trace.sigma.every((v) => v > 0)).toBe(true);
    expect(out.trace.sigma.every(Number.isFinite)).toBe(true);
    expect(out.trace.mu.every(Number.isFinite)).toBe(true);
  });

  it('recovers the data', () => {
    const out = fit(3);
    const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
    const dataMean = mean(Y);
    const dataSd = Math.sqrt(mean(Y.map((v) => (v - dataMean) ** 2)));
    expect(Math.abs(mean(out.trace.mu) - dataMean)).toBeLessThan(0.6);
    expect(Math.abs(mean(out.trace.sigma) - dataSd)).toBeLessThan(0.6);
  });

  it('mixes without a frozen chain or a non-finite diagnostic', () => {
    const out = fit(11);
    const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
    const sd = (a) => { const m = mean(a); return Math.sqrt(mean(a.map((v) => (v - m) ** 2))); };
    expect(Number.isFinite(out.acceptanceRate)).toBe(true);
    expect(Number.isFinite(out.stepSize)).toBe(true);
    expect(sd(out.trace.sigma)).toBeGreaterThan(0.05);
  });
});

describe('every gradient sampler moves through the unconstrained space', () => {
  // Only NUTS did until 0.10. A HalfNormal scale stepped through directly by
  // leapfrog walks past zero, where the density is -Infinity; through the
  // transform it never can. So the test is that no draw of the scale is ever
  // outside its support, and the posterior is still the right one.
  const ys = [1.9, 2.4, 2.1, 1.7, 2.6, 2.2, 1.8, 2.3, 2.0, 2.5, 1.6, 2.7];
  const build = () => {
    const m = new Model();
    m.addVariable('mu', new Normal(0, 10));
    m.addVariable('sigma', new HalfNormal(2));
    m.observe('y', (v) => new Normal(v.mu, v.sigma), ys);
    return m;
  };
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;

  it('HamiltonianMC', async () => {
    const { HamiltonianMC } = await import('../src/samplers/hmc.js');
    const { setRandomSeed } = await import('../src/rng.js');
    setRandomSeed(2);
    // A fixed-step sampler needs a start it can move from; what is under test
    // is that the scale, once moving, never crosses zero.
    const fit = new HamiltonianMC({ stepSize: 0.02, nSteps: 15 }).sample(build(), { mu: 1, sigma: 0.5 }, { nSamples: 600, nWarmup: 300 });
    expect(Math.min(...fit.trace.sigma)).toBeGreaterThan(0);
    expect(mean(fit.trace.mu)).toBeCloseTo(2.15, 0);
    expect(fit.acceptanceRate).toBeGreaterThan(0.5);
  });

  it('the vector HMC', async () => {
    const { HMC } = await import('../src/samplers/hmc-vector.js');
    const { setRandomSeed } = await import('../src/rng.js');
    setRandomSeed(2);
    const fit = new HMC({ stepSize: 0.05, nSteps: 10 }).sample(build(), { mu: 0, sigma: 0.05 }, { nSamples: 400, nWarmup: 200 });
    expect(Math.min(...fit.trace.sigma)).toBeGreaterThan(0);
    expect(mean(fit.trace.mu)).toBeCloseTo(2.15, 0);
  });

  it('and records the draws on the natural scale, not the transformed one', () => {
    const { setRandomSeed } = require('../src/rng.js');
    setRandomSeed(2);
    const fit = new NUTS({ stepSize: 0.05 }).sample(build(), { mu: 0, sigma: 1 }, { nSamples: 200, nWarmup: 200 });
    // sigma's posterior sits near the sample SD, 0.35; a log-scale trace would sit near -1.
    expect(mean(fit.trace.sigma)).toBeCloseTo(0.35, 0);
    expect(Math.min(...fit.trace.sigma)).toBeGreaterThan(0);
  });
});

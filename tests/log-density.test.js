/**
 * `logDensity`: each distribution's log-density as a differentiable expression.
 *
 * Two references. On plain numbers it must equal the sum of `logProb`, which
 * is proba's validated density, to rounding: that pins the formula, including
 * every normalizing constant. With `Var` parameters its gradient must match
 * central finite differences of that same sum: that pins the adjoint path,
 * lgamma included.
 */

import { describe, expect, it } from 'vitest';
import { add, mul, valueAndGrad, variable } from '@tangent.to/grad';
import {
  Bernoulli, Beta, Gamma, HalfNormal, Lognormal, Normal, Uniform, Distribution,
} from '../src/distributions/index.js';

const total = (arr) => (Array.isArray(arr) ? arr.reduce((a, b) => a + b, 0) : arr);

/** Central finite differences of a scalar function of a parameter map. */
function fd(f, p, h = 1e-6) {
  const g = {};
  for (const k of Object.keys(p)) {
    const up = { ...p, [k]: p[k] + h };
    const dn = { ...p, [k]: p[k] - h };
    g[k] = (f(up) - f(dn)) / (2 * h);
  }
  return g;
}

describe('logDensity equals the sum of logProb on plain numbers', () => {
  const cases = [
    ['Normal scalar', () => new Normal(1.2, 0.7), 0.4],
    ['Normal vector value', () => new Normal(1.2, 0.7), [0.4, 1.9, -0.3]],
    ['Normal vector mean', () => new Normal([0, 1, 2], 0.7), [0.4, 1.9, -0.3]],
    ['Normal per-element sigma', () => new Normal(0, [0.5, 1, 2]), [0.4, 1.9, -0.3]],
    ['HalfNormal', () => new HalfNormal(0.8), [0.1, 0.9, 2.2]],
    ['Lognormal', () => new Lognormal(0.3, 0.6), [0.5, 1.4, 3.1]],
    ['Gamma', () => new Gamma(2.5, 1.7), [0.3, 1.1, 2.8]],
    ['Beta', () => new Beta(2.2, 3.4), [0.1, 0.45, 0.9]],
    ['Uniform', () => new Uniform(-1, 3), [-0.5, 0, 2.7]],
    ['Bernoulli', () => new Bernoulli(0.3), [0, 1, 1, 0]],
  ];
  for (const [name, make, x] of cases) {
    it(name, () => {
      const d = make();
      expect(d.logDensity(x).data[0]).toBeCloseTo(total(d.logProb(x)), 12);
    });
  }

  it('Uniform outside the support is -Infinity, as logProb is', () => {
    expect(new Uniform(0, 1).logDensity([0.5, 1.5]).data[0]).toBe(-Infinity);
  });
});

describe('logDensity differentiates in Var parameters', () => {
  // Each case: a parameter map, a builder taking (Var params) and returning
  // the distribution, and the observed value. The FD reference is the sum of
  // logProb at perturbed numeric parameters.
  const cases = [
    ['Normal in mu and sigma', { mu: 0.8, sigma: 0.6 }, (p) => new Normal(p.mu, p.sigma), [0.4, 1.9, -0.3]],
    ['HalfNormal in sigma', { sigma: 0.9 }, (p) => new HalfNormal(p.sigma), [0.1, 0.9, 2.2]],
    ['Lognormal in mu and sigma', { mu: 0.3, sigma: 0.6 }, (p) => new Lognormal(p.mu, p.sigma), [0.5, 1.4, 3.1]],
    ['Gamma in shape and rate', { alpha: 2.5, beta: 1.7 }, (p) => new Gamma(p.alpha, p.beta), [0.3, 1.1, 2.8]],
    ['Beta in both shapes', { alpha: 2.2, beta: 3.4 }, (p) => new Beta(p.alpha, p.beta), [0.1, 0.45, 0.9]],
    ['Uniform in its bounds', { lower: -1, upper: 3 }, (p) => new Uniform(p.lower, p.upper), [-0.5, 0, 2.7]],
    ['Bernoulli in p', { p: 0.3 }, (p) => new Bernoulli(p.p), [0, 1, 1, 0]],
  ];
  for (const [name, params, build, x] of cases) {
    it(name, () => {
      const { value, gradient } = valueAndGrad((v) => build(v).logDensity(x))(params);
      const numeric = (p) => total(build(p).logProb(x));
      expect(value).toBeCloseTo(numeric(params), 10);
      const ref = fd(numeric, params);
      for (const k of Object.keys(params)) expect(gradient[k], k).toBeCloseTo(ref[k], 5);
    });
  }

  it('a computed mean: the observation model a regression needs', () => {
    // mu = a + b x as an expression, sigma a Var. This is what observe() will
    // build, and the gradient has to flow through the mean into a and b.
    const xs = [0, 0.5, 1, 1.5];
    const ys = [1.1, 1.9, 3.2, 3.9];
    const f = (v) => new Normal(add(v.a, mul(v.b, xs)), v.sigma).logDensity(ys);
    const { gradient } = valueAndGrad(f)({ a: 1, b: 2, sigma: 0.5 });
    const numeric = (p) => total(new Normal(xs.map((x) => p.a + p.b * x), p.sigma).logProb(ys));
    const ref = fd(numeric, { a: 1, b: 2, sigma: 0.5 });
    expect(gradient.a).toBeCloseTo(ref.a, 5);
    expect(gradient.b).toBeCloseTo(ref.b, 5);
    expect(gradient.sigma).toBeCloseTo(ref.sigma, 5);
  });
});

describe('a Var is a parameter, not an options object', () => {
  it('new Normal(muVar, sigma) keeps muVar', () => {
    // isOptions() used to accept any non-array object, so a Var landed as
    // `{ mu: undefined }` and the distribution silently became Normal(0, 1).
    const mu = variable(2.5);
    const d = new Normal(mu, 1);
    expect(d.mu).toBe(mu);
    expect(d.logDensity(2.5).data[0]).toBeCloseTo(new Normal(2.5, 1).logProb(2.5), 12);
  });
});

describe('a distribution without logDensity', () => {
  it('says it cannot be differentiated rather than returning something', () => {
    class Custom extends Distribution {
      _params() { return {}; }
    }
    expect(() => new Custom('Custom').logDensity(1)).toThrow(/cannot be differentiated/);
  });
});

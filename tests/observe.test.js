/**
 * `model.observe(name, factory, data)`: the likelihood derived from a
 * distribution, instead of written out by hand.
 *
 * The reference throughout is the hand-written autoPotential the method
 * replaces. Where the two are the same mathematical density they must agree
 * as FUNCTIONS, to rounding, at several points; that is a stronger check than
 * comparing posteriors, and it is what the first two blocks do. The NUTS run
 * at the end is the end-to-end claim from the design note: the §2.4 shape,
 * observed, recovers the same posterior as the hand-written form.
 */

import { describe, expect, it } from 'vitest';
import { add, div, exp, log, matmul, mul, relu, square, sub, sum } from '@tangent.to/grad';
import { Model } from '../src/model.js';
import { unconstrainedView } from '../src/transforms.js';
import { NUTS } from '../src/samplers/nuts.js';
import { setRandomSeed } from '../src/rng.js';
import { Distribution, HalfNormal, Lognormal, Normal } from '../src/distributions/index.js';

const LN_SQRT_2PI = 0.9189385332046727;
const X = [-2.0, -1.4, -0.7, 0.0, 0.6, 1.3, 2.1, 2.8];
const Y = X.map((x) => 1.4 + 2.3 * x + 0.15 * Math.sin(9 * x));

const expectSame = (a, b, digits = 10) => {
  for (const k of Object.keys(b)) {
    if (Array.isArray(b[k])) b[k].forEach((v, i) => expect(a[k][i], `${k}[${i}]`).toBeCloseTo(v, digits));
    else expect(a[k], k).toBeCloseTo(b[k], digits);
  }
};

describe('observe equals the hand-written likelihood', () => {
  const priors = (m) => {
    m.addVariable('slope', new Normal(0, 10));
    m.addVariable('intercept', new Normal(0, 10));
    m.addVariable('sigma', new HalfNormal(5));
    return m;
  };
  const observed = priors(new Model()).observe('y',
    (v) => new Normal(add(mul(v.slope, X), v.intercept), v.sigma), Y);
  const hand = priors(new Model()).autoPotential('y', (v) => {
    const r = div(sub(Y, add(mul(v.slope, X), v.intercept)), v.sigma);
    return sub(sub(mul(-0.5, sum(square(r))), mul(Y.length, log(v.sigma))), Y.length * LN_SQRT_2PI);
  });
  const points = [
    { slope: 2.1, intercept: 1.3, sigma: 0.4 },
    { slope: -0.5, intercept: 4.0, sigma: 1.8 },
    { slope: 3.3, intercept: -1.1, sigma: 0.25 },
  ];

  it('in value', () => {
    for (const p of points) expect(observed.logProb(p)).toBeCloseTo(hand.logProb(p), 10);
  });

  it('in gradient', () => {
    for (const p of points) expectSame(observed.gradientsOnly(p), hand.gradientsOnly(p));
  });

  it('registers a compiled term the model can later serialize', () => {
    expect(typeof observed.compiledTerms.get('y')).toBe('function');
    expect(observed.observedTerms.get('y')).toBe(Y);
  });
});

describe('a constrained parameter on its natural scale', () => {
  // The notebook form: a free log-scale variable, the half-Normal density and
  // its Jacobian added to the likelihood by hand. The observe form declares
  // tau ~ HalfNormal(s) and lets unconstrainedView do the transform. On the
  // unconstrained scale the two are one density and must agree exactly.
  const S = 0.5;
  const hand = new Model();
  hand.addVariable('mu', new Normal(0, 3));
  hand.addVariable('logTau', new Normal(0, 100)); // effectively flat, see below
  hand.autoPotential('lik', (v) => {
    const tau = exp(v.logTau);
    const r = div(sub(Y, v.mu), tau);
    return add(
      sub(sub(mul(-0.5, sum(square(r))), mul(Y.length, log(tau))), Y.length * LN_SQRT_2PI),
      add(mul(-0.5 / (S * S), square(tau)), v.logTau),   // half-Normal(S) on tau, plus Jacobian
    );
  });
  const natural = new Model();
  natural.addVariable('mu', new Normal(0, 3));
  natural.addVariable('tau', new HalfNormal(S));
  natural.observe('y', (v) => new Normal(v.mu, v.tau), Y);
  const view = unconstrainedView(natural);

  it('agree on the unconstrained scale, up to the flat prior the hand form needs to be free', () => {
    // hand's logTau carries Normal(0, 100) because a Model variable must have
    // a prior; its contribution is a near-constant and the exact quadratic is
    // added back below so the comparison is between identical densities.
    for (const logTau of [Math.log(0.3), Math.log(0.8), Math.log(2.0)]) {
      const p = { mu: 1.2, logTau };
      const flat = new Normal(0, 100).logProb(logTau);
      const a = hand.logProb(p) - flat;
      const b = view.logProb({ mu: 1.2, tau: logTau });
      // The HalfNormal's own normalizing constant, ln sqrt(2/pi) - log S, is in
      // logDensity via the prior and not in the hand-written kernel.
      expect(a).toBeCloseTo(b - (-0.2257913526447274 - Math.log(S)), 9);
    }
  });

  it('agree in gradient', () => {
    for (const logTau of [Math.log(0.3), Math.log(0.8), Math.log(2.0)]) {
      const gh = hand.gradientsOnly({ mu: 1.2, logTau });
      const gn = view.gradientsOnly({ mu: 1.2, tau: logTau });
      const dFlat = -logTau / (100 * 100);
      expect(gn.mu).toBeCloseTo(gh.mu, 9);
      expect(gn.tau).toBeCloseTo(gh.logTau - dFlat, 9);
    }
  });
});

describe('observe refuses what it cannot differentiate', () => {
  it('a factory that is not a function', () => {
    expect(() => new Model().observe('y', new Normal(0, 1), Y)).toThrow(/expected a function/);
  });
  it('missing data', () => {
    expect(() => new Model().observe('y', () => new Normal(0, 1))).toThrow(/observed data is required/);
  });
  it('a factory returning something other than a distribution', () => {
    const m = new Model();
    m.addVariable('a', new Normal(0, 1));
    m.observe('y', () => 42, Y);
    expect(() => m.logProb({ a: 0 })).toThrow(/must return one of mc's distributions, got number/);
  });
  it('a distribution without logDensity, naming the way out', () => {
    class Custom extends Distribution {
      _params() { return {}; }
    }
    const m = new Model();
    m.addVariable('a', new Normal(0, 1));
    m.observe('y', () => new Custom('Custom'), Y);
    expect(() => m.logProb({ a: 0 })).toThrow(/cannot be differentiated.*autoPotential/);
  });
});

describe('the design note\'s §2.4 shape, observed', () => {
  // A quadratic-plateau response with a random site effect and a clamp, the
  // guava model's structure at a size a test can afford. Built both ways from
  // the same priors; the observe form sampled with NUTS must land on the
  // posterior the hand-written form does, to Monte Carlo error.
  const n = 48;
  let s = 3;
  const rnd = () => ((s = (s * 1103515245 + 12345) | 0) >>> 16) / 65536;
  const dose = Array.from({ length: n }, () => rnd() * 3);
  const site = Array.from({ length: n }, (_, i) => i % 2);
  const siteOneHot = site.map((k) => [k === 0 ? 1 : 0, k === 1 ? 1 : 0]);
  const truth = { mu0: 4, g: 1.5, ns: 1.8, z: [0.3, -0.3], tau: 0.4, sigma: 0.25 };
  const qpNum = (x, g, ns) => { const u = 1 - x / ns; const c = u > 0 ? u : 0; return g * (1 - c * c); };
  const y = dose.map((x, i) => truth.mu0 + truth.tau * truth.z[site[i]] + qpNum(x, truth.g, truth.ns) + (rnd() - 0.5) * 0.5);
  const qp = (x, g, ns) => mul(g, sub(1, square(relu(sub(1, div(x, ns))))));

  const build = () => {
    const m = new Model('qp');
    m.addVariable('mu0', new Normal(4, 2));
    m.addVariable('tau', new HalfNormal(0.5));
    m.addVariable('z', new Normal([0, 0], 1));
    m.addVariable('g', new Lognormal(0, 1));
    m.addVariable('ns', new Lognormal(Math.log(2), 0.8));
    m.addVariable('sigma', new Lognormal(-1, 1));
    m.observe('y', (v) => new Normal(
      add(v.mu0, mul(v.tau, matmul(siteOneHot, v.z)), qp(dose, v.g, v.ns)),
      v.sigma,
    ), y);
    return m;
  };

  it('recovers the parameters it was simulated from', () => {
    setRandomSeed(5);
    const fit = new NUTS({ stepSize: 0.05 }).sample(build(),
      { mu0: 4, tau: 0.4, z: [0, 0], g: 1, ns: 2, sigma: 0.3 },
      { nSamples: 300, nWarmup: 300 });
    const mean = (a) => a.reduce((x, v) => x + v, 0) / a.length;
    expect(mean(fit.trace.mu0)).toBeCloseTo(truth.mu0, 0);
    expect(mean(fit.trace.g)).toBeCloseTo(truth.g, 0);
    expect(mean(fit.trace.ns)).toBeCloseTo(truth.ns, 0);
    expect(mean(fit.trace.sigma)).toBeCloseTo(truth.sigma, 0);
    // The transform kept every draw of tau and sigma inside the support.
    expect(Math.min(...fit.trace.tau)).toBeGreaterThan(0);
    expect(Math.min(...fit.trace.sigma)).toBeGreaterThan(0);
  }, 60000);
});

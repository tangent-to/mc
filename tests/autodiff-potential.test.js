/**
 * `model.autoPotential(name, fn)` — potentials differentiated by
 * `@tangent.to/grad` instead of hand-derived or finite-differenced.
 *
 * The reference is the hand-derived closed form, not finite differences: the
 * whole point is that autodiff should reach the exact gradient, which central
 * differences only approach to ~1e-7.
 */

import { describe, expect, it } from 'vitest';
import { add, div, log, mul, square, sub, sum } from '@tangent.to/grad';
import { Model } from '../src/model.js';
import { Normal } from '../src/distributions/normal.js';
import { HalfNormal } from '../src/distributions/halfnormal.js';

// Gaussian regression: the canonical potential from Model#potential's docs.
const X = [-2.0, -1.4, -0.7, 0.0, 0.6, 1.3, 2.1, 2.8];
const Y = X.map((x) => 1.4 + 2.3 * x + 0.15 * Math.sin(9 * x));
const N = X.length;

// The full Gaussian log-density, normalizing constant included, so it is
// value-for-value the same term as `logLikPlain` below — not merely the same
// up to a constant. The constant has zero gradient either way.
const HALF_N_LOG_2PI = 0.5 * N * Math.log(2 * Math.PI);
const logLik = (p) => {
  const resid = sub(Y, add(mul(p.slope, X), p.intercept));
  const core = sub(mul(-0.5, sum(square(div(resid, p.sigma)))), mul(N, log(p.sigma)));
  return sub(core, HALF_N_LOG_2PI);
};

/** The same term's gradient, derived by hand. */
function handDerived(p) {
  let gS = 0, gI = 0, gSig = 0;
  for (let i = 0; i < N; i++) {
    const r = Y[i] - (p.slope * X[i] + p.intercept);
    gS += (r * X[i]) / p.sigma ** 2;
    gI += r / p.sigma ** 2;
    gSig += (r * r) / p.sigma ** 3;
  }
  return { slope: gS, intercept: gI, sigma: gSig - N / p.sigma };
}

/** The same term on plain numbers, for the finite-difference path. */
const logLikPlain = (p) =>
  new Normal(X.map((x) => p.slope * x + p.intercept), p.sigma).logProb(Y);

const withPriors = (m) => {
  m.addVariable('slope', new Normal(0, 10));
  m.addVariable('intercept', new Normal(0, 10));
  m.addVariable('sigma', new HalfNormal(5));
  return m;
};

const AT = { slope: 2.1, intercept: 1.3, sigma: 0.4 };

describe('autoPotential', () => {
  it('reaches the hand-derived gradient to machine precision', () => {
    const m = withPriors(new Model());
    m.autoPotential('y', logLik);
    // Isolate the potential's own contribution: a model with priors only.
    const priorsOnly = withPriors(new Model()).gradientsOnly(AT);
    const full = m.gradientsOnly(AT);
    const ref = handDerived(AT);

    for (const k of Object.keys(ref)) {
      expect(full[k] - priorsOnly[k]).toBeCloseTo(ref[k], 9);
    }
  });

  it('beats the finite-difference fallback on accuracy', () => {
    const ad = withPriors(new Model());
    ad.autoPotential('y', logLik);
    const fd = withPriors(new Model());
    fd.potential('y', logLikPlain);

    const priors = withPriors(new Model()).gradientsOnly(AT);
    const ref = handDerived(AT);
    const gAD = ad.gradientsOnly(AT);
    const gFD = fd.gradientsOnly(AT);

    let adErr = 0, fdErr = 0;
    for (const k of Object.keys(ref)) {
      adErr = Math.max(adErr, Math.abs(gAD[k] - priors[k] - ref[k]));
      fdErr = Math.max(fdErr, Math.abs(gFD[k] - priors[k] - ref[k]));
    }
    expect(adErr).toBeLessThan(1e-9);
    expect(fdErr).toBeGreaterThan(adErr * 100);
  });

  it('evaluates the likelihood ONCE per gradient, against 2·P for finite differences', () => {
    for (const [mode, expected] of [['ad', 1], ['fd', 6]]) {
      let calls = 0;
      const m = withPriors(new Model());
      if (mode === 'ad') {
        m.autoPotential('y', (p) => { calls++; return logLik(p); });
      } else {
        m.potential('y', (p) => { calls++; return logLikPlain(p); });
      }
      m.gradientsOnly(AT);
      expect(calls).toBe(expected); // 3 free parameters -> 2*3 = 6
    }
  });

  it('shares one evaluation between the value and gradient passes', () => {
    // logProbAndGradient calls the value pass and the gradient pass in turn.
    let calls = 0;
    const m = withPriors(new Model());
    m.autoPotential('y', (p) => { calls++; return logLik(p); });
    m.logProbAndGradient(AT);
    expect(calls).toBe(1);
  });

  it('agrees with the finite-difference path on the log-probability value', () => {
    const ad = withPriors(new Model());
    ad.autoPotential('y', logLik);
    const fd = withPriors(new Model());
    fd.potential('y', logLikPlain);
    expect(ad.logProbAndGradient(AT).logProb).toBeCloseTo(fd.logProbAndGradient(AT).logProb, 9);
  });

  it('survives a parameter stepped outside its support', () => {
    // NUTS pushes sigma past 0 on its way to rejecting a trajectory. The term
    // must report a non-finite value there, not throw and kill the run.
    const m = withPriors(new Model());
    m.autoPotential('y', logLik);
    expect(() => m.gradientsOnly({ ...AT, sigma: -0.5 })).not.toThrow();
    expect(Number.isFinite(m.logProbAndGradient({ ...AT, sigma: -0.5 }).logProb)).toBe(false);
  });

  it('composes with a second, finite-differenced potential', () => {
    const m = withPriors(new Model());
    m.autoPotential('y', logLik);
    m.potential('extra', (p) => -0.5 * p.slope * p.slope);
    const base = withPriors(new Model());
    base.autoPotential('y', logLik);
    const g = m.gradientsOnly(AT);
    const g0 = base.gradientsOnly(AT);
    expect(g.slope - g0.slope).toBeCloseTo(-AT.slope, 5);
  });
});

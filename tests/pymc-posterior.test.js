/**
 * Cross-validation against stored posteriors from a real PyMC run.
 *
 * This closes the task `tests/pymc-comparison.js` names in its own header:
 * that file recovers the DATA-GENERATING parameters, which checks that
 * inference is not broken, but cannot catch a sampler that is subtly biased —
 * a biased sampler still lands near the truth on easy problems. Comparing
 * against another mature implementation's posterior can.
 *
 * Fixtures come from `tests/generate_pymc_fixtures.py` (PyMC 6.3.1, 4000 draws
 * x 4 chains) and are committed, so this runs without a Python toolchain.
 *
 * Tolerances are in units of the POSTERIOR STANDARD DEVIATION, not absolute:
 * both sides are Monte Carlo estimates, so the question is whether they agree
 * within their own sampling noise.
 */

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { add, div, log, matmul, mul, square, sub, sum } from '@tangent.to/grad';
import { Model } from '../src/model.js';
import { Normal } from '../src/distributions/normal.js';
import { HalfNormal } from '../src/distributions/halfnormal.js';
import { NUTS } from '../src/samplers/nuts.js';
import { setRandomSeed } from '../src/rng.js';

const FIX = JSON.parse(
  readFileSync(new URL('./fixtures/pymc-posteriors.json', import.meta.url), 'utf8'),
);

const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const sd = (a) => { const m = mean(a); return Math.sqrt(mean(a.map((v) => (v - m) ** 2))); };

describe('multiple regression vs PyMC', () => {
  const { X, y, pymc } = FIX.linear_regression;
  const N = y.length;
  const P = X[0].length;

  const trace = (() => {
    const m = new Model();
    m.addVariable('beta', new Normal(new Array(P).fill(0), new Array(P).fill(5)));
    m.addVariable('sigma', new HalfNormal(2));
    m.autoPotential('y', (p) => {
      const z = div(sub(y, matmul(X, p.beta)), p.sigma);
      const core = sub(mul(-0.5, sum(square(z))), mul(N, log(p.sigma)));
      return sub(core, 0.5 * N * Math.log(2 * Math.PI));
    });
    setRandomSeed(20260901);
    return new NUTS({ stepSize: 0.01 }).sample(
      m, { beta: new Array(P).fill(0), sigma: 1 }, { nSamples: 2000, nWarmup: 1000 },
    ).trace;
  })();

  it('matches PyMC on every posterior mean, within Monte Carlo noise', () => {
    let worst = 0;
    for (let j = 0; j < P; j++) {
      const col = trace.beta.map((r) => r[j]);
      worst = Math.max(worst, Math.abs(mean(col) - pymc.beta_mean[j]) / pymc.beta_sd[j]);
    }
    worst = Math.max(worst, Math.abs(mean(trace.sigma) - pymc.sigma_mean) / pymc.sigma_sd);
    expect(worst).toBeLessThan(0.25);
  });

  it('matches PyMC on the posterior spread', () => {
    // A sampler that mixes badly lands the means near enough but understates
    // the width. Checking only the means would miss that.
    for (let j = 0; j < P; j++) {
      const col = trace.beta.map((r) => r[j]);
      expect(sd(col) / pymc.beta_sd[j]).toBeGreaterThan(0.75);
      expect(sd(col) / pymc.beta_sd[j]).toBeLessThan(1.35);
    }
    expect(sd(trace.sigma) / pymc.sigma_sd).toBeGreaterThan(0.75);
    expect(sd(trace.sigma) / pymc.sigma_sd).toBeLessThan(1.35);
  });
});

describe('eight schools vs PyMC', () => {
  // The standard hierarchical funnel. Non-centred, as PyMC's fixture is: the
  // centred form is hard for any sampler and would test the parameterization
  // rather than the implementation.
  const { y, sd: se, pymc } = FIX.eight_schools;
  const K = y.length;

  // Priors identical to the fixture's: mu ~ Normal(0, 10), tau ~ HalfNormal(10).
  // The parameterization is non-centred (thetaRaw ~ Normal(0,1),
  // theta = mu + tau*thetaRaw), which is a change of variables and leaves the
  // posterior for (mu, tau) unchanged — PyMC's fixture is centred. Getting the
  // PRIOR wrong, on the other hand, changes the posterior: an earlier draft of
  // this test used logTau ~ Normal(0, 2), i.e. a lognormal tau, and disagreed
  // with PyMC by a full posterior standard deviation for exactly that reason.
  const trace = (() => {
    const m = new Model();
    m.addVariable('mu', new Normal(0, 10));
    m.addVariable('tau', new HalfNormal(10));
    m.addVariable('thetaRaw', new Normal(new Array(K).fill(0), new Array(K).fill(1)));
    m.autoPotential('obs', (p) => {
      const theta = add(mul(p.tau, p.thetaRaw), p.mu);
      const z = div(sub(y, theta), se);
      return mul(-0.5, sum(square(z)));
    });
    setRandomSeed(20260901);
    return new NUTS({ stepSize: 0.05 }).sample(
      m,
      { mu: 0, tau: 5, thetaRaw: new Array(K).fill(0) },
      { nSamples: 4000, nWarmup: 2000 },
    ).trace;
  })();

  it('recovers PyMC\'s posterior for the population mean', () => {
    expect(Math.abs(mean(trace.mu) - pymc.mu_mean) / pymc.mu_sd).toBeLessThan(0.4);
  });

  it('recovers the between-school scale, the hard part of the funnel', () => {
    expect(Math.abs(mean(trace.tau) - pymc.tau_mean) / pymc.tau_sd).toBeLessThan(0.4);
  });
});

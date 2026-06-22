/**
 * Inference-correctness tests for @tangent.to/mc
 *
 * These are *parameter-recovery* checks: each test generates data from a known
 * data-generating process and asserts that inference recovers the generating
 * parameter — or, for the Beta-Bernoulli case, the closed-form conjugate
 * posterior mean — within MCMC tolerance.
 *
 * Coverage:
 *   - Samplers:      Metropolis-Hastings, HamiltonianMC, NUTS, vector HMC
 *   - Distributions: Normal, Uniform, Lognormal, HalfNormal, Gamma, Beta,
 *                    Bernoulli
 *
 * NOTE: references here are analytic / data-generating values, NOT outputs from
 * a real PyMC run. Cross-validation against stored PyMC posterior fixtures is a
 * separate, still-open task (see README/roadmap).
 */

import {
  Model,
  Normal,
  Uniform,
  Lognormal,
  HalfNormal,
  Gamma,
  Beta,
  Bernoulli,
  MetropolisHastings,
  HamiltonianMC,
  NUTS,
  HMC,
  summary
} from '../src/index.js';
import { exportTraceForBrowser, importTraceFromJSON } from '../src/utils/persistence.js';
import * as tf from '@tensorflow/tfjs';

// Deterministic RNG — seeds both data generation and the Metropolis/jStat
// proposals so MH-based recoveries are reproducible run to run.
Math.random = (() => {
  let s = 42;
  return () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
})();

// Standard normal via Box-Muller on the seeded RNG.
function randn() {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;

function assertClose(actual, expected, tolerance, name) {
  const diff = Math.abs(actual - expected);
  const relativeError = diff / Math.abs(expected);

  if (diff > tolerance && relativeError > 0.1) {
    console.log(`  FAIL: ${name}`);
    console.log(`    Expected: ${expected.toFixed(4)}`);
    console.log(`    Actual:   ${actual.toFixed(4)}`);
    console.log(`    Difference: ${diff.toFixed(4)} (${(relativeError * 100).toFixed(1)}%)`);
    return false;
  }
  console.log(`  PASS: ${name} (got ${actual.toFixed(4)}, expected ${expected.toFixed(4)})`);
  return true;
}

function computeStats(samples) {
  const n = samples.length;
  const m = samples.reduce((a, b) => a + b, 0) / n;
  const sorted = [...samples].sort((a, b) => a - b);
  const variance = samples.reduce((acc, val) => acc + (val - m) ** 2, 0) / n;
  return {
    mean: m,
    std: Math.sqrt(variance),
    q025: sorted[Math.floor(n * 0.025)],
    q975: sorted[Math.floor(n * 0.975)]
  };
}

// Posterior mean of a parameter from a classic sampler result ({ trace: {...} })
const classicMean = (res, name) => computeStats(res.trace[name]).mean;
// ...and from the vector HMC result via summary().
const vectorMean = (res, name) => summary([res]).find((r) => r.param === name).mean;

let totalTests = 0;
let passedTests = 0;
function check(actual, expected, tol, name) {
  totalTests += 1;
  if (assertClose(actual, expected, tol, name)) passedTests += 1;
}

console.log('=== @tangent.to/mc inference-correctness tests ===\n');

// ---------------------------------------------------------------------------
// Group A: sampler coverage
// Recover the mean of Normal data (known sigma) with every sampler. mu is
// unconstrained, so all four samplers handle it well.
// ---------------------------------------------------------------------------
console.log('Group A: samplers recover the mean of Normal data (known sigma)');
console.log('---------------------------------------------------------------');
{
  const n = 24;
  const trueSigma = 1.0;
  const yArr = Array.from({ length: n }, () => 5.0 + trueSigma * randn());
  const yMean = mean(yArr); // analytic posterior mean (≈) under the weak prior
  const y = tf.tensor1d(yArr);

  const build = () => {
    const m = new Model('mu_recovery');
    m.addVariable('mu', new Normal(0, 10));
    m.potential('y', (v) => new Normal(v.mu, trueSigma).logProb(y));
    return m;
  };

  const mh = new MetropolisHastings(0.3).sample(build(), { mu: 0 }, 2000, 1000, 1);
  check(classicMean(mh, 'mu'), yMean, 0.3, 'Metropolis-Hastings recovers mu');

  const hmc = new HamiltonianMC({ stepSize: 0.05, nSteps: 15 })
    .sample(build(), { mu: 0 }, { nSamples: 1000, burnIn: 500 });
  check(classicMean(hmc, 'mu'), yMean, 0.3, 'HamiltonianMC recovers mu');

  const nuts = new NUTS({ stepSize: 0.1, maxTreeDepth: 8, targetAcceptance: 0.8 })
    .sample(build(), { mu: 0 }, { nSamples: 600, nWarmup: 500 });
  check(classicMean(nuts, 'mu'), yMean, 0.3, 'NUTS recovers mu');

  const vhmc = new HMC({ stepSize: 0.05, nSteps: 15, seed: 7 })
    .sample(build(), { mu: 0 }, { nSamples: 400, nWarmup: 400 });
  check(vectorMean(vhmc, 'mu'), yMean, 0.3, 'vector HMC recovers mu');
}

console.log('');

// ---------------------------------------------------------------------------
// Group B: distribution coverage (parameter recovery via Metropolis-Hastings,
// which handles the constrained parameters via proposal rejection).
// ---------------------------------------------------------------------------
console.log('Group B: distributions recover their generating parameter');
console.log('---------------------------------------------------------');

// Normal likelihood (scale) + Uniform prior: recover sigma from centered data.
{
  const trueSigma = 1.5;
  const yArr = Array.from({ length: 40 }, () => trueSigma * randn());
  const y = tf.tensor1d(yArr);
  const mleSigma = Math.sqrt(mean(yArr.map((v) => v * v))); // MLE with known mean 0

  const model = new Model('normal_scale');
  model.addVariable('sigma', new Uniform(0.01, 5));
  model.potential('y', (v) => new Normal(0, v.sigma).logProb(y));
  const res = new MetropolisHastings(0.15).sample(model, { sigma: 1 }, 3000, 1500, 1);
  check(classicMean(res, 'sigma'), mleSigma, 0.3, 'Normal + Uniform: recover sigma');
}

// Lognormal: recover the (unconstrained) log-location with known log-scale.
{
  const trueM = 0.5;
  const s = 0.4;
  const yArr = Array.from({ length: 40 }, () => Math.exp(trueM + s * randn()));
  const y = tf.tensor1d(yArr);
  const mleM = mean(yArr.map((v) => Math.log(v))); // MLE log-location

  const model = new Model('lognormal');
  model.addVariable('m', new Normal(0, 10));
  model.potential('y', (v) => new Lognormal(v.m, s).logProb(y));
  const res = new MetropolisHastings(0.1).sample(model, { m: 0 }, 3000, 1500, 1);
  check(classicMean(res, 'm'), mleM, 0.2, 'Lognormal: recover log-location');
}

// HalfNormal: recover the scale of folded-normal data.
{
  const trueSigma = 2.0;
  const yArr = Array.from({ length: 40 }, () => Math.abs(trueSigma * randn()));
  const y = tf.tensor1d(yArr);
  const mleSigma = Math.sqrt(mean(yArr.map((v) => v * v))); // MLE: sigma^2 = mean(y^2)

  const model = new Model('halfnormal');
  model.addVariable('sigma', new Uniform(0.01, 10));
  model.potential('y', (v) => new HalfNormal(v.sigma).logProb(y));
  const res = new MetropolisHastings(0.25).sample(model, { sigma: 1 }, 3000, 1500, 1);
  check(classicMean(res, 'sigma'), mleSigma, 0.4, 'HalfNormal: recover sigma');
}

// Gamma: recover the rate with shape fixed.
{
  const shape = 3;
  const trueRate = 2.0;
  const yArr = Array.from({ length: 50 }, () => {
    // Gamma(shape, rate) as a sum of `shape` Exp(rate) draws (inverse-CDF).
    let g = 0;
    for (let k = 0; k < shape; k += 1) g += -Math.log(Math.random()) / trueRate;
    return g;
  });
  const y = tf.tensor1d(yArr);
  const mleRate = shape / mean(yArr); // MLE of rate with known shape

  const model = new Model('gamma');
  model.addVariable('rate', new Uniform(0.01, 10));
  model.potential('y', (v) => new Gamma(shape, v.rate).logProb(y));
  const res = new MetropolisHastings(0.2).sample(model, { rate: 1 }, 3000, 1500, 1);
  check(classicMean(res, 'rate'), mleRate, 0.4, 'Gamma: recover rate');
}

// Beta prior + Bernoulli likelihood: recover p, checked against the EXACT
// conjugate posterior mean (a0 + k) / (a0 + b0 + n).
{
  const a0 = 2;
  const b0 = 2;
  const n = 30;
  const k = 21; // successes
  const data = tf.tensor1d([...Array(k).fill(1), ...Array(n - k).fill(0)]);
  const posteriorMean = (a0 + k) / (a0 + b0 + n);

  const model = new Model('beta_bernoulli');
  model.addVariable('p', new Beta(a0, b0));
  model.potential('y', (v) => new Bernoulli(v.p).logProb(data));
  const res = new MetropolisHastings(0.08).sample(model, { p: 0.5 }, 4000, 2000, 1);
  check(classicMean(res, 'p'), posteriorMean, 0.05, 'Beta-Bernoulli: recover p (analytic posterior)');
}

console.log('');

// ---------------------------------------------------------------------------
// Group C: posterior predictive
// ---------------------------------------------------------------------------
console.log('Group C: posterior predictive');
console.log('-----------------------------');
{
  const trueData = [2.1, 1.9, 2.3, 1.8, 2.0, 2.2];
  const y = tf.tensor1d(trueData);

  const model = new Model('mean_estimation');
  model.addVariable('mu', new Normal(0, 10));
  model.addVariable('sigma', new Uniform(0.01, 5));
  model.potential('y', (v) => new Normal(v.mu, v.sigma).logProb(y));

  const trace = new MetropolisHastings(0.3).sample(model, { mu: 0, sigma: 1 }, 1500, 800, 1);
  const predictions = model.predictPosterior(trace, (p) => p.mu, 200);
  check(computeStats(predictions).mean, mean(trueData), 0.3, 'posterior predictive mean ≈ data mean');
}

console.log('');

// ---------------------------------------------------------------------------
// Group D: trace persistence round-trip
// ---------------------------------------------------------------------------
console.log('Group D: trace persistence round-trip');
console.log('-------------------------------------');
{
  const mockTrace = {
    trace: { param1: [1, 2, 3, 4, 5], param2: [5, 4, 3, 2, 1] },
    acceptanceRate: 0.35,
    nSamples: 5
  };

  const loaded = importTraceFromJSON(exportTraceForBrowser(mockTrace));

  totalTests += 3;
  if (JSON.stringify(loaded.trace) === JSON.stringify(mockTrace.trace)) {
    console.log('  PASS: trace data preserved');
    passedTests += 1;
  } else {
    console.log('  FAIL: trace data corrupted');
  }
  if (loaded.metadata.acceptanceRate === mockTrace.acceptanceRate) {
    console.log('  PASS: metadata preserved');
    passedTests += 1;
  } else {
    console.log('  FAIL: metadata corrupted');
  }
  if (loaded.metadata.timestamp) {
    console.log('  PASS: timestamp added');
    passedTests += 1;
  } else {
    console.log('  FAIL: no timestamp');
  }
}

console.log('');
console.log('='.repeat(60));
console.log(`Test Summary: ${passedTests}/${totalTests} tests passed`);

if (passedTests === totalTests) {
  console.log('Status: ALL TESTS PASSED');
  process.exit(0);
} else {
  console.log(`Status: ${totalTests - passedTests} TESTS FAILED`);
  process.exit(1);
}

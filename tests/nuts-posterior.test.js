import { NUTS } from '../src/samplers/index.js';
import { Model } from '../src/model.js';
import { Normal } from '../src/distributions/index.js';

/**
 * Regression test for the NUTS slice-membership criterion.
 *
 * With a mis-scaled slice test the sampler recovers the posterior MEAN but
 * inflates the posterior STANDARD DEVIATION (~40% on this conjugate Gaussian:
 * NUTS sd ≈ 0.31 vs analytic 0.224). This test pins the spread, so a regression
 * of the energy-weighting logic is caught.
 */
describe('NUTS recovers a conjugate Gaussian posterior', () => {
  test('posterior mean AND standard deviation match the analytic values', () => {
    // Deterministic pseudo-data x_i ~ N(5, 2) via an LCG (no RNG dependence).
    let s = 12345;
    const u = () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296;
    const randn = () => Math.sqrt(-2 * Math.log(u() || 1e-9)) * Math.cos(2 * Math.PI * u());
    const n = 60;
    const data = Array.from({ length: n }, () => 5 + 2 * randn());

    // Conjugate posterior for the mean with KNOWN sigma = 2 and prior N(0, 10):
    //   precision = n/sigma^2 + 1/priorVar ; mean = (Σx / sigma^2) / precision
    const sigma2 = 4, priorVar = 100;
    const prec = n / sigma2 + 1 / priorVar;
    const analyticMean = (data.reduce((a, b) => a + b, 0) / sigma2) / prec;
    const analyticSd = Math.sqrt(1 / prec);

    const yT = data;
    const model = new Model('conjugate-normal');
    model.addVariable('mu', new Normal(0, 10));
    model.potential('lik', (p) => new Normal(p.mu, 2).logProb(yT));

    const nuts = new NUTS({ stepSize: 0.05, maxTreeDepth: 8, targetAcceptance: 0.8 });
    const fit = nuts.sample(model, { mu: 0 }, { nSamples: 1500, nWarmup: 1000 });

    const draws = fit.trace.mu;
    const mean = draws.reduce((a, b) => a + b, 0) / draws.length;
    const sd = Math.sqrt(draws.reduce((a, b) => a + (b - mean) ** 2, 0) / draws.length);

    expect(Math.abs(mean - analyticMean)).toBeLessThan(0.1);
    // The bug inflated this ratio to ~1.4; require it within 15% of analytic.
    expect(sd / analyticSd).toBeGreaterThan(0.85);
    expect(sd / analyticSd).toBeLessThan(1.15);

    // Reported acceptance is now the mean Metropolis probability near the target.
    expect(fit.acceptanceRate).toBeGreaterThan(0.5);
    expect(fit.acceptanceRate).toBeLessThanOrEqual(1);
  });
});

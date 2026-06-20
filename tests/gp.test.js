/**
 * Regression tests for the Gaussian Process.
 *
 * These exercise the fit -> predict -> sample -> samplePosterior path, which
 * goes through ml-matrix Cholesky decomposition. A floating-point asymmetry in
 * the kernel matrix used to make Cholesky throw "Matrix is not symmetric"; the
 * prior sampler also used a non-existent `Matrix.from`. These tests guard both.
 */
import { GaussianProcess, RBF, Matern32, Matern52 } from '../src/distributions/index.js';

const X = [[-2], [-1], [0], [1], [2]];
const y = [-1.8, -0.9, 0.1, 1.0, 2.1];

describe('GaussianProcess', () => {
  test('fit() succeeds (no Cholesky symmetry error) and returns this', () => {
    const gp = new GaussianProcess({ kernel: new RBF({ lengthScale: 1, variance: 1 }), noiseVariance: 0.1 });
    expect(gp.isFitted()).toBe(false);
    const result = gp.fit(X, y);
    expect(result).toBe(gp);
    expect(gp.isFitted()).toBe(true);
  });

  test('predict() returns mean and std of the right length', () => {
    const gp = new GaussianProcess({ kernel: new RBF({ lengthScale: 1, variance: 1 }), noiseVariance: 0.1 });
    gp.fit(X, y);

    const Xtest = [[-1.5], [0.5], [1.5]];
    const out = gp.predict(Xtest, { returnStd: true });

    expect(out.mean).toHaveLength(3);
    expect(out.std).toHaveLength(3);
    for (const s of out.std) {
      expect(s).toBeGreaterThanOrEqual(0);
      expect(Number.isFinite(s)).toBe(true);
    }
    // Prediction near a training point should track the data
    const near = gp.predict([[0]], { returnStd: false });
    expect(near.mean[0]).toBeCloseTo(0.1, 0);
  });

  test('samplePosterior() draws functions of the right shape', () => {
    const gp = new GaussianProcess({ kernel: new Matern32({ lengthScale: 1, variance: 1 }), noiseVariance: 0.1 });
    gp.fit(X, y);

    // Use test points that don't coincide with training inputs.
    const Xtest = [[-1.5], [0.3], [1.7]];
    const samples = gp.samplePosterior(Xtest, 4);
    expect(samples).toHaveLength(4);
    for (const s of samples) {
      expect(s).toHaveLength(3);
      expect(s.every(Number.isFinite)).toBe(true);
    }
  });

  test('sample() draws from the prior without throwing', () => {
    const gp = new GaussianProcess({ kernel: new Matern52({ lengthScale: 1, variance: 1 }) });
    const samples = gp.sample([[-1], [0], [1]], 2);
    expect(samples).toHaveLength(2);
    expect(samples[0]).toHaveLength(3);
    expect(samples[0].every(Number.isFinite)).toBe(true);
  });
});

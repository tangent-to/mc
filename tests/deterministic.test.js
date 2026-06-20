import * as tf from '@tensorflow/tfjs-node';
import { Model, Normal, Lognormal, HalfNormal, HMC, summary } from '../src/index.js';

describe('Lognormal distribution', () => {
  test('logProb matches the closed form', () => {
    const d = new Lognormal(0, 1);
    // log p(1) = -log(1) - log(1) - 0.5 log(2π) - 0 = -0.5 log(2π)
    expect(d.logProb(1).arraySync()).toBeCloseTo(-0.5 * Math.log(2 * Math.PI), 5);
    // x = e: log x = 1 -> -log(e) - 0.5 log(2π) - 0.5
    expect(d.logProb(Math.E).arraySync()).toBeCloseTo(-1 - 0.5 * Math.log(2 * Math.PI) - 0.5, 5);
  });
  test('sample returns positive values', () => {
    const s = new Lognormal(0, 1).sample([200]).arraySync();
    expect(s.every((v) => v > 0)).toBe(true);
  });
});

describe('HalfNormal distribution', () => {
  test('logProb matches the closed form and is -Inf below zero', () => {
    const d = new HalfNormal(1);
    // log p(0) = 0.5 log(2/π)
    expect(d.logProb(0).arraySync()).toBeCloseTo(0.5 * Math.log(2 / Math.PI), 5);
    const neg = d.logProb(tf.tensor1d([-1])).arraySync();
    expect(neg[0]).toBe(-Infinity);
  });
  test('sample returns non-negative values', () => {
    const s = new HalfNormal(2).sample([200]).arraySync();
    expect(s.every((v) => v >= 0)).toBe(true);
  });
});

describe('Model.potential (generic deterministic likelihood)', () => {
  test('adds to logProb and yields order-correct gradients', () => {
    const x = tf.tensor1d([0, 1, 2]);
    const yObs = tf.tensor1d([1, 3, 5]); // y = 1 + 2x
    const model = new Model();
    model.addVariable('a', new Normal(0, 10));
    model.addVariable('b', new Normal(0, 10));
    model.potential('lik', (v) => {
      const mu = tf.add(v.a, tf.mul(v.b, x));
      return new Normal(mu, 1).logProb(yObs);
    });

    const { logProb, gradients } = model.logProbAndGradient({ a: 1, b: 2 });
    // At the true params the residuals are zero -> likelihood is at its max;
    // gradient of the (Gaussian) likelihood wrt a,b is ~0, leaving only the
    // weak prior pull toward 0 (negative for positive a,b).
    expect(Number.isFinite(logProb)).toBe(true);
    expect(gradients.a.arraySync()).toBeCloseTo(-1 / 100, 5); // -a/sigma_prior^2
    expect(gradients.b.arraySync()).toBeCloseTo(-2 / 100, 5);
  });
});

describe('HMC vector sampler', () => {
  test('recovers the mean of a Normal likelihood', () => {
    const data = tf.tensor1d([4.6, 5.2, 4.9, 5.5, 5.0, 4.8, 5.3, 5.1]);
    const model = new Model();
    model.addVariable('mu', new Normal(0, 10));
    model.potential('y', (v) => new Normal(v.mu, 0.3).logProb(data));

    const hmc = new HMC({ stepSize: 0.05, nSteps: 15, seed: 1 });
    const res = hmc.sample(model, { mu: 0 }, { nSamples: 400, nWarmup: 400 });
    const rows = summary([res]);
    const muHat = rows.find((r) => r.param === 'mu').mean;
    expect(muHat).toBeCloseTo(5.05, 1); // sample mean of the data
    expect(rows[0].ess).toBeGreaterThan(0);
  }, 30000);

  test('handles a vector parameter (per-group means)', () => {
    // two groups, distinct means
    const g0 = [2, 2.2, 1.8, 2.1];
    const g1 = [7, 6.8, 7.2, 7.1];
    const y = tf.tensor1d([...g0, ...g1]);
    const idx = tf.tensor1d([0, 0, 0, 0, 1, 1, 1, 1], 'int32');
    const model = new Model();
    model.addVariable('groupMean', new Normal(0, 10)); // 2-vector prior
    model.potential('y', (v) => {
      const mu = tf.gather(v.groupMean, idx);
      return new Normal(mu, 0.3).logProb(y);
    });
    const hmc = new HMC({ stepSize: 0.05, nSteps: 15, seed: 2 });
    const res = hmc.sample(model, { groupMean: [0, 0] }, { nSamples: 400, nWarmup: 400 });
    const rows = summary([res]);
    expect(rows.find((r) => r.param === 'groupMean[0]').mean).toBeCloseTo(2.0, 0);
    expect(rows.find((r) => r.param === 'groupMean[1]').mean).toBeCloseTo(7.0, 0);
  }, 30000);
});

import { Normal, Uniform, Beta, Gamma, Bernoulli } from '../src/distributions/index.js';
import * as tf from '@tensorflow/tfjs-node';

describe('Normal Distribution', () => {
  test('constructor creates distribution with correct parameters', () => {
    const dist = new Normal(5, 2);
    expect(dist.mu.arraySync()).toBeCloseTo(5, 5);
    expect(dist.sigma.arraySync()).toBeCloseTo(2, 5);
  });

  test('logProb calculates correctly for standard normal', () => {
    const dist = new Normal(0, 1);
    const logProb = dist.logProb(0).arraySync();
    // log(1/sqrt(2π)) ≈ -0.9189
    expect(logProb).toBeCloseTo(-0.9189385, 4);
  });

  test('logProb calculates correctly for non-zero mean', () => {
    const dist = new Normal(5, 1);
    const logProb = dist.logProb(5).arraySync();
    expect(logProb).toBeCloseTo(-0.9189385, 4);
  });

  test('sample generates values', () => {
    const dist = new Normal(0, 1);
    const samples = dist.sample([100]);
    expect(samples.shape).toEqual([100]);
    samples.dispose();
  });

  test('mean returns correct value', () => {
    const dist = new Normal(5, 2);
    const mean = dist.mean().arraySync();
    expect(mean).toBeCloseTo(5, 5);
  });

  test('variance returns correct value', () => {
    const dist = new Normal(0, 2);
    const variance = dist.variance().arraySync();
    expect(variance).toBeCloseTo(4, 5);
  });

  test('observed data can be set', () => {
    const dist = new Normal(0, 1);
    dist.observe([1, 2, 3]);
    expect(dist.observed).toBeDefined();
    expect(dist.observed.shape).toEqual([3]);
    const observedArray = dist.observed.arraySync();
    expect(observedArray).toEqual([1, 2, 3]);
  });
});

describe('Uniform Distribution', () => {
  test('constructor creates distribution with correct bounds', () => {
    const dist = new Uniform(0, 10);
    expect(dist.lower.arraySync()).toBeCloseTo(0, 5);
    expect(dist.upper.arraySync()).toBeCloseTo(10, 5);
  });

  test('logProb inside bounds is correct', () => {
    const dist = new Uniform(0, 10);
    const logProb = dist.logProb(5).arraySync();
    // log(1/10) = -2.302585
    expect(logProb).toBeCloseTo(-2.302585, 4);
  });

  test('logProb outside bounds is -Infinity', () => {
    const dist = new Uniform(0, 10);
    const logProb = dist.logProb(15).arraySync();
    expect(logProb).toBe(-Infinity);
  });

  test('sample generates values within bounds', () => {
    const dist = new Uniform(0, 10);
    const samples = dist.sample([100]);
    const samplesArray = samples.arraySync();

    for (const sample of samplesArray) {
      expect(sample).toBeGreaterThanOrEqual(0);
      expect(sample).toBeLessThanOrEqual(10);
    }

    samples.dispose();
  });

  test('mean returns correct value', () => {
    const dist = new Uniform(0, 10);
    const mean = dist.mean().arraySync();
    expect(mean).toBeCloseTo(5, 5);
  });

  test('variance returns correct value', () => {
    const dist = new Uniform(0, 12);
    const variance = dist.variance().arraySync();
    // Var = (b-a)²/12 = 144/12 = 12
    expect(variance).toBeCloseTo(12, 5);
  });
});

describe('Beta Distribution', () => {
  test('logProb calculates correctly', () => {
    const dist = new Beta(2, 2);
    const logProb = dist.logProb(0.5).arraySync();
    expect(logProb).toBeCloseTo(0.405465, 1);
  });

  test('logProb at mid-range is finite', () => {
    const dist = new Beta(2, 2);
    const logProb = dist.logProb(0.5).arraySync();
    expect(isFinite(logProb)).toBe(true);
    expect(logProb).toBeGreaterThan(-10);
  });
  test('logProb calculates correctly', () => {
    const dist = new Beta(2, 2);
    const logProb = dist.logProb(0.5).arraySync();
    expect(logProb).toBeCloseTo(0.405465, 1);
  });

  test('logProb at mid-range is finite', () => {
    const dist = new Beta(2, 2);
    const logProb = dist.logProb(0.5).arraySync();
    expect(isFinite(logProb)).toBe(true);
    expect(logProb).toBeGreaterThan(-10);
  });
  test('logProb calculates correctly', () => {
    const dist = new Beta(2, 2);
    const logProb = dist.logProb(0.5).arraySync();
    expect(logProb).toBeCloseTo(0.405465, 1);
  });

  test('logProb at mid-range is finite', () => {
    const dist = new Beta(2, 2);
    const logProb = dist.logProb(0.5).arraySync();
    expect(isFinite(logProb)).toBe(true);
    expect(logProb).toBeGreaterThan(-10);
  });
  test('logProb calculates correctly', () => {
    const dist = new Beta(2, 2);
    const logProb = dist.logProb(0.5).arraySync();
    expect(logProb).toBeCloseTo(0.405465, 1);
  });

  test('logProb at mid-range is finite', () => {
    const dist = new Beta(2, 2);
    const logProb = dist.logProb(0.5).arraySync();
    expect(isFinite(logProb)).toBe(true);
    expect(logProb).toBeGreaterThan(-10);
  });
  test('logProb calculates correctly', () => {
    const dist = new Beta(2, 2);
    const logProb = dist.logProb(0.5).arraySync();
    expect(logProb).toBeCloseTo(0.405465, 1);
  });

  test('logProb at mid-range is finite', () => {
    const dist = new Beta(2, 2);
    const logProb = dist.logProb(0.5).arraySync();
    expect(isFinite(logProb)).toBe(true);
    expect(logProb).toBeGreaterThan(-10);
  });
  test('logProb calculates correctly', () => {
    const dist = new Beta(2, 2);
    const logProb = dist.logProb(0.5).arraySync();
    expect(logProb).toBeCloseTo(0.405465, 1);
  });

  test('logProb at mid-range is finite', () => {
    const dist = new Beta(2, 2);
    const logProb = dist.logProb(0.5).arraySync();
    expect(isFinite(logProb)).toBe(true);
    expect(logProb).toBeGreaterThan(-10);
  test('logProb for positive values is finite', () => {
    const dist = new Gamma(2, 1);
    const logProb = dist.logProb(1).arraySync();
    expect(isFinite(logProb)).toBe(true);
  });
  test('logProb for positive values is finite', () => {
    const dist = new Gamma(2, 1);
    const logProb = dist.logProb(1).arraySync();
    expect(isFinite(logProb)).toBe(true);
  });
  test('logProb for positive values is finite', () => {
    const dist = new Gamma(2, 1);
    const logProb = dist.logProb(1).arraySync();
    expect(isFinite(logProb)).toBe(true);
  });
  test('logProb for positive values is finite', () => {
    const dist = new Gamma(2, 1);
    const logProb = dist.logProb(1).arraySync();
    expect(isFinite(logProb)).toBe(true);
  });
  test('logProb for positive values is finite', () => {
    const dist = new Gamma(2, 1);
    const logProb = dist.logProb(1).arraySync();
    expect(isFinite(logProb)).toBe(true);
  });
  });

  test('logProb at mid-range is finite', () => {
    const dist = new Beta(2, 2);
    const logProb = dist.logProb(0.5).arraySync();
    expect(isFinite(logProb)).toBe(true);
    expect(logProb).toBeGreaterThan(-10);
  });
  test('logProb calculates correctly', () => {
    const dist = new Beta(2, 2);
    const logProb = dist.logProb(0.5).arraySync();
    expect(logProb).toBeCloseTo(0.405465, 1);
  });

  test('logProb at mid-range is finite', () => {
    const dist = new Beta(2, 2);
    const logProb = dist.logProb(0.5).arraySync();
    expect(isFinite(logProb)).toBe(true);
    expect(logProb).toBeGreaterThan(-10);
  });
    const logProb = dist.logProb(0.5).arraySync();
    // For Beta(2,2), pdf(0.5) = 6 * 0.5^1 * 0.5^1 = 1.5
    // log(1.5) ≈ 0.405
    expect(logProb).toBeCloseTo(0.405465, 3);
  });

  test('logProb at boundaries', () => {
    const dist = new Beta(2, 2);
    const logProb0 = dist.logProb(0).arraySync();
    const logProb1 = dist.logProb(1).arraySync();
    expect(logProb0).toBe(-Infinity);
    expect(logProb1).toBe(-Infinity);
  });

  test('sample generates values in [0, 1]', () => {
    const dist = new Beta(2, 5);
    const samples = dist.sample([100]);
    const samplesArray = samples.arraySync();

    for (const sample of samplesArray) {
      expect(sample).toBeGreaterThanOrEqual(0);
      expect(sample).toBeLessThanOrEqual(1);
    }

    samples.dispose();
  });

  test('mean returns correct value', () => {
    const dist = new Beta(2, 3);
    const mean = dist.mean().arraySync();
    // Mean = α / (α + β) = 2 / 5 = 0.4
    expect(mean).toBeCloseTo(0.4, 5);
  });
});

describe('Gamma Distribution', () => {
  test('constructor creates distribution with correct parameters', () => {
    const dist = new Gamma(2, 1);
    expect(dist.alpha.arraySync()).toBeCloseTo(2, 5);
    expect(dist.beta.arraySync()).toBeCloseTo(1, 5);
  });

  test('logProb calculates correctly', () => {
    const dist = new Gamma(2, 1);
    const logProb = dist.logProb(1).arraySync();
    // For Gamma(2, 1), pdf(1) = 1^1 * exp(-1) = 0.3679
    // log(0.3679) ≈ -1.0
    expect(logProb).toBeCloseTo(-1.0, 1);
  });

  test('logProb for negative values is -Infinity', () => {
    const dist = new Gamma(2, 1);
    const logProb = dist.logProb(-1).arraySync();
    expect(logProb).toBe(-Infinity);
  });

  test('sample generates positive values', () => {
    const dist = new Gamma(2, 1);
    const samples = dist.sample([100]);
    const samplesArray = samples.arraySync();

    for (const sample of samplesArray) {
      expect(sample).toBeGreaterThan(0);
    }

    samples.dispose();
  });

  test('mean returns correct value', () => {
    const dist = new Gamma(2, 1);
    const mean = dist.mean().arraySync();
    // Mean = α / β = 2 / 1 = 2
    expect(mean).toBeCloseTo(2, 5);
  });
});

describe('Bernoulli Distribution', () => {
  test('constructor creates distribution with correct parameter', () => {
    const dist = new Bernoulli(0.7);
    expect(dist.p.arraySync()).toBeCloseTo(0.7, 5);
  });

  test('logProb for success (1)', () => {
    const dist = new Bernoulli(0.7);
    const logProb = dist.logProb(1).arraySync();
    // log(0.7) ≈ -0.357
    expect(logProb).toBeCloseTo(Math.log(0.7), 5);
  });

  test('logProb for failure (0)', () => {
    const dist = new Bernoulli(0.7);
    const logProb = dist.logProb(0).arraySync();
    // log(0.3) ≈ -1.204
    expect(logProb).toBeCloseTo(Math.log(0.3), 5);
  });

  test('sample generates binary values', () => {
    const dist = new Bernoulli(0.5);
    const samples = dist.sample([100]);
    const samplesArray = samples.arraySync();

    for (const sample of samplesArray) {
      expect([0, 1]).toContain(sample);
    }

    samples.dispose();
  });

  test('mean returns correct value', () => {
    const dist = new Bernoulli(0.7);
    const mean = dist.mean().arraySync();
    expect(mean).toBeCloseTo(0.7, 5);
  });

  test('variance returns correct value', () => {
    const dist = new Bernoulli(0.7);
    const variance = dist.variance().arraySync();
    // Var = p(1-p) = 0.7 * 0.3 = 0.21
    expect(variance).toBeCloseTo(0.21, 5);
  });
});

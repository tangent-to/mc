import { Normal, Uniform, Beta, Gamma, Bernoulli } from '../src/distributions/index.js';

describe('Normal Distribution', () => {
  test('constructor creates distribution with correct parameters', () => {
    const dist = new Normal(5, 2);
    expect(dist.mu).toBeCloseTo(5, 5);
    expect(dist.sigma).toBeCloseTo(2, 5);
  });

  test('logProb calculates correctly for standard normal', () => {
    const dist = new Normal(0, 1);
    const logProb = dist.logProb(0);
    // log(1/sqrt(2π)) ≈ -0.9189
    expect(logProb).toBeCloseTo(-0.9189385, 4);
  });

  test('logProb calculates correctly for non-zero mean', () => {
    const dist = new Normal(5, 1);
    const logProb = dist.logProb(5);
    expect(logProb).toBeCloseTo(-0.9189385, 4);
  });

  test('sample generates values', () => {
    const dist = new Normal(0, 1);
    const samples = dist.sample([100]);
    expect(samples.length).toBe(100);
  });

  test('mean returns correct value', () => {
    const dist = new Normal(5, 2);
    const mean = dist.mean();
    expect(mean).toBeCloseTo(5, 5);
  });

  test('variance returns correct value', () => {
    const dist = new Normal(0, 2);
    const variance = dist.variance();
    expect(variance).toBeCloseTo(4, 5);
  });

  test('observed data can be set', () => {
    const dist = new Normal(0, 1);
    dist.observe([1, 2, 3]);
    expect(dist.observed).toBeDefined();
    expect(dist.observed.length).toBe(3);
    const observedArray = dist.observed;
    expect(observedArray).toEqual([1, 2, 3]);
  });
});

describe('Uniform Distribution', () => {
  test('constructor creates distribution with correct bounds', () => {
    const dist = new Uniform(0, 10);
    expect(dist.lower).toBeCloseTo(0, 5);
    expect(dist.upper).toBeCloseTo(10, 5);
  });

  test('logProb inside bounds is correct', () => {
    const dist = new Uniform(0, 10);
    const logProb = dist.logProb(5);
    // log(1/10) = -2.302585
    expect(logProb).toBeCloseTo(-2.302585, 4);
  });

  test('logProb outside bounds is -Infinity', () => {
    const dist = new Uniform(0, 10);
    const logProb = dist.logProb(15);
    expect(logProb).toBe(-Infinity);
  });

  test('sample generates values within bounds', () => {
    const dist = new Uniform(0, 10);
    const samples = dist.sample([100]);
    const samplesArray = samples;

    for (const sample of samplesArray) {
      expect(sample).toBeGreaterThanOrEqual(0);
      expect(sample).toBeLessThanOrEqual(10);
    }

  });

  test('mean returns correct value', () => {
    const dist = new Uniform(0, 10);
    const mean = dist.mean();
    expect(mean).toBeCloseTo(5, 5);
  });

  test('variance returns correct value', () => {
    const dist = new Uniform(0, 12);
    const variance = dist.variance();
    // Var = (b-a)²/12 = 144/12 = 12
    expect(variance).toBeCloseTo(12, 5);
  });
});

describe('Beta Distribution', () => {
  test('logProb calculates correctly', () => {
    const dist = new Beta(2, 2);
    const logProb = dist.logProb(0.5);
    // For Beta(2,2), pdf(0.5) = 6 * 0.5^1 * 0.5^1 = 1.5
    // log(1.5) ≈ 0.405
    expect(logProb).toBeCloseTo(0.405465, 3);
  });

  test('logProb at boundaries', () => {
    const dist = new Beta(2, 2);
    const logProb0 = dist.logProb(0);
    const logProb1 = dist.logProb(1);
    expect(logProb0).toBe(-Infinity);
    expect(logProb1).toBe(-Infinity);
  });

  test('sample generates values in [0, 1]', () => {
    const dist = new Beta(2, 5);
    const samples = dist.sample([100]);
    const samplesArray = samples;

    for (const sample of samplesArray) {
      expect(sample).toBeGreaterThanOrEqual(0);
      expect(sample).toBeLessThanOrEqual(1);
    }

  });

  test('mean returns correct value', () => {
    const dist = new Beta(2, 3);
    const mean = dist.mean();
    // Mean = α / (α + β) = 2 / 5 = 0.4
    expect(mean).toBeCloseTo(0.4, 5);
  });
});

describe('Gamma Distribution', () => {
  test('constructor creates distribution with correct parameters', () => {
    const dist = new Gamma(2, 1);
    expect(dist.alpha).toBeCloseTo(2, 5);
    expect(dist.beta).toBeCloseTo(1, 5);
  });

  test('logProb calculates correctly', () => {
    const dist = new Gamma(2, 1);
    const logProb = dist.logProb(1);
    // For Gamma(2, 1), pdf(1) = 1^1 * exp(-1) = 0.3679
    // log(0.3679) ≈ -1.0
    expect(logProb).toBeCloseTo(-1.0, 1);
  });

  test('logProb for negative values is -Infinity', () => {
    const dist = new Gamma(2, 1);
    const logProb = dist.logProb(-1);
    expect(logProb).toBe(-Infinity);
  });

  test('sample generates positive values', () => {
    const dist = new Gamma(2, 1);
    const samples = dist.sample([100]);
    const samplesArray = samples;

    for (const sample of samplesArray) {
      expect(sample).toBeGreaterThan(0);
    }

  });

  test('mean returns correct value', () => {
    const dist = new Gamma(2, 1);
    const mean = dist.mean();
    // Mean = α / β = 2 / 1 = 2
    expect(mean).toBeCloseTo(2, 5);
  });
});

describe('Bernoulli Distribution', () => {
  test('constructor creates distribution with correct parameter', () => {
    const dist = new Bernoulli(0.7);
    expect(dist.p).toBeCloseTo(0.7, 5);
  });

  test('logProb for success (1)', () => {
    const dist = new Bernoulli(0.7);
    const logProb = dist.logProb(1);
    // log(0.7) ≈ -0.357
    expect(logProb).toBeCloseTo(Math.log(0.7), 5);
  });

  test('logProb for failure (0)', () => {
    const dist = new Bernoulli(0.7);
    const logProb = dist.logProb(0);
    // log(0.3) ≈ -1.204
    expect(logProb).toBeCloseTo(Math.log(0.3), 5);
  });

  test('sample generates binary values', () => {
    const dist = new Bernoulli(0.5);
    const samples = dist.sample([100]);
    const samplesArray = samples;

    for (const sample of samplesArray) {
      expect([0, 1]).toContain(sample);
    }

  });

  test('mean returns correct value', () => {
    const dist = new Bernoulli(0.7);
    const mean = dist.mean();
    expect(mean).toBeCloseTo(0.7, 5);
  });

  test('variance returns correct value', () => {
    const dist = new Bernoulli(0.7);
    const variance = dist.variance();
    // Var = p(1-p) = 0.7 * 0.3 = 0.21
    expect(variance).toBeCloseTo(0.21, 5);
  });
});

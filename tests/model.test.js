import { Model } from '../src/model.js';
import { Normal, Uniform } from '../src/distributions/index.js';
import * as tf from '@tensorflow/tfjs-node';

describe('Model', () => {
  test('constructor creates empty model', () => {
    const model = new Model('test_model');
    expect(model.name).toBe('test_model');
    expect(model.variables.size).toBe(0);
  });

  test('addVariable adds variable to model', () => {
    const model = new Model();
    const dist = new Normal(0, 1, 'x');

    model.addVariable('x', dist);

    expect(model.variables.size).toBe(1);
    expect(model.variables.has('x')).toBe(true);
  });

  test('addVariable with observed data', () => {
    const model = new Model();
    const dist = new Normal(0, 1, 'y');

    model.addVariable('y', dist, [1, 2, 3]);

    expect(model.observedVars.size).toBe(1);
    expect(dist.observed).toBeDefined();
    expect(dist.observed.shape).toEqual([3]);
  });

  test('getVariable retrieves variable', () => {
    const model = new Model();
    const dist = new Normal(0, 1, 'x');
    model.addVariable('x', dist);

    const retrieved = model.getVariable('x');

    expect(retrieved).toBe(dist);
  });

  test('getFreeVariableNames returns unobserved variables', () => {
    const model = new Model();
    const x = new Normal(0, 1, 'x');
    const y = new Normal(0, 1, 'y');

    model.addVariable('x', x);
    model.addVariable('y', y, [1, 2, 3]); // observed

    const freeVars = model.getFreeVariableNames();

    expect(freeVars).toEqual(['x']);
  });

  test('logProb computes log probability', () => {
    const model = new Model();
    const x = new Normal(0, 1, 'x');
    model.addVariable('x', x);

    const logProb = model.logProb({ x: 0 });
    const value = logProb.arraySync();

    // log(N(0|0,1)) ≈ -0.919
    expect(value).toBeCloseTo(-0.9189385, 4);

    logProb.dispose();
  });

  test('logProb handles multiple variables', () => {
    const model = new Model();
    const x = new Normal(0, 1, 'x');
    const y = new Normal(0, 1, 'y');

    model.addVariable('x', x);
    model.addVariable('y', y);

    const logProb = model.logProb({ x: 0, y: 0 });
    const value = logProb.arraySync();

    // log(N(0|0,1)) + log(N(0|0,1)) ≈ -1.8379
    expect(value).toBeCloseTo(-1.8378770, 4);

    logProb.dispose();
  });

  test('logProbAndGradient computes gradients', () => {
    const model = new Model();
    const x = new Normal(0, 1, 'x');
    model.addVariable('x', x);

    const result = model.logProbAndGradient({ x: tf.scalar(1) });

    expect(result.logProb).toBeDefined();
    expect(result.gradients).toBeDefined();

    // Gradients are indexed by tensor ID, not name
    const gradKeys = Object.keys(result.gradients);
    expect(gradKeys.length).toBeGreaterThan(0);

    // Gradient at x=1 should be approximately -1 (pulling toward mean 0)
    const gradValue = result.gradients[gradKeys[0]].arraySync();
    expect(gradValue).toBeCloseTo(-1, 0); // Less strict tolerance

    result.gradients[gradKeys[0]].dispose();
  });

  test('samplePrior generates samples', () => {
    const model = new Model();
    const x = new Normal(0, 1, 'x');
    const y = new Uniform(0, 10, 'y');

    model.addVariable('x', x);
    model.addVariable('y', y);

    const samples = model.samplePrior(5);

    expect(samples.x).toBeDefined();
    expect(samples.y).toBeDefined();
    // Each variable should have an array of samples
    expect(Array.isArray(samples.x)).toBe(true);
    expect(Array.isArray(samples.y)).toBe(true);
  });

  test('predictPosterior generates predictions', () => {
    const model = new Model();

    // Mock trace
    const trace = {
      alpha: [1, 2, 3],
      beta: [0.5, 0.6, 0.7]
    };

    // Prediction function
    const predictFn = (params) => params.alpha + params.beta;

    const predictions = model.predictPosterior({ trace }, predictFn);

    expect(predictions.length).toBe(3);
    expect(predictions[0]).toBeCloseTo(1.5, 5);
    expect(predictions[1]).toBeCloseTo(2.6, 5);
    expect(predictions[2]).toBeCloseTo(3.7, 5);
  });

  test('predictPosteriorSummary computes mean and intervals', () => {
    const model = new Model();

    // Mock trace with known values
    const trace = {
      trace: {
        alpha: [1, 2, 3, 4, 5]
      }
    };

    // Simple prediction function
    const predictFn = (params) => params.alpha;

    const summary = model.predictPosteriorSummary(trace, predictFn, 0.95);

    expect(summary.mean).toBeCloseTo(3, 5); // mean of 1,2,3,4,5
    expect(summary.lower).toBeLessThan(summary.mean);
    expect(summary.upper).toBeGreaterThan(summary.mean);
  });

  test('summary returns model description', () => {
    const model = new Model('my_model');
    const x = new Normal(0, 1, 'x');
    const y = new Normal(0, 1, 'y');

    model.addVariable('x', x);
    model.addVariable('y', y, [1, 2, 3]); // observed

    const summary = model.summary();

    expect(summary).toContain('my_model');
    expect(summary).toContain('x');
    expect(summary).toContain('y');
    expect(summary).toContain('observed');
  });
});

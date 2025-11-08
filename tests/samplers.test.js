import { MetropolisHastings, HamiltonianMC, NUTS } from '../src/samplers/index.js';
import { Model } from '../src/model.js';
import { Normal } from '../src/distributions/index.js';

describe('MetropolisHastings', () => {
  test('constructor creates sampler with correct parameters', () => {
    const sampler = new MetropolisHastings(0.5);
    expect(sampler.proposalStd).toBe(0.5);
  });

  test('sample generates trace with correct structure', () => {
    const model = new Model();
    const x = new Normal(0, 1, 'x');
    model.addVariable('x', x);

    const sampler = new MetropolisHastings(0.1);
    const trace = sampler.sample(model, { x: 0 }, 10, 5, 1);

    expect(trace.trace).toBeDefined();
    expect(trace.trace.x).toBeDefined();
    expect(trace.trace.x.length).toBe(10);
    expect(trace.acceptanceRate).toBeDefined();
    expect(trace.nSamples).toBe(10);
  });

  test('sample produces reasonable acceptance rate', () => {
    const model = new Model();
    const x = new Normal(0, 1, 'x');
    model.addVariable('x', x);

    const sampler = new MetropolisHastings(0.5);
    const trace = sampler.sample(model, { x: 0 }, 100, 50, 1);

    // Acceptance rate should be between 0 and 1
    expect(trace.acceptanceRate).toBeGreaterThan(0);
    expect(trace.acceptanceRate).toBeLessThanOrEqual(1);
  });

  test('tuneProposal adjusts step size', () => {
    const sampler = new MetropolisHastings(1.0);
    const initialStd = sampler.proposalStd;

    // High acceptance rate should increase step size
    sampler.tuneProposal(0.5);
    expect(sampler.proposalStd).toBeGreaterThan(initialStd);

    // Low acceptance rate should decrease step size
    sampler.proposalStd = 1.0;
    sampler.tuneProposal(0.1);
    expect(sampler.proposalStd).toBeLessThan(1.0);
  });

  test('sample with multiple variables', () => {
    const model = new Model();
    const alpha = new Normal(0, 1, 'alpha');
    const beta = new Normal(0, 1, 'beta');

    model.addVariable('alpha', alpha);
    model.addVariable('beta', beta);

    const sampler = new MetropolisHastings(0.5);
    const trace = sampler.sample(model, { alpha: 0, beta: 0 }, 20, 10, 1);

    expect(trace.trace.alpha.length).toBe(20);
    expect(trace.trace.beta.length).toBe(20);
  });
});

describe('HamiltonianMC', () => {
  test('constructor creates sampler with correct parameters', () => {
    const sampler = new HamiltonianMC(0.01, 10);
    expect(sampler.stepSize).toBe(0.01);
    expect(sampler.nSteps).toBe(10);
  });

  test('leapfrog performs integration', () => {
    const model = new Model();
    const x = new Normal(0, 1, 'x');
    model.addVariable('x', x);

    const sampler = new HamiltonianMC(0.01, 5);
    const result = sampler.leapfrog({ x: 0 }, { x: 1 }, model);

    expect(result.position).toBeDefined();
    expect(result.momentum).toBeDefined();
    expect(result.position.x).toBeDefined();
    expect(result.momentum.x).toBeDefined();
  });

  test('hamiltonian computes energy', () => {
    const model = new Model();
    const x = new Normal(0, 1, 'x');
    model.addVariable('x', x);

    const sampler = new HamiltonianMC(0.01, 10);
    const H = sampler.hamiltonian({ x: 0 }, { x: 0 }, model);

    expect(H).toBeDefined();
    expect(typeof H).toBe('number');
  });

  test('sample generates trace', () => {
    const model = new Model();
    const x = new Normal(0, 1, 'x');
    model.addVariable('x', x);

    const sampler = new HamiltonianMC(0.01, 5);
    const trace = sampler.sample(model, { x: 0 }, 10, 5, 1);

    expect(trace.trace.x.length).toBe(10);
    expect(trace.acceptanceRate).toBeDefined();
  });

  test('sample with multiple variables', () => {
    const model = new Model();
    const alpha = new Normal(0, 1, 'alpha');
    const beta = new Normal(0, 1, 'beta');

    model.addVariable('alpha', alpha);
    model.addVariable('beta', beta);

    const sampler = new HamiltonianMC(0.01, 5);
    const trace = sampler.sample(model, { alpha: 0, beta: 0 }, 10, 5, 1);

    expect(trace.trace.alpha.length).toBe(10);
    expect(trace.trace.beta.length).toBe(10);
  });
});

describe('NUTS', () => {
  test('constructor creates sampler with correct parameters', () => {
    const sampler = new NUTS(0.01, 10, 0.8);
    expect(sampler.stepSize).toBe(0.01);
    expect(sampler.maxTreeDepth).toBe(10);
    expect(sampler.targetAcceptance).toBe(0.8);
  });

  test('leapfrog performs integration', () => {
    const model = new Model();
    const x = new Normal(0, 1, 'x');
    model.addVariable('x', x);

    const sampler = new NUTS(0.01, 5);
    const result = sampler.leapfrog({ x: 0 }, { x: 1 }, 0.01, model);

    expect(result.position).toBeDefined();
    expect(result.momentum).toBeDefined();
    expect(result.position.x).toBeDefined();
    expect(result.momentum.x).toBeDefined();
  });

  test('isUTurn detects U-turns', () => {
    const sampler = new NUTS(0.01, 10);

    // No U-turn: momentum and position aligned
    const noUTurn = sampler.isUTurn(
      { x: 0 },
      { x: 1 },
      { x: 1 },
      { x: 1 }
    );
    expect(noUTurn).toBe(false);

    // U-turn: momentum points backward
    const hasUTurn = sampler.isUTurn(
      { x: 0 },
      { x: 1 },
      { x: 1 },
      { x: -1 }
    );
    expect(hasUTurn).toBe(true);
  });

  test('buildTree generates valid tree', () => {
    const model = new Model();
    const x = new Normal(0, 1, 'x');
    model.addVariable('x', x);

    const sampler = new NUTS(0.01, 5);
    const H0 = sampler.hamiltonian({ x: 0 }, { x: 0 }, model);
    const slice = Math.random() * Math.exp(-H0);

    const tree = sampler.buildTree(
      { x: 0 },
      { x: 0 },
      slice,
      1,  // direction
      0,  // depth (base case)
      0.01,
      model,
      H0
    );

    expect(tree.positionMinus).toBeDefined();
    expect(tree.positionPlus).toBeDefined();
    expect(tree.momentumMinus).toBeDefined();
    expect(tree.momentumPlus).toBeDefined();
    expect(tree.positionPrime).toBeDefined();
    expect(tree.nValid).toBeDefined();
    expect(tree.stop).toBeDefined();
  });

  test('sample generates trace with warmup', () => {
    const model = new Model();
    const x = new Normal(0, 1, 'x');
    model.addVariable('x', x);

    const sampler = new NUTS(0.1, 5);
    const trace = sampler.sample(model, { x: 0 }, 10, 5, 1);

    expect(trace.trace.x.length).toBe(10);
    expect(trace.acceptanceRate).toBeDefined();
    expect(trace.stepSize).toBeDefined();

    // Step size should be adapted during warmup
    expect(trace.stepSize).toBeGreaterThan(0);
  });

  test('sample adapts step size during warmup', () => {
    const model = new Model();
    const x = new Normal(0, 1, 'x');
    model.addVariable('x', x);

    const initialStepSize = 0.5;
    const sampler = new NUTS(initialStepSize, 5);
    const trace = sampler.sample(model, { x: 0 }, 10, 10, 1);

    // Step size should change from initial value
    expect(sampler.stepSize).not.toBe(initialStepSize);
  });

  test('sample with multiple variables', () => {
    const model = new Model();
    const alpha = new Normal(0, 1, 'alpha');
    const beta = new Normal(0, 1, 'beta');

    model.addVariable('alpha', alpha);
    model.addVariable('beta', beta);

    const sampler = new NUTS(0.1, 5);
    const trace = sampler.sample(model, { alpha: 0, beta: 0 }, 10, 5, 1);

    expect(trace.trace.alpha.length).toBe(10);
    expect(trace.trace.beta.length).toBe(10);
  });
});

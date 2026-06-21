/**
 * Verifies the public API follows the @tangent.to/ds conventions:
 *   - namespaced exports + a default export bundling every namespace
 *   - options-object constructors (alongside positional ones)
 *   - getParams() introspection
 */
import mc, {
  Model,
  distributions,
  samplers,
  diagnostics,
  plot,
  Normal,
  Uniform,
  Beta,
  Gamma,
  Bernoulli,
  MetropolisHastings,
  HamiltonianMC,
  NUTS
} from '../src/index.js';

describe('Namespaced exports', () => {
  test('named namespaces are objects with the expected members', () => {
    expect(distributions.Normal).toBe(Normal);
    expect(samplers.MetropolisHastings).toBe(MetropolisHastings);
    expect(typeof diagnostics.summarize).toBe('function');
    expect(typeof plot.tracePlot).toBe('function');
  });

  test('default export bundles every namespace', () => {
    expect(mc.Model).toBe(Model);
    expect(mc.distributions.Normal).toBe(Normal);
    expect(mc.samplers.NUTS).toBe(NUTS);
    expect(mc.diagnostics).toBe(diagnostics);
    expect(mc.plot).toBe(plot);
  });
});

describe('Options-object constructors for distributions', () => {
  test('Normal accepts { mean, sd } and matches positional form', () => {
    const a = new Normal({ mean: 5, sd: 2 });
    const b = new Normal(5, 2);
    expect(a.getParams()).toEqual(b.getParams());
    expect(a.getParams()).toEqual({ mu: 5, sigma: 2 });
  });

  test('Uniform accepts { min, max }', () => {
    const u = new Uniform({ min: 0, max: 10 });
    expect(u.getParams()).toEqual({ lower: 0, upper: 10 });
  });

  test('Beta accepts { alpha, beta }', () => {
    const d = new Beta({ alpha: 2, beta: 5 });
    expect(d.getParams()).toEqual({ alpha: 2, beta: 5 });
  });

  test('Gamma accepts { shape, rate }', () => {
    const d = new Gamma({ shape: 2, rate: 3 });
    expect(d.getParams()).toEqual({ alpha: 2, beta: 3 });
  });

  test('Bernoulli accepts { p }', () => {
    const d = new Bernoulli({ p: 0.7 });
    expect(d.getParams().p).toBeCloseTo(0.7, 6);
  });

  test('name flows through the options object', () => {
    const d = new Normal({ mean: 0, sd: 1, name: 'alpha' });
    expect(d.name).toBe('alpha');
  });
});

describe('Options-object constructors for samplers', () => {
  test('MetropolisHastings accepts { proposalStd }', () => {
    const s = new MetropolisHastings({ proposalStd: 0.5 });
    expect(s.proposalStd).toBe(0.5);
    expect(s.getParams()).toEqual({ proposalStd: 0.5 });
  });

  test('HamiltonianMC accepts { stepSize, nSteps }', () => {
    const s = new HamiltonianMC({ stepSize: 0.02, nSteps: 7 });
    expect(s.getParams()).toEqual({ stepSize: 0.02, nSteps: 7 });
  });

  test('NUTS accepts { stepSize, maxTreeDepth, targetAcceptance }', () => {
    const s = new NUTS({ stepSize: 0.05, maxTreeDepth: 8, targetAcceptance: 0.9 });
    expect(s.getParams()).toEqual({
      stepSize: 0.05,
      maxTreeDepth: 8,
      targetAcceptance: 0.9
    });
  });

  test('sample() accepts an options object for run controls', () => {
    const model = new Model({ name: 'opts' });
    model.addVariable('x', new Normal({ mean: 0, sd: 1, name: 'x' }));

    const sampler = new MetropolisHastings({ proposalStd: 0.5 });
    const trace = sampler.sample(model, { x: 0 }, { nSamples: 8, burnIn: 4, thin: 1 });
    expect(trace.trace.x.length).toBe(8);
  });
});

describe('Model options', () => {
  test('Model accepts { name }', () => {
    const m = new Model({ name: 'my_model' });
    expect(m.name).toBe('my_model');
  });
});

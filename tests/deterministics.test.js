import { MetropolisHastings, NUTS } from '../src/samplers/index.js';
import { Model } from '../src/model.js';
import { Normal } from '../src/distributions/index.js';

/**
 * Model.deterministic(name, fn) registers a post-hoc transform of the posterior
 * draws; the samplers call Model.computeDeterministics() before returning, so the
 * transform appears as an extra trace column aligned with the free variables.
 */
describe('Model.deterministic()', () => {
  test('deterministic columns are recorded in the trace (Metropolis)', () => {
    const model = new Model('det');
    model.addVariable('x', new Normal(0, 1));
    model.deterministic('x2', (p) => p.x * p.x);

    const trace = new MetropolisHastings({ proposalStd: 0.6 })
      .sample(model, { x: 0 }, { nSamples: 50, burnIn: 20 }).trace;

    expect(trace.x2).toBeDefined();
    expect(trace.x2.length).toBe(trace.x.length);
    for (let i = 0; i < trace.x.length; i++) {
      expect(trace.x2[i]).toBeCloseTo(trace.x[i] * trace.x[i], 8);
    }
  });

  test('a model with no deterministics is unaffected (NUTS)', () => {
    const model = new Model('nodet');
    model.addVariable('x', new Normal(0, 1));

    const trace = new NUTS({ stepSize: 0.3 })
      .sample(model, { x: 0 }, { nSamples: 30, nWarmup: 30 }).trace;

    expect(Object.keys(trace)).toEqual(['x']);
  });
});

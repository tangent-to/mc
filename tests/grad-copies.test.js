import { describe, expect, it } from 'vitest';
import { Model } from '../src/model.js';
import { Normal } from '../src/distributions/index.js';

// A second copy of @tangent.to/grad cannot be loaded inside one vitest module graph, so the
// foreign copy is simulated: proba's logDensity returns what a foreign copy returns, an object
// that is not mc's Var. Observing through it must throw, not sample the prior.
describe('observe with a density from another copy of grad', () => {
  it('throws instead of dropping the likelihood', () => {
    const model = new Model('copies');
    model.addVariable('mu', new Normal(0, 10));
    model.observe('y', (v) => {
      const d = new Normal(v.mu, 1);
      const real = d._dist;
      d._dist = Object.create(real, {
        logDensity: { value: () => ({ value: { data: new Float64Array([0]) }, shape: [] }) },
      });
      return d;
    }, [1, 2, 3]);
    expect(() => model.logProb({ mu: 0 })).toThrow(/different copy of\s+@tangent.to\/grad/);
  });

  it('still observes normally with a single copy', () => {
    const model = new Model('one');
    model.addVariable('mu', new Normal(0, 10));
    model.observe('y', (v) => new Normal(v.mu, 1), [1, 2, 3]);
    expect(Number.isFinite(model.logProb({ mu: 2 }))).toBe(true);
    expect(model.logProb({ mu: 2 })).toBeGreaterThan(model.logProb({ mu: 10 }));
  });
});

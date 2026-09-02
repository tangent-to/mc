/**
 * `sample(model, init, { chains })`: the model written as it is, several
 * chains, and no say in where they run.
 *
 * The contract has three parts, each tested here. The model survives the
 * round trip through data (same value and gradient as the live one). Workers
 * and the calling thread draw the same samples from the same seeds, so the
 * choice is invisible in the result. And when the model cannot travel, the
 * chains still run, in series, with the reason named once.
 */

import { describe, expect, it, vi } from 'vitest';
import { add, mul } from '@tangent.to/grad';
import { Model } from '../src/model.js';
import { NUTS } from '../src/samplers/nuts.js';
import { MetropolisHastings } from '../src/samplers/metropolis.js';
import { Distribution, HalfNormal, Normal } from '../src/distributions/index.js';
import { gelmanRubin } from '../src/utils/trace.js';

const X = Array.from({ length: 40 }, (_, i) => i / 10 - 2);
const Y = X.map((x) => 1.5 + 0.8 * x + 0.2 * Math.sin(7 * x));
const INIT = { a: 0, b: 0, sigma: 1 };

const regression = () => {
  const m = new Model('lin');
  m.addVariable('a', new Normal(0, 5));
  m.addVariable('b', new Normal(0, 5));
  m.addVariable('sigma', new HalfNormal(2));
  m.observe('y', (v) => new Normal(add(v.a, mul(v.b, X)), v.sigma), Y);
  return m;
};
const mean = (arr) => arr.reduce((s, v) => s + v, 0) / arr.length;

describe('a model as data', () => {
  it('round-trips to the same value and gradient', () => {
    const live = regression();
    const json = structuredClone(live.toJSON(INIT));
    const back = Model.fromJSON(json);
    for (const p of [INIT, { a: 1.4, b: 0.9, sigma: 0.3 }, { a: -2, b: 3, sigma: 1.7 }]) {
      expect(back.logProb(p)).toBe(live.logProb(p));
      const ga = live.gradientsOnly(p);
      const gb = back.gradientsOnly(p);
      for (const k of Object.keys(ga)) expect(gb[k], k).toBe(ga[k]);
    }
  });

  it('carries the variables as kinds and parameters, the terms as plans', () => {
    const json = regression().toJSON(INIT);
    expect(json.variables.map((v) => [v.name, v.kind])).toEqual([['a', 'Normal'], ['b', 'Normal'], ['sigma', 'HalfNormal']]);
    expect(json.variables[2].params).toEqual({ sigma: 2 });
    expect(json.terms).toHaveLength(1);
    expect(json.terms[0].plan.version).toBe(1);
  });

  it('needs a point to trace at', () => {
    expect(() => regression().toJSON()).toThrow(/pass a point to trace/);
  });

  it('names the plain potential that keeps it from travelling', () => {
    const m = regression();
    m.potential('extra', (p) => -0.5 * p.a * p.a);
    expect(m.serializable()).toEqual({
      ok: false,
      reason: 'potential "extra" is a plain function and cannot be sent to a worker; write it with autoPotential or observe',
    });
    expect(() => m.toJSON(INIT)).toThrow(/potential "extra"/);
  });

  it('names a distribution that is not one of its own', () => {
    class Custom extends Distribution {
      _params() { return {}; }
      getParams() { return {}; }
    }
    const m = new Model();
    m.addVariable('w', new Custom('Custom'));
    expect(m.serializable().ok).toBe(false);
    expect(m.serializable().reason).toMatch(/variable "w".*Custom.*not one of mc's own/);
  });
});

describe('sample(model, init, { chains })', () => {
  const RUN = { chains: 3, nSamples: 150, nWarmup: 150, seed: 11 };

  it('runs the chains in workers and recovers the posterior', async () => {
    const fit = await new NUTS({ stepSize: 0.05 }).sample(regression(), INIT, RUN);
    expect(fit.parallel).toBe(true);
    expect(fit.parallelReason).toBeNull();
    expect(fit.byChain.a).toHaveLength(3);
    expect(fit.byChain.a[0]).toHaveLength(150);
    expect(fit.trace.a).toHaveLength(450);
    expect(mean(fit.trace.a)).toBeCloseTo(1.5, 0);
    expect(mean(fit.trace.b)).toBeCloseTo(0.8, 0);
    expect(gelmanRubin(fit.byChain.b)).toBeLessThan(1.2);
  }, 60000);

  it('draws the same samples on the calling thread as in workers', async () => {
    // The choice of thread must be invisible in the result. Same seeds, same
    // model, same sampler: byte-identical draws, not merely the same posterior.
    const workers = await new NUTS({ stepSize: 0.05 }).sample(regression(), INIT, RUN);
    const series = await new NUTS({ stepSize: 0.05 }).sample(regression(), INIT, { ...RUN, parallel: false });
    expect(workers.parallel).toBe(true);
    expect(series.parallel).toBe(false);
    expect(series.seeds).toEqual(workers.seeds);
    expect(series.trace.a).toEqual(workers.trace.a);
    expect(series.trace.b).toEqual(workers.trace.b);
    expect(series.trace.sigma).toEqual(workers.trace.sigma);
  }, 120000);

  it('falls back to series when a term cannot travel, and says why once', async () => {
    const m = regression();
    m.potential('extra', (p) => -0.5 * p.a * p.a);
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const fit = await new NUTS({ stepSize: 0.05 }).sample(m, INIT, RUN);
    expect(fit.parallel).toBe(false);
    expect(fit.parallelReason).toMatch(/potential "extra" is a plain function/);
    const fallbackWarnings = warn.mock.calls.filter(([m]) => /running 3 chains in series/.test(m));
    expect(fallbackWarnings).toHaveLength(1);
    expect(fallbackWarnings[0][0]).toMatch(/running 3 chains in series; potential "extra"/);
    expect(fit.byChain.a).toHaveLength(3);
    warn.mockRestore();
  }, 60000);

  it('stays quiet when asked', async () => {
    const m = regression();
    m.potential('extra', (p) => -0.5 * p.a * p.a);
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await new NUTS({ stepSize: 0.05 }).sample(m, INIT, { ...RUN, quiet: true });
    expect(warn).not.toHaveBeenCalled(); // neither the fallback notice nor a divergence report
    warn.mockRestore();
  }, 60000);

  it('applies deterministics on the calling thread after the chains return', async () => {
    const m = regression();
    m.deterministic('slope_x2', (p) => 2 * p.b);
    const fit = await new NUTS({ stepSize: 0.05 }).sample(m, INIT, RUN);
    expect(fit.parallel).toBe(true);
    expect(fit.trace.slope_x2).toHaveLength(450);
    fit.trace.slope_x2.forEach((v, i) => expect(v).toBe(2 * fit.trace.b[i]));
    expect(fit.byChain.slope_x2).toHaveLength(3);
  }, 60000);

  it('takes one init per chain for over-dispersed starts', async () => {
    const inits = [{ a: -3, b: -3, sigma: 2 }, { a: 0, b: 0, sigma: 1 }, { a: 3, b: 3, sigma: 0.5 }];
    const fit = await new NUTS({ stepSize: 0.05 }).sample(regression(), INIT, { ...RUN, inits });
    expect(fit.chains).toHaveLength(3);
    expect(gelmanRubin(fit.byChain.a)).toBeLessThan(1.3);
  }, 60000);

  it('rejects a mismatched number of inits', async () => {
    await expect(new NUTS().sample(regression(), INIT, { chains: 3, inits: [INIT, INIT] }))
      .rejects.toThrow(/got 2 inits for 3 chains/);
  });

  it('one chain, or the positional form, stays synchronous and unchanged', () => {
    const fit = new NUTS({ stepSize: 0.05 }).sample(regression(), INIT, { nSamples: 20, nWarmup: 20 });
    expect(fit.trace.a).toHaveLength(20);
    expect(typeof fit.then).toBe('undefined');
  });

  it('works for the vector HMC as well', async () => {
    const { HMC } = await import('../src/samplers/hmc-vector.js');
    const fit = await new HMC({ stepSize: 0.05, nSteps: 10 }).sample(regression(), INIT, { chains: 2, nSamples: 200, nWarmup: 200, seed: 3 });
    expect(fit.parallel).toBe(true);
    expect(fit.byChain.a).toHaveLength(2);
    expect(mean(fit.trace.a)).toBeCloseTo(1.5, 0);
  }, 60000);

  it('works for Metropolis as well', async () => {
    const fit = await new MetropolisHastings(0.3).sample(regression(), INIT, { chains: 2, nSamples: 200, burnIn: 200, seed: 3 });
    expect(fit.parallel).toBe(true);
    expect(fit.byChain.a).toHaveLength(2);
  }, 60000);
});

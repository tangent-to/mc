import { describe, expect, it } from 'vitest';
import { sampleChains } from '../src/parallel.js';
import { gelmanRubin, effectiveSampleSize } from '../src/utils/trace.js';

// Self-contained factory: linear regression with an analytic potential
// gradient, the shape the guava notebook uses. Uses ONLY (data, mc).
const linearFactory = (data, mc) => {
  const model = new mc.Model('lin');
  model.addVariable('a', new mc.distributions.Normal(0, 5));
  model.addVariable('b', new mc.distributions.Normal(0, 5));
  model.addVariable('logSig', new mc.distributions.Normal(0, 1));
  const like = (p) => {
    const sig = Math.exp(p.logSig);
    const mu = data.xs.map((x) => p.a + p.b * x);
    return new mc.distributions.Normal(mu, sig).logProb(data.ys);
  };
  const grad = (p) => {
    const sig = Math.exp(p.logSig);
    const inv2 = 1 / (sig * sig);
    let ga = 0;
    let gb = 0;
    let gs = 0;
    for (let i = 0; i < data.xs.length; i++) {
      const z = data.ys[i] - (p.a + p.b * data.xs[i]);
      const r = z * inv2;
      ga += r;
      gb += r * data.xs[i];
      gs += -1 + z * z * inv2;
    }
    return { a: ga, b: gb, logSig: gs };
  };
  model.potential('lik', like, grad);
  return model;
};

// Deterministic synthetic data: y = 1.5 + 0.8 x + small noise.
function makeData() {
  let s = 123;
  const rnd = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
  const xs = Array.from({ length: 60 }, () => rnd() * 4);
  const ys = xs.map((x) => 1.5 + 0.8 * x + 0.2 * (rnd() - 0.5));
  return { xs, ys };
}

const RUN = { nSamples: 150, nWarmup: 150, seed: 7, samplerOptions: { stepSize: 0.05 } };
const INITS = [
  { a: 0, b: 0, logSig: 0 },
  { a: 1, b: 1, logSig: -0.3 },
];

describe('sampleChains', () => {
  it('runs chains in workers and recovers the posterior', async () => {
    const data = makeData();
    const fit = await sampleChains(linearFactory, { ...RUN, data, chains: 2, inits: INITS });

    expect(fit.parallel).toBe(true);
    expect(fit.chains).toHaveLength(2);
    expect(fit.byChain.a).toHaveLength(2);
    expect(fit.byChain.a[0]).toHaveLength(150);
    expect(fit.trace.a).toHaveLength(300);
    expect(fit.seeds[0]).not.toBe(fit.seeds[1]);

    const mean = (arr) => arr.reduce((s, v) => s + v, 0) / arr.length;
    expect(mean(fit.trace.a)).toBeCloseTo(1.5, 0);
    expect(mean(fit.trace.b)).toBeCloseTo(0.8, 0);

    // by-chain arrays feed the diagnostics directly.
    expect(gelmanRubin(fit.byChain.b)).toBeLessThan(1.2);
    expect(effectiveSampleSize(fit.trace.b)).toBeGreaterThan(10);
  }, 60000);

  it('sequential fallback (parallel: false) is identical to the worker run', async () => {
    const data = makeData();
    const par = await sampleChains(linearFactory, { ...RUN, data, chains: 2, inits: INITS });
    const seq = await sampleChains(linearFactory, {
      ...RUN,
      data,
      chains: 2,
      inits: INITS,
      parallel: false,
    });

    expect(par.parallel).toBe(true);
    expect(seq.parallel).toBe(false);
    expect(seq.seeds).toEqual(par.seeds);
    expect(seq.trace.a).toEqual(par.trace.a);
    expect(seq.trace.b).toEqual(par.trace.b);
    expect(seq.trace.logSig).toEqual(par.trace.logSig);
  }, 120000);

  it('rejects a factory that closes over outer variables, with guidance', async () => {
    const outer = { xs: [1, 2], ys: [1, 2] };
    const leakyFactory = (data, mc) => {
      const model = new mc.Model('leaky');
      model.addVariable('a', new mc.distributions.Normal(0, 5));
      // BUG on purpose: reads `outer` from the enclosing test scope.
      model.potential('lik', (p) => new mc.distributions.Normal(p.a, 1).logProb(outer.ys));
      return model;
    };
    await expect(
      sampleChains(leakyFactory, { ...RUN, chains: 1, inits: { a: 0 } }),
    ).rejects.toThrow(/self-contained/);
  }, 60000);

  it('validates inits', async () => {
    await expect(sampleChains(linearFactory, { chains: 2 })).rejects.toThrow(/inits is required/);
    await expect(
      sampleChains(linearFactory, { chains: 3, inits: [{ a: 0 }] }),
    ).rejects.toThrow(/1 inits for 3 chains/);
  });
});

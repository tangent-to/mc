/**
 * Parallel MCMC chains across worker threads.
 *
 * MCMC chains are independent by construction, but `sampler.sample()` runs
 * them one after another on one thread. {@link sampleChains} runs each chain
 * in its own worker (browser/Deno `Worker` or Node `worker_threads`), so four
 * chains cost roughly one chain of wall-clock time.
 *
 * The one design constraint is serialization: a worker cannot receive live
 * closures, so the model is described by a SELF-CONTAINED factory function
 * plus a structured-clonable `data` object. The factory's source is sent to
 * each worker (via `Function.prototype.toString`) and evaluated there:
 *
 * ```js
 * const fit = await sampleChains(
 *   (data, mc) => {
 *     const model = new mc.Model('lin');
 *     model.addVariable('a', new mc.distributions.Normal(0, 5));
 *     model.addVariable('logSig', new mc.distributions.Normal(0, 1));
 *     model.potential('lik', (p) => {
 *       const sig = Math.exp(p.logSig);
 *       const mu = data.xs.map(() => p.a);
 *       return new mc.distributions.Normal(mu, sig).logProb(data.ys);
 *     });
 *     return model;
 *   },
 *   {
 *     data: { xs, ys },              // structured-clonable only
 *     chains: 4,
 *     inits: [{ a: 0, logSig: 0 }, …], // one per chain (over-dispersed)
 *     nSamples: 400, nWarmup: 400,
 *     seed: 20240115,
 *   },
 * );
 * fit.byChain.a   // [[chain-0 draws], [chain-1 draws], …] → gelmanRubin
 * fit.trace.a     // pooled draws
 * ```
 *
 * The factory MUST only use its two arguments (`data`, `mc`) and JavaScript
 * built-ins — referencing any outer variable throws a ReferenceError in the
 * worker. Everything the model needs goes in `data`.
 *
 * Seeding: each chain gets its own seed derived from `options.seed`, so a
 * run is reproducible but its draws differ from a single-stream sequential
 * run (chains are independently seeded, which is what R-hat assumes anyway).
 * When workers are unavailable (or `parallel: false`), chains run in-process
 * with the SAME per-chain seeds and factory-source evaluation, producing
 * identical results, just serially.
 *
 * @module
 */

import { Model } from './model.js';
import { setRandomSeed, getRng } from './rng.js';
import {
  Distribution,
  Normal,
  Uniform,
  Bernoulli,
  Beta,
  Gamma,
  Lognormal,
  HalfNormal,
} from './distributions/index.js';
import { MetropolisHastings, HamiltonianMC, NUTS, HMC } from './samplers/index.js';

const MODULE_URL = import.meta.url;

/**
 * The `mc` argument handed to the model factory (in the parent AND in each
 * worker): the Model class, every distribution and sampler, and the RNG
 * controls. A factory needs nothing else from the package.
 * @type {Object}
 */
export const chainToolkit = {
  Model,
  setRandomSeed,
  getRng,
  distributions: {
    Distribution,
    Normal,
    Uniform,
    Bernoulli,
    Beta,
    Gamma,
    Lognormal,
    HalfNormal,
  },
  samplers: { MetropolisHastings, HamiltonianMC, NUTS, HMC },
};

const SAMPLERS = { nuts: NUTS, hmc: HMC, metropolis: MetropolisHastings };

/** Derive a per-chain seed from the base seed (golden-ratio increment). */
function chainSeed(seed, chain) {
  return ((seed >>> 0) + Math.imul(0x9e3779b9, chain + 1)) >>> 0;
}

/**
 * Run ONE chain from a serialized spec. Shared by the worker entry and the
 * in-process fallback so both paths are byte-for-byte the same computation.
 * Internal — exported so the worker bootstrap can `import()` this module and
 * call it; not part of the public API.
 *
 * @param {Object} spec - {factorySrc, data, samplerName, samplerOptions,
 *   init, runOptions, seed}
 * @returns {{trace: Object, acceptanceRate: number|undefined, stepSize: number|undefined}}
 */
export function __runChain(spec) {
  setRandomSeed(spec.seed);
  let factory;
  try {
    factory = (0, eval)(`(${spec.factorySrc})`);
  } catch (err) {
    throw new Error(`sampleChains: could not evaluate the model factory: ${err.message}`);
  }
  const model = factory(spec.data, chainToolkit);
  const Sampler = SAMPLERS[spec.samplerName];
  if (!Sampler) {
    throw new Error(
      `sampleChains: unknown sampler "${spec.samplerName}" (use ${Object.keys(SAMPLERS).join(' | ')})`,
    );
  }
  const sampler = new Sampler(spec.samplerOptions ?? {});
  const fit = sampler.sample(model, spec.init, spec.runOptions);
  return {
    trace: fit.trace,
    acceptanceRate: fit.acceptanceRate,
    stepSize: fit.stepSize ?? sampler.stepSize,
  };
}

// ---------------------------------------------------------------------------
// Worker plumbing per runtime. Each chain gets a short-lived worker that
// imports THIS module (by its own URL) and calls __runChain.
// ---------------------------------------------------------------------------

const isNode = () =>
  typeof process !== 'undefined' && !!process.versions?.node && typeof globalThis.Deno === 'undefined';

// Browser / Deno: a blob module worker. It carries no code of its own — it
// dynamically imports this module and delegates, so bundled and unbundled
// distributions both work (MODULE_URL points at whichever file this is).
const BLOB_WORKER_SRC = `
self.onmessage = async (e) => {
  try {
    const mod = await import(e.data.moduleUrl);
    const result = mod.__runChain(e.data.spec);
    self.postMessage({ ok: true, result });
  } catch (err) {
    self.postMessage({ ok: false, error: String((err && err.stack) || err) });
  }
  self.close();
};
`;

// Node: an eval'd CommonJS worker (worker_threads has no blob URLs).
const NODE_WORKER_SRC = `
const { parentPort } = require('worker_threads');
parentPort.on('message', async (msg) => {
  try {
    const mod = await import(msg.moduleUrl);
    const result = mod.__runChain(msg.spec);
    parentPort.postMessage({ ok: true, result });
  } catch (err) {
    parentPort.postMessage({ ok: false, error: String((err && err.stack) || err) });
  }
});
`;

/** Run one chain spec in a fresh worker; resolves with __runChain's result. */
async function runChainInWorker(spec) {
  if (isNode()) {
    // Computed specifier so browser bundlers (esbuild/vite/rollup) don't try
    // to resolve the Node built-in; this branch only runs under Node.
    const nodeWorkerThreads = 'node:worker_threads';
    const { Worker } = await import(/* @vite-ignore */ nodeWorkerThreads);
    return new Promise((resolve, reject) => {
      const worker = new Worker(NODE_WORKER_SRC, { eval: true });
      worker.once('message', (msg) => {
        worker.terminate();
        if (msg.ok) resolve(msg.result);
        else reject(new Error(msg.error));
      });
      worker.once('error', (err) => {
        worker.terminate();
        reject(err);
      });
      worker.postMessage({ moduleUrl: MODULE_URL, spec });
    });
  }

  if (typeof Worker === 'undefined') {
    throw new Error('no Worker support in this runtime');
  }
  const blobUrl = URL.createObjectURL(new Blob([BLOB_WORKER_SRC], { type: 'text/javascript' }));
  try {
    return await new Promise((resolve, reject) => {
      let worker;
      try {
        worker = new Worker(blobUrl, { type: 'module' });
      } catch (err) {
        reject(err);
        return;
      }
      worker.onmessage = (e) => {
        worker.terminate();
        if (e.data.ok) resolve(e.data.result);
        else reject(new Error(e.data.error));
      };
      worker.onerror = (e) => {
        worker.terminate();
        reject(new Error(e.message || 'chain worker failed to start'));
      };
      worker.postMessage({ moduleUrl: MODULE_URL, spec });
    });
  } finally {
    URL.revokeObjectURL(blobUrl);
  }
}

/**
 * Sample several MCMC chains in parallel, one worker per chain.
 *
 * Accepts a SELF-CONTAINED model factory `(data, mc) => Model` — see the
 * module doc for the contract — and returns the per-chain fits plus pooled
 * and by-chain traces ready for `gelmanRubin` / `effectiveSampleSize`.
 *
 * Falls back to running the chains sequentially in-process (identical
 * numbers, same per-chain seeds) when the runtime has no workers or when
 * `parallel: false` is passed.
 *
 * @param {(data: Object, mc: Object) => Model} modelFactory - Builds the model
 *   from `data` and the mc toolkit; must not reference outer variables.
 * @param {Object} options
 * @param {Object} [options.data={}] - Structured-clonable data for the factory
 * @param {number} [options.chains=4] - Number of chains
 * @param {Array<Object>|Object} [options.inits] - Per-chain initial values
 *   (array, one per chain — over-dispersed starts recommended), or a single
 *   init object used for every chain. Required.
 * @param {string} [options.sampler='nuts'] - 'nuts' | 'hmc' | 'metropolis'
 * @param {Object} [options.samplerOptions] - Constructor options for the sampler
 *   (e.g. `{stepSize, maxTreeDepth, targetAcceptance}`)
 * @param {number} [options.nSamples=1000] - Draws per chain
 * @param {number} [options.nWarmup=500] - Warmup iterations per chain
 * @param {number} [options.thin=1] - Thinning interval
 * @param {number} [options.seed=42] - Base seed; chain c uses a seed derived
 *   from it, so runs are reproducible and chains are independent
 * @param {boolean} [options.parallel=true] - Force the sequential in-process
 *   path with `false` (same results, no workers)
 * @returns {Promise<{chains: Array<Object>, byChain: Object, trace: Object,
 *   acceptanceRates: Array<number>, seeds: Array<number>, parallel: boolean}>}
 */
export async function sampleChains(modelFactory, options = {}) {
  const {
    data = {},
    chains = 4,
    inits,
    sampler = 'nuts',
    samplerOptions = {},
    nSamples = 1000,
    nWarmup = 500,
    thin = 1,
    seed = 42,
    parallel = true,
  } = options;

  if (typeof modelFactory !== 'function') {
    throw new Error('sampleChains: modelFactory must be a function (data, mc) => Model');
  }
  if (!inits) {
    throw new Error(
      'sampleChains: options.inits is required — an array of per-chain initial values (over-dispersed starts recommended), or one init object for all chains',
    );
  }
  const initList = Array.isArray(inits) ? inits : Array.from({ length: chains }, () => inits);
  if (initList.length !== chains) {
    throw new Error(`sampleChains: got ${initList.length} inits for ${chains} chains`);
  }

  const factorySrc = modelFactory.toString();
  const specs = initList.map((init, c) => ({
    factorySrc,
    data,
    samplerName: sampler,
    samplerOptions,
    init,
    runOptions: { nSamples, nWarmup, thin },
    seed: chainSeed(seed, c),
  }));

  let results;
  let ranParallel = false;
  if (parallel) {
    try {
      results = await Promise.all(specs.map((spec) => runChainInWorker(spec)));
      ranParallel = true;
    } catch (err) {
      const msg = String((err && err.message) || err);
      // A missing-variable error means the factory is not self-contained —
      // rerunning sequentially would NOT fix it, so fail loudly with guidance.
      if (/is not defined/.test(msg)) {
        throw new Error(
          `sampleChains: the model factory references a variable that does not exist inside the worker (${msg}). ` +
            'The factory must be self-contained: use only its (data, mc) arguments and pass everything else through options.data.',
        );
      }
      results = null; // worker machinery unavailable → sequential fallback
    }
  }
  if (!results) {
    results = specs.map((spec) => __runChain(spec));
  }

  // Assemble: per-chain fits, by-chain arrays per parameter, pooled trace.
  const paramNames = Object.keys(results[0].trace);
  const byChain = {};
  const trace = {};
  for (const name of paramNames) {
    byChain[name] = results.map((r) => r.trace[name]);
    trace[name] = results.flatMap((r) => r.trace[name]);
  }

  return {
    chains: results.map((r, c) => ({ ...r, seed: specs[c].seed })),
    byChain,
    trace,
    acceptanceRates: results.map((r) => r.acceptanceRate),
    seeds: specs.map((s) => s.seed),
    parallel: ranParallel,
  };
}

import * as tf from '@tensorflow/tfjs-node';
import { effectiveSampleSize, gelmanRubin } from '../utils/trace.js';

/**
 * Vector-aware Hamiltonian Monte Carlo.
 *
 * Unlike the scalar `HamiltonianMC`/`NUTS` in this package, this sampler flattens
 * all free variables — scalars and 1-D vectors alike — into a single real vector
 * and runs leapfrog dynamics on it. That makes it suitable for hierarchical
 * models whose parameters are vectors (per-group effects, per-site plateaus, …)
 * and for likelihoods defined through {@link Model#potential} (a deterministic
 * mean computed from the latent variables and data).
 *
 * Step size is tuned during warm-up by dual averaging (Hoffman & Gelman, 2014)
 * toward a target acceptance rate; a unit mass matrix is used.
 *
 * @example
 * const hmc = new HMC({ stepSize: 0.05, nSteps: 20 });
 * const { trace } = hmc.sample(model, { slope: 0, intercept: 0, sigma: 1 },
 *                              { nSamples: 1000, nWarmup: 500 });
 */
export class HMC {
  /**
   * @param {Object} [opts]
   * @param {number} [opts.stepSize=0.05] - Initial leapfrog step size (adapted in warm-up).
   * @param {number} [opts.nSteps=20] - Leapfrog steps per proposal.
   * @param {number} [opts.targetAccept=0.8] - Target acceptance for step-size adaptation.
   * @param {boolean} [opts.adapt=true] - Adapt the step size during warm-up.
   * @param {number} [opts.seed] - Optional RNG seed for reproducibility.
   */
  constructor({ stepSize = 0.05, nSteps = 20, targetAccept = 0.8, adapt = true, seed } = {}) {
    this.stepSize = stepSize;
    this.nSteps = nSteps;
    this.targetAccept = targetAccept;
    this.adapt = adapt;
    this.seed = seed;
  }

  /**
   * Run a single chain.
   *
   * @param {Model} model
   * @param {Object} initialValues - {name: number | number[]} starting point.
   * @param {Object} [opts]
   * @param {number} [opts.nSamples=1000]
   * @param {number} [opts.nWarmup=500]
   * @param {number} [opts.thin=1]
   * @param {boolean} [opts.progress=false]
   * @returns {{ trace: Object, acceptanceRate: number, stepSize: number,
   *            divergences: number, specs: Array }}
   */
  sample(model, initialValues, { nSamples = 1000, nWarmup = 500, thin = 1, progress = false } = {}) {
    const names = model.getFreeVariableNames().filter((n) => initialValues[n] !== undefined);
    // Layout: flatten scalars and 1-D arrays into a single vector.
    const specs = names.map((name) => {
      const v = initialValues[name];
      const isVec = Array.isArray(v);
      return { name, isVec, size: isVec ? v.length : 1 };
    });
    const dim = specs.reduce((s, sp) => s + sp.size, 0);

    const rng = makeRng(this.seed);
    const randn = () => {
      // Box–Muller
      const u1 = Math.max(rng(), 1e-12);
      const u2 = rng();
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    };

    // Flatten / unflatten helpers.
    const flatten = (dict) => {
      const q = new Float64Array(dim);
      let off = 0;
      for (const sp of specs) {
        if (sp.isVec) { const a = dict[sp.name]; for (let k = 0; k < sp.size; k++) q[off + k] = a[k]; }
        else q[off] = dict[sp.name];
        off += sp.size;
      }
      return q;
    };
    const unflatten = (q) => {
      const dict = {};
      let off = 0;
      for (const sp of specs) {
        if (sp.isVec) dict[sp.name] = Array.from(q.slice(off, off + sp.size));
        else dict[sp.name] = q[off];
        off += sp.size;
      }
      return dict;
    };

    // log p(q) and ∇ log p(q) as flat arrays, evaluated through the model.
    const logpGrad = (q) => {
      let logp = 0;
      const grad = new Float64Array(dim);
      tf.tidy(() => {
        const dict = {};
        let off = 0;
        for (const sp of specs) {
          dict[sp.name] = sp.isVec
            ? tf.tensor1d(Array.from(q.slice(off, off + sp.size)))
            : tf.scalar(q[off]);
          off += sp.size;
        }
        const res = model.logProbAndGradient(dict);
        logp = res.logProb;
        off = 0;
        for (const sp of specs) {
          const g = res.gradients[sp.name].dataSync();
          for (let k = 0; k < sp.size; k++) grad[off + k] = g[k] ?? 0;
          off += sp.size;
        }
      });
      return { logp, grad };
    };

    const kinetic = (p) => { let s = 0; for (let k = 0; k < dim; k++) s += 0.5 * p[k] * p[k]; return s; };

    // One HMC proposal via leapfrog. Returns the new state + acceptance prob.
    const step = (q0, eps) => {
      const q = Float64Array.from(q0);
      const p = new Float64Array(dim);
      for (let k = 0; k < dim; k++) p[k] = randn();
      const start = logpGrad(q);
      if (!Number.isFinite(start.logp)) return { q: q0, accept: 0, diverged: true };
      const H0 = -start.logp + kinetic(p);

      // Half momentum, then alternating full position / full momentum steps.
      let g = start.grad;
      for (let k = 0; k < dim; k++) p[k] += 0.5 * eps * g[k];
      let cur;
      for (let s = 0; s < this.nSteps; s++) {
        for (let k = 0; k < dim; k++) q[k] += eps * p[k];
        cur = logpGrad(q);
        const half = s === this.nSteps - 1 ? 0.5 : 1.0;
        for (let k = 0; k < dim; k++) p[k] += half * eps * cur.grad[k];
      }
      const logp1 = cur ? cur.logp : start.logp;
      if (!Number.isFinite(logp1)) return { q: q0, accept: 0, diverged: true };
      const H1 = -logp1 + kinetic(p);
      const dH = H0 - H1;
      const accept = Number.isFinite(dH) ? Math.min(1, Math.exp(dH)) : 0;
      const diverged = !Number.isFinite(dH) || Math.abs(dH) > 1000;
      if (rng() < accept) return { q, accept, diverged };
      return { q: q0, accept, diverged };
    };

    // Dual-averaging step-size adaptation state.
    let eps = this.stepSize;
    const mu = Math.log(10 * eps);
    let logEpsBar = 0, hBar = 0;
    const gamma = 0.05, t0 = 10, kappa = 0.75;

    let q = flatten(initialValues);
    const trace = {}; names.forEach((n) => (trace[n] = []));
    let accCount = 0, accTotal = 0, divergences = 0;
    const total = nWarmup + nSamples * thin;

    for (let i = 0; i < total; i++) {
      const { q: qNew, accept, diverged } = step(q, eps);
      q = qNew;
      if (diverged && i >= nWarmup) divergences++;
      accTotal++; if (accept > 0.5) accCount++;

      if (this.adapt && i < nWarmup) {
        const m = i + 1;
        const eta = 1 / (m + t0);
        hBar = (1 - eta) * hBar + eta * (this.targetAccept - accept);
        const logEps = mu - (Math.sqrt(m) / gamma) * hBar;
        const w = Math.pow(m, -kappa);
        logEpsBar = w * logEps + (1 - w) * logEpsBar;
        eps = Math.exp(logEps);
      } else if (i === nWarmup && this.adapt) {
        eps = Math.exp(logEpsBar);
      }

      if (i >= nWarmup && (i - nWarmup) % thin === 0) {
        const dict = unflatten(q);
        for (const n of names) trace[n].push(dict[n]);
      }
      if (progress && (i + 1) % Math.max(1, Math.floor(total / 10)) === 0) {
        const phase = i < nWarmup ? 'warmup' : 'sample';
        console.log(`HMC ${phase} ${Math.round((100 * (i + 1)) / total)}% | step=${eps.toExponential(2)} | accept=${(100 * accCount / accTotal).toFixed(0)}%`);
      }
    }

    return { trace, acceptanceRate: accCount / accTotal, stepSize: eps, divergences, specs };
  }

  /**
   * Run several independent chains (sequentially) from (optionally) jittered
   * starting points. Returns an array of single-chain results, ready for
   * {@link summary}.
   *
   * @param {Model} model
   * @param {Object|((chain:number)=>Object)} initial - Starting values, or a
   *   function returning starting values for each chain index.
   * @param {Object} [opts] - As {@link HMC#sample}, plus `chains` (default 4).
   * @returns {Array} per-chain results
   */
  sampleChains(model, initial, { chains = 4, ...opts } = {}) {
    const results = [];
    for (let c = 0; c < chains; c++) {
      const init = typeof initial === 'function' ? initial(c) : initial;
      const sampler = new HMC({ stepSize: this.stepSize, nSteps: this.nSteps, targetAccept: this.targetAccept, adapt: this.adapt, seed: this.seed !== undefined ? this.seed + c : undefined });
      results.push(sampler.sample(model, init, opts));
    }
    return results;
  }
}

/**
 * ArviZ-style posterior summary across one or more chains.
 *
 * @param {Array|Object} chainsOrResults - Array of chain results (`{trace}` from
 *   {@link HMC#sample}), an array of raw trace dicts, or a single trace dict.
 * @param {Object} [opts]
 * @param {number} [opts.hdi=0.94] - HDI mass (e.g. 0.94 → hdi_3%/hdi_97%).
 * @returns {Array<Object>} One row per scalar parameter component with
 *   `{ param, mean, sd, hdi_lo, hdi_hi, ess, rhat }`.
 */
export function summary(chainsOrResults, { hdi = 0.94 } = {}) {
  // Normalize to an array of trace dicts.
  let traces;
  if (Array.isArray(chainsOrResults)) {
    traces = chainsOrResults.map((c) => (c && c.trace ? c.trace : c));
  } else {
    traces = [chainsOrResults.trace ? chainsOrResults.trace : chainsOrResults];
  }
  const names = Object.keys(traces[0]);
  const rows = [];

  for (const name of names) {
    const first = traces[0][name][0];
    const size = Array.isArray(first) ? first.length : 1;
    for (let comp = 0; comp < size; comp++) {
      // Per-chain scalar series for this component.
      const perChain = traces.map((t) => t[name].map((v) => (Array.isArray(v) ? v[comp] : v)));
      const pooled = perChain.flat();
      const label = size > 1 ? `${name}[${comp}]` : name;

      const mean = pooled.reduce((a, b) => a + b, 0) / pooled.length;
      const variance = pooled.reduce((a, b) => a + (b - mean) ** 2, 0) / (pooled.length - 1 || 1);
      const sd = Math.sqrt(variance);
      const [lo, hi] = hdiInterval(pooled, hdi);
      const ess = perChain.reduce((s, ch) => s + effectiveSampleSize(ch), 0);
      const rhat = perChain.length > 1 ? gelmanRubin(perChain) : NaN;

      rows.push({ param: label, mean, sd, hdi_lo: lo, hdi_hi: hi, ess: Math.round(ess), rhat });
    }
  }
  return rows;
}

/** Highest-density interval of a 1-D sample. */
function hdiInterval(samples, mass) {
  const s = [...samples].sort((a, b) => a - b);
  const n = s.length;
  const w = Math.max(1, Math.floor(mass * n));
  let lo = s[0], hi = s[n - 1], best = Infinity;
  for (let i = 0; i + w - 1 < n; i++) {
    const width = s[i + w - 1] - s[i];
    if (width < best) { best = width; lo = s[i]; hi = s[i + w - 1]; }
  }
  return [lo, hi];
}

/** Small deterministic PRNG (mulberry32); falls back to Math.random if no seed. */
function makeRng(seed) {
  if (seed === undefined) return Math.random;
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

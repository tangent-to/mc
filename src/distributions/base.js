/**
 * Base class for mc distributions.
 *
 * Since 0.5.0, mc runs on plain numbers and arrays (no tensors): each mc
 * distribution wraps a @tangent.to/proba distribution and broadcasts it
 * over array-valued parameters and values. Vectorized likelihoods keep
 * working: `new Normal(muArray, sigma).logProb(yArray)` returns an array
 * of per-observation log-densities, which Model sums.
 */

import { getRng } from '../rng.js';

/**
 * Determine whether the first constructor argument is an options object
 * (e.g. `new Normal({ mu, sigma })`) rather than a positional value.
 *
 * Plain objects are treated as options; numbers and arrays are not. This
 * mirrors the dual positional/options constructor convention used across
 * the sibling `@tangent.to/ds` package.
 *
 * @param {*} value - First constructor argument
 * @returns {boolean} True if `value` should be interpreted as an options object
 */
export function isOptions(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/** Read tensors (or tensor-likes) back to plain numbers/arrays. */
function toPlain(v) {
  return v && typeof v.arraySync === 'function' ? v.arraySync() : v;
}

/**
 * Base class for probability distributions.
 *
 * Subclasses set `this._dist` (a @tangent.to/proba distribution) in their
 * constructor and implement `_params()` returning the proba parameter
 * object (fields may be numbers or arrays of numbers).
 */
export class Distribution {
  /**
   * Create a base distribution; subclasses set `this._dist` and parameters.
   * @param {string} [name='Distribution'] - Name of the distribution
   */
  constructor(name = 'Distribution') {
    this.name = name;
    this.observed = null;
  }

  /**
   * The proba parameter object for this distribution; subclasses must implement.
   * @returns {Object} proba parameter object (fields may be numbers or arrays)
   */
  _params() {
    throw new Error('_params must be implemented by subclass');
  }

  /**
   * Broadcast length across value and parameters (0 = all scalar).
   * @param {number|Array} value - Value(s) whose length participates in broadcasting
   * @returns {number} The broadcast length (0 when every input is scalar)
   */
  _len(value) {
    let n = Array.isArray(value) ? value.length : 0;
    for (const v of Object.values(this._params())) {
      if (Array.isArray(v)) n = Math.max(n, v.length);
    }
    return n;
  }

  /**
   * The proba parameter object with each array parameter indexed at `i`.
   * @param {number} i - Broadcast index
   * @returns {Object} Per-element parameter object (scalars passed through)
   */
  _paramsAt(i) {
    const out = {};
    for (const [k, v] of Object.entries(this._params())) {
      out[k] = Array.isArray(v) ? v[i] : v;
    }
    return out;
  }

  /**
   * Log probability density/mass function. Broadcasts over array values
   * and/or array parameters.
   *
   * @param {number|Array|Object} value - Value(s) to evaluate
   * @returns {number|Array<number>} Log probability, elementwise for arrays
   */
  logProb(value) {
    const x = toPlain(value);
    const n = this._len(x);
    const base = this._params();
    if (n === 0) return this._dist.logpdf(x, base);
    // Hot path: reuse one params object across elements (proba functions do
    // not retain their params argument), avoiding n allocations per call.
    const keys = Object.keys(base);
    const cur = { ...base };
    const xIsArr = Array.isArray(x);
    const out = new Array(n);
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < keys.length; k++) {
        const v = base[keys[k]];
        if (Array.isArray(v)) cur[keys[k]] = v[i];
      }
      out[i] = this._dist.logpdf(xIsArr ? x[i] : x, cur);
    }
    return out;
  }

  /**
   * Alias for {@link Distribution#logProb}, matching the `@tangent.to/proba`
   * distribution contract (which names the method `logpdf`). Lets code written
   * against proba's distributions work unchanged on mc's.
   *
   * @param {number|Array|Object} value - Value(s) to evaluate
   * @returns {number|Array<number>}
   */
  logpdf(value) {
    return this.logProb(value);
  }

  /**
   * Derivative of logProb with respect to the value, elementwise.
   * Used by Model.logProbAndGradient for analytic prior gradients.
   * Discrete distributions return 0 (no dx in their gradient contract).
   *
   * @param {number|Array} value - Value(s) at which to differentiate
   * @returns {number|Array<number>}
   */
  dlogProbDx(value) {
    const x = toPlain(value);
    const grad = (xi, params) => {
      const d = this._dist.dlogpdf(xi, params);
      return d.dx !== undefined ? d.dx : 0;
    };
    const n = this._len(x);
    if (n === 0) return grad(x, this._params());
    const out = new Array(n);
    for (let i = 0; i < n; i++) {
      out[i] = grad(Array.isArray(x) ? x[i] : x, this._paramsAt(i));
    }
    return out;
  }

  /**
   * Probability density/mass function, `exp(logProb(value))`.
   * @param {number|Array} value - Value(s) to evaluate
   * @returns {number|Array<number>}
   */
  pdf(value) {
    const lp = this.logProb(value);
    return Array.isArray(lp) ? lp.map(Math.exp) : Math.exp(lp);
  }

  /**
   * Cumulative distribution function (scalar parameters).
   * @param {number} value
   * @returns {number}
   */
  cdf(value) {
    return this._dist.cdf(toPlain(value), this._params());
  }

  /**
   * Quantile (inverse cdf) function (scalar parameters).
   * @param {number} p - Probability in [0, 1]
   * @returns {number}
   */
  quantile(p) {
    return this._dist.quantile(p, this._params());
  }

  /**
   * Sample from the distribution using the package RNG (see setRandomSeed).
   * `sample()` / `sample([])` return a number; `sample(n)` / `sample([n])`
   * return an Array of n draws.
   *
   * @param {number|Array<number>} [shape=[]] - Number of samples
   * @returns {number|Array<number>}
   */
  sample(shape = []) {
    const n = Array.isArray(shape) ? (shape.length ? shape[0] : null) : shape;
    if (n === null || n === undefined) return this._dist.sample(this._params(), getRng());
    return this._dist.sampleN(this._params(), getRng(), n);
  }

  /**
   * Set observed data for this distribution
   * @param {number|Array} data - Observed data
   * @returns {Distribution} this, for chaining
   */
  observe(data) {
    this.observed = toPlain(data);
    return this;
  }

  /**
   * Get the mean of the distribution
   * @returns {number|Array<number>} The mean
   */
  mean() {
    const n = this._len(null);
    if (n === 0) return this._dist.mean(this._params());
    return Array.from({ length: n }, (_, i) => this._dist.mean(this._paramsAt(i)));
  }

  /**
   * Get the variance of the distribution
   * @returns {number|Array<number>} The variance
   */
  variance() {
    const n = this._len(null);
    if (n === 0) return this._dist.variance(this._params());
    return Array.from({ length: n }, (_, i) => this._dist.variance(this._paramsAt(i)));
  }

  /**
   * Get the distribution's parameters as a plain object.
   * Subclasses override to expose their specific parameters.
   * @returns {Object} Parameters
   */
  getParams() {
    return {};
  }
}

import { uniform } from '@tangent.to/proba';
import { log, mul, sub } from '@tangent.to/grad';
import { Distribution, isOptions } from './base.js';

/**
 * Continuous uniform distribution on [lower, upper].
 */
export class Uniform extends Distribution {
  /**
   * Create a continuous uniform distribution on [lower, upper].
   * @param {number|Array|Object} lower - Lower bound, or an options object
   *   `{ lower | min, upper | max, name }`
   * @param {number|Array} [upper] - Upper bound
   * @param {string} [name] - Name of the distribution
   */
  constructor(lower = 0, upper = 1, name = 'Uniform') {
    super(name);
    if (isOptions(lower)) {
      const o = lower;
      this.name = o.name ?? 'Uniform';
      lower = o.lower ?? o.min ?? 0;
      upper = o.upper ?? o.max ?? 1;
    }
    this.lower = lower;
    this.upper = upper;
    this._dist = uniform;
  }

  /**
   * The proba parameter object for this distribution (proba `{low, high}` keys).
   * @returns {{low: number|Array, high: number|Array}}
   */
  _params() {
    return { low: this.lower, high: this.upper };
  }

  logDensity(value) {
    // -log(high - low) per element, assuming every value lies in the support.
    // With numeric bounds the check is made here and -Infinity returned, as
    // logProb does; with Var bounds it cannot be, and the caller is expected
    // to be a sampler that keeps the value inside by construction.
    const n = Array.isArray(value) ? value.length : 1;
    if (typeof this.lower === 'number' && typeof this.upper === 'number') {
      const xs = Array.isArray(value) ? value : [value];
      if (xs.some((x) => x < this.lower || x > this.upper)) return mul(-Infinity, 1);
    }
    return mul(-n, log(sub(this.upper, this.lower)));
  }

  /**
   * Get the distribution's parameters.
   * @returns {{lower: number|Array, upper: number|Array}}
   */
  getParams() {
    return { lower: this.lower, upper: this.upper };
  }
}

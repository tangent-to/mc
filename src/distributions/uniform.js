import { uniform } from '@tangent.to/proba';
import { Distribution, isOptions } from './base.js';

/**
 * Continuous uniform distribution on [lower, upper].
 */
export class Uniform extends Distribution {
  /**
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

  _params() {
    return { low: this.lower, high: this.upper };
  }

  /**
   * Get the distribution's parameters.
   * @returns {{lower: number|Array, upper: number|Array}}
   */
  getParams() {
    return { lower: this.lower, upper: this.upper };
  }
}

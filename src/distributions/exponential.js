import { exponential } from '@tangent.to/proba';
import { Distribution, isOptions } from './base.js';

/**
 * Exponential distribution, rate parameterized
 *
 * $$ p(x | \lambda) = \lambda e^{-\lambda x}, \quad x \ge 0 $$
 *
 * @see {@link https://en.wikipedia.org/wiki/Exponential_distribution|Exponential Distribution}
 */
export class Exponential extends Distribution {
  /**
   * Accepts either positional arguments or a single options object, matching the
   * dual-constructor convention of `@tangent.to/ds`.
   *
   * @param {number|Array|Object} lambda - Rate parameter, lambda > 0, or an options object
   *   `{ lambda | rate, name }`
   * @param {string} [name] - Name of the distribution
   *
   * @example
   * new Exponential(2)
   * @example
   * new Exponential({ rate: 2 })
   */
  constructor(lambda = 1, name = 'Exponential') {
    super(name);
    if (isOptions(lambda)) {
      const o = lambda;
      this.name = o.name ?? 'Exponential';
      lambda = o.lambda ?? o.rate ?? 1;
    }
    this.lambda = lambda;
    this._dist = exponential;
  }

  /**
   * The proba parameter object for this distribution.
   */
  _params() {
    return { lambda: this.lambda };
  }

  /**
   * Get the distribution's parameters.
   */
  getParams() {
    return { lambda: this.lambda };
  }
}

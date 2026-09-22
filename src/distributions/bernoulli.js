import { bernoulli } from '@tangent.to/proba';
import { Distribution, isOptions } from './base.js';

/**
 * Bernoulli distribution for binary outcomes.
 */
export class Bernoulli extends Distribution {
  /**
   * Create a Bernoulli distribution.
   * @param {number|Array|Object} p - Probability of success in [0, 1], or an
   *   options object `{ p, name }`
   * @param {string} [name] - Name of the distribution
   */
  constructor(p = 0.5, name = 'Bernoulli') {
    super(name);
    if (isOptions(p)) {
      const o = p;
      this.name = o.name ?? 'Bernoulli';
      p = o.p ?? 0.5;
    }
    this.p = p;
    this._dist = bernoulli;
  }

  /**
   * The proba parameter object for this distribution.
   * @returns {{p: number|Array}}
   */
  _params() {
    return { p: this.p };
  }

  /**
   * Get the distribution's parameters.
   * @returns {{p: number|Array}}
   */
  getParams() {
    return { p: this.p };
  }
}

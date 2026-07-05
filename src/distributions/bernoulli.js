import { bernoulli } from '@tangent.to/proba';
import { Distribution, isOptions } from './base.js';

/**
 * Bernoulli distribution for binary outcomes.
 */
export class Bernoulli extends Distribution {
  /**
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

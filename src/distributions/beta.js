import { beta as probaBeta } from '@tangent.to/proba';
import { Distribution, isOptions } from './base.js';

/**
 * Beta distribution on (0, 1).
 */
export class Beta extends Distribution {
  /**
   * @param {number|Array|Object} alpha - First shape, or an options object
   *   `{ alpha, beta, name }`
   * @param {number|Array} [beta] - Second shape
   * @param {string} [name] - Name of the distribution
   */
  constructor(alpha = 1, beta = 1, name = 'Beta') {
    super(name);
    if (isOptions(alpha)) {
      const o = alpha;
      this.name = o.name ?? 'Beta';
      alpha = o.alpha ?? 1;
      beta = o.beta ?? 1;
    }
    this.alpha = alpha;
    this.beta = beta;
    this._dist = probaBeta;
  }

  _params() {
    return { alpha: this.alpha, beta: this.beta };
  }

  /**
   * Get the distribution's parameters.
   * @returns {{alpha: number|Array, beta: number|Array}}
   */
  getParams() {
    return { alpha: this.alpha, beta: this.beta };
  }
}

import { gamma } from '@tangent.to/proba';
import { Distribution, isOptions } from './base.js';

/**
 * Gamma distribution (shape/rate parameterization, PyMC convention):
 * mean = alpha / beta.
 */
export class Gamma extends Distribution {
  /**
   * @param {number|Array|Object} alpha - Shape, or an options object
   *   `{ alpha | shape, beta | rate, name }`
   * @param {number|Array} [beta] - Rate
   * @param {string} [name] - Name of the distribution
   */
  constructor(alpha = 1, beta = 1, name = 'Gamma') {
    super(name);
    if (isOptions(alpha)) {
      const o = alpha;
      this.name = o.name ?? 'Gamma';
      alpha = o.alpha ?? o.shape ?? 1;
      beta = o.beta ?? o.rate ?? 1;
    }
    this.alpha = alpha;
    this.beta = beta;
    this._dist = gamma;
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

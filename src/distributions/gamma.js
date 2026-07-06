import { gamma } from '@tangent.to/proba';
import { Distribution, isOptions } from './base.js';

/**
 * Gamma distribution (shape/rate parameterization, PyMC convention):
 * mean = alpha / beta.
 */
export class Gamma extends Distribution {
  /**
   * Shape/RATE parameterization (PyMC/Stan convention): mean = alpha / beta.
   * Note this differs from R and `@tangent.to/ds`, which use shape/SCALE
   * (scale = 1 / rate). A `scale` key is therefore rejected here rather than
   * silently misread as a rate — pass `rate` (or `beta`) explicitly.
   *
   * @param {number|Array|Object} alpha - Shape, or an options object
   *   `{ alpha | shape, beta | rate, name }`
   * @param {number|Array} [beta] - Rate (NOT scale)
   * @param {string} [name] - Name of the distribution
   */
  constructor(alpha = 1, beta = 1, name = 'Gamma') {
    super(name);
    if (isOptions(alpha)) {
      const o = alpha;
      if ('scale' in o) {
        throw new Error(
          'Gamma uses shape/RATE ({ alpha|shape, beta|rate }); got a `scale` key. ' +
          'This is the R / @tangent.to/ds convention — pass rate = 1/scale instead.',
        );
      }
      this.name = o.name ?? 'Gamma';
      alpha = o.alpha ?? o.shape ?? 1;
      beta = o.beta ?? o.rate ?? 1;
    }
    this.alpha = alpha;
    this.beta = beta;
    this._dist = gamma;
  }

  /**
   * The proba parameter object for this distribution (shape/rate).
   * @returns {{alpha: number|Array, beta: number|Array}}
   */
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

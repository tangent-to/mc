import { beta as probaBeta } from '@tangent.to/proba';
import { add, lgamma, log, mul, sub, sum } from '@tangent.to/grad';
import { Distribution, isOptions } from './base.js';

/**
 * Beta distribution on (0, 1).
 */
export class Beta extends Distribution {
  /**
   * Create a Beta distribution.
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

  /**
   * The proba parameter object for this distribution.
   * @returns {{alpha: number|Array, beta: number|Array}}
   */
  _params() {
    return { alpha: this.alpha, beta: this.beta };
  }

  logDensity(value) {
    // (alpha - 1) log x + (beta - 1) log(1 - x) - lbeta(alpha, beta), 0 < x < 1,
    // with lbeta = lgamma(a) + lgamma(b) - lgamma(a + b).
    const lbeta = sub(add(lgamma(this.alpha), lgamma(this.beta)), lgamma(add(this.alpha, this.beta)));
    const perElement = sub(
      add(mul(sub(this.alpha, 1), log(value)), mul(sub(this.beta, 1), log(sub(1, value)))),
      lbeta,
    );
    return sum(perElement);
  }

  /**
   * Get the distribution's parameters.
   * @returns {{alpha: number|Array, beta: number|Array}}
   */
  getParams() {
    return { alpha: this.alpha, beta: this.beta };
  }
}

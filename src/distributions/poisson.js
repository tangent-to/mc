import { poisson } from '@tangent.to/proba';
import { Distribution, isOptions } from './base.js';

/**
 * Poisson distribution, for counts
 *
 * $$ p(k | \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots $$
 *
 * As an observation model the rate is usually an expression, `exp(...)` of a
 * linear predictor, which is what keeps it positive.
 *
 * @see {@link https://en.wikipedia.org/wiki/Poisson_distribution|Poisson Distribution}
 */
export class Poisson extends Distribution {
  /**
   * Accepts either positional arguments or a single options object, matching the
   * dual-constructor convention of `@tangent.to/ds`.
   *
   * @param {number|Array|Object} lambda - Rate, lambda > 0, or an options object
   *   `{ lambda | rate | mu, name }`
   * @param {string} [name] - Name of the distribution
   *
   * @example
   * new Poisson(3)
   * @example
   * new Poisson({ rate: 3 })
   */
  constructor(lambda = 1, name = 'Poisson') {
    super(name);
    if (isOptions(lambda)) {
      const o = lambda;
      this.name = o.name ?? 'Poisson';
      lambda = o.lambda ?? o.rate ?? o.mu ?? 1;
    }
    this.lambda = lambda;
    this._dist = poisson;
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

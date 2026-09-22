import { studentT } from '@tangent.to/proba';
import { Distribution, isOptions } from './base.js';

/**
 * Student-t distribution, with location and scale
 *
 * $$ p(x | \nu, \mu, \sigma) = \frac{\Gamma((\nu+1)/2)}{\Gamma(\nu/2)\sqrt{\nu\pi}\,\sigma}
 *    \left(1 + \frac{1}{\nu}\left(\frac{x-\mu}{\sigma}\right)^2\right)^{-(\nu+1)/2} $$
 *
 * A robust alternative to the Normal as an observation model: its heavy tails
 * let an outlying observation pull the mean less. `nu` may be a free variable.
 *
 * @see {@link https://en.wikipedia.org/wiki/Student%27s_t-distribution|Student's t-distribution}
 */
export class StudentT extends Distribution {
  /**
   * Accepts either positional arguments or a single options object, matching the
   * dual-constructor convention of `@tangent.to/ds`.
   *
   * @param {number|Array|Object} nu - Degrees of freedom, nu > 0, or an options object
   *   `{ nu | df, mu | mean, sigma | sd | std, name }`
   * @param {number|Array} [mu] - Location
   * @param {number|Array} [sigma] - Scale, sigma > 0
   * @param {string} [name] - Name of the distribution
   *
   * @example
   * new StudentT(4, 0, 1)
   * @example
   * new StudentT({ df: 4, mean: 0, sd: 1 })
   */
  constructor(nu = 1, mu = 0, sigma = 1, name = 'StudentT') {
    super(name);
    if (isOptions(nu)) {
      const o = nu;
      this.name = o.name ?? 'StudentT';
      nu = o.nu ?? o.df ?? 1;
      mu = o.mu ?? o.mean ?? 0;
      sigma = o.sigma ?? o.sd ?? o.std ?? 1;
    }
    this.nu = nu;
    this.mu = mu;
    this.sigma = sigma;
    this._dist = studentT;
  }

  /**
   * The proba parameter object for this distribution.
   */
  _params() {
    return { nu: this.nu, mu: this.mu, sigma: this.sigma };
  }

  /**
   * Get the distribution's parameters.
   */
  getParams() {
    return { nu: this.nu, mu: this.mu, sigma: this.sigma };
  }
}

import { normal } from '@tangent.to/proba';
import { Distribution, isOptions } from './base.js';

/**
 * Normal (Gaussian) distribution
 *
 * $$ p(x | \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) $$
 *
 * @see {@link https://en.wikipedia.org/wiki/Normal_distribution|Normal Distribution}
 */
export class Normal extends Distribution {
  /**
   * Accepts either positional arguments or a single options object, matching the
   * dual-constructor convention of `@tangent.to/ds`.
   *
   * @param {number|Array|Object} mu - Mean parameter, or an options object
   *   `{ mu | mean, sigma | sd | std, name }`
   * @param {number|Array} [sigma] - Standard deviation, sigma > 0
   * @param {string} [name] - Name of the distribution
   *
   * @example
   * new Normal(0, 1)
   * @example
   * new Normal({ mean: 0, sd: 1 })
   */
  constructor(mu = 0, sigma = 1, name = 'Normal') {
    super(name);
    if (isOptions(mu)) {
      const o = mu;
      this.name = o.name ?? 'Normal';
      mu = o.mu ?? o.mean ?? 0;
      sigma = o.sigma ?? o.sd ?? o.std ?? 1;
    }
    this.mu = mu;
    this.sigma = sigma;
    this._dist = normal;
  }

  _params() {
    return { mu: this.mu, sigma: this.sigma };
  }

  /**
   * Get the distribution's parameters.
   * @returns {{mu: number|Array, sigma: number|Array}}
   */
  getParams() {
    return { mu: this.mu, sigma: this.sigma };
  }
}

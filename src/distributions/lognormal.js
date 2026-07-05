import { lognormal } from '@tangent.to/proba';
import { Distribution, isOptions } from './base.js';

/**
 * Log-normal distribution: if log X ~ Normal(mu, sigma^2) then
 * X ~ LogNormal(mu, sigma). Parameters are on the log scale.
 */
export class Lognormal extends Distribution {
  /**
   * @param {number|Array|Object} mu - Log-scale location, or an options object
   *   `{ mu | mean, sigma | sd | std, name }`
   * @param {number|Array} [sigma] - Log-scale standard deviation
   * @param {string} [name] - Name of the distribution
   */
  constructor(mu = 0, sigma = 1, name = 'Lognormal') {
    super(name);
    if (isOptions(mu)) {
      const o = mu;
      this.name = o.name ?? 'Lognormal';
      mu = o.mu ?? o.mean ?? 0;
      sigma = o.sigma ?? o.sd ?? o.std ?? 1;
    }
    this.mu = mu;
    this.sigma = sigma;
    this._dist = lognormal;
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

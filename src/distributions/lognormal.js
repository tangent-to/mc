import { lognormal } from '@tangent.to/proba';
import { div, log, mul, square, sub, sum } from '@tangent.to/grad';
const LN_SQRT_2PI = 0.9189385332046727;
import { Distribution, isOptions } from './base.js';

/**
 * Log-normal distribution: if log X ~ Normal(mu, sigma^2) then
 * X ~ LogNormal(mu, sigma). Parameters are on the log scale.
 */
export class Lognormal extends Distribution {
  /**
   * Create a log-normal distribution (parameters on the log scale).
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

  /**
   * The proba parameter object for this distribution.
   * @returns {{mu: number|Array, sigma: number|Array}}
   */
  _params() {
    return { mu: this.mu, sigma: this.sigma };
  }

  logDensity(value) {
    // -log x - log sigma - ln sqrt(2 pi) - z^2 / 2 with z = (log x - mu) / sigma.
    const lx = log(value);
    const z = div(sub(lx, this.mu), this.sigma);
    return sum(sub(sub(sub(mul(-0.5, square(z)), lx), log(this.sigma)), LN_SQRT_2PI));
  }

  /**
   * Get the distribution's parameters.
   * @returns {{mu: number|Array, sigma: number|Array}}
   */
  getParams() {
    return { mu: this.mu, sigma: this.sigma };
  }
}

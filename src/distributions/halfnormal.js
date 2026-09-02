import { halfnormal } from '@tangent.to/proba';
import { add, div, log, mul, square, sub, sum } from '@tangent.to/grad';
const LN_SQRT_2_OVER_PI = -0.2257913526447274;
import { Distribution, isOptions } from './base.js';

/**
 * Half-normal distribution on [0, Infinity) — the absolute value of a
 * Normal(0, sigma^2). A standard weakly-informative prior for scales.
 */
export class HalfNormal extends Distribution {
  /**
   * Create a half-normal distribution.
   * @param {number|Array|Object} sigma - Scale, or an options object
   *   `{ sigma | sd | std | scale, name }`
   * @param {string} [name] - Name of the distribution
   */
  constructor(sigma = 1, name = 'HalfNormal') {
    super(name);
    if (isOptions(sigma)) {
      const o = sigma;
      this.name = o.name ?? 'HalfNormal';
      sigma = o.sigma ?? o.sd ?? o.std ?? o.scale ?? 1;
    }
    this.sigma = sigma;
    this._dist = halfnormal;
  }

  /**
   * The proba parameter object for this distribution.
   * @returns {{sigma: number|Array}}
   */
  _params() {
    return { sigma: this.sigma };
  }

  logDensity(value) {
    // ln sqrt(2/pi) - log sigma - z^2 / 2 for x >= 0. The support is the
    // sampler's business (it moves through log sigma), so no clamp here.
    const z = div(value, this.sigma);
    return sum(add(sub(mul(-0.5, square(z)), log(this.sigma)), LN_SQRT_2_OVER_PI));
  }

  /**
   * Get the distribution's parameters.
   * @returns {{sigma: number|Array}}
   */
  getParams() {
    return { sigma: this.sigma };
  }
}

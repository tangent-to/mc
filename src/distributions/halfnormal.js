import * as tf from '@tensorflow/tfjs';
import { Distribution } from './base.js';

/**
 * Half-normal distribution
 *
 * The distribution of $|Z|$ where $Z \sim \mathcal{N}(0, \sigma^2)$; a positive
 * variable concentrated near zero. Commonly used as a weakly-informative prior
 * for scale / standard-deviation parameters (variance components).
 *
 * Probability density function (for $x \ge 0$):
 * $$
 * p(x \mid \sigma) = \frac{\sqrt{2}}{\sigma\sqrt{\pi}}
 *   \exp\!\left(-\frac{x^2}{2\sigma^2}\right)
 * $$
 *
 * @see {@link https://en.wikipedia.org/wiki/Half-normal_distribution|Half-normal distribution}
 */
export class HalfNormal extends Distribution {
  /**
   * @param {number|tf.Tensor} sigma - Scale parameter ($\sigma > 0$)
   * @param {string} name - Name of the distribution
   */
  constructor(sigma = 1, name = 'HalfNormal') {
    super(name);
    this.sigma = typeof sigma === 'number' ? tf.scalar(sigma) : sigma;
  }

  /**
   * Log probability density function.
   *
   * $$
   * \log p(x) = \tfrac{1}{2}\log\frac{2}{\pi} - \log\sigma - \frac{x^2}{2\sigma^2},
   * \quad x \ge 0
   * $$
   *
   * Returns $-\infty$ for negative inputs.
   *
   * @param {tf.Tensor|number} value - Value to evaluate ($x \ge 0$)
   * @returns {tf.Tensor} Log probability density
   */
  logProb(value) {
    return tf.tidy(() => {
      const x = typeof value === 'number' ? tf.scalar(value) : value;
      const c = 0.5 * Math.log(2 / Math.PI);
      const logDensity = tf.sub(
        tf.sub(tf.scalar(c), tf.log(this.sigma)),
        tf.mul(0.5, tf.square(tf.div(x, this.sigma)))
      );
      // Half-normal support is x >= 0; penalize negative values.
      const negInf = tf.fill(x.shape, -Infinity);
      return tf.where(tf.greaterEqual(x, 0), logDensity, negInf);
    });
  }

  /**
   * Sample from the half-normal distribution: $|\sigma Z|$, $Z \sim \mathcal{N}(0,1)$.
   * @param {number|Array<number>} shape - Shape of samples to generate
   * @returns {tf.Tensor} Samples
   */
  sample(shape = []) {
    return tf.tidy(() => {
      const sampleShape = Array.isArray(shape) ? shape : [shape];
      const z = tf.randomNormal(sampleShape);
      return tf.abs(tf.mul(z, this.sigma));
    });
  }

  /** Mean of the distribution: $\sigma\sqrt{2/\pi}$. */
  mean() {
    return tf.tidy(() => tf.mul(this.sigma, Math.sqrt(2 / Math.PI)));
  }
}

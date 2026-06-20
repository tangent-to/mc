import * as tf from '@tensorflow/tfjs-node';
import { Distribution, isOptions } from './base.js';

/**
 * Normal (Gaussian) distribution
 *
 * Probability density function:
 * $$
 * p(x | \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
 * $$
 *
 * @see {@link https://en.wikipedia.org/wiki/Normal_distribution|Normal Distribution}
 */
export class Normal extends Distribution {
  /**
   * Accepts either positional arguments or a single options object, matching the
   * dual-constructor convention of `@tangent.to/ds`.
   *
   * @param {number|tf.Tensor|Object} mu - Mean parameter $\mu$, or an options object
   *   `{ mu | mean, sigma | sd, name }`
   * @param {number|tf.Tensor} [sigma] - Standard deviation parameter $\sigma > 0$
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
    this.mu = typeof mu === 'number' ? tf.scalar(mu) : mu;
    this.sigma = typeof sigma === 'number' ? tf.scalar(sigma) : sigma;
  }

  /**
   * Log probability density function
   *
   * $$
   * \log p(x | \mu, \sigma) = -\frac{1}{2}\log(2\pi) - \log(\sigma) - \frac{(x-\mu)^2}{2\sigma^2}
   * $$
   *
   * @param {tf.Tensor|number} value - Value to evaluate
   * @returns {tf.Tensor} Log probability
   */
  logProb(value) {
    return tf.tidy(() => {
      const x = typeof value === 'number' ? tf.scalar(value) : value;

      // log(p(x)) = -0.5 * log(2π) - log(σ) - 0.5 * ((x - μ) / σ)^2
      const logNormalization = tf.log(tf.mul(this.sigma, Math.sqrt(2 * Math.PI)));
      const logKernel = tf.mul(
        -0.5,
        tf.square(tf.div(tf.sub(x, this.mu), this.sigma))
      );

      return tf.sub(logKernel, logNormalization);
    });
  }

  /**
   * Sample from the normal distribution
   * @param {number|Array<number>} shape - Shape of samples to generate
   * @returns {tf.Tensor} Samples
   */
  sample(shape = []) {
    return tf.tidy(() => {
      const sampleShape = Array.isArray(shape) ? shape : [shape];
      const standardNormal = tf.randomNormal(sampleShape);
      return tf.add(tf.mul(standardNormal, this.sigma), this.mu);
    });
  }

  /**
   * Get the mean of the distribution
   */
  mean() {
    return this.mu;
  }

  /**
   * Get the variance of the distribution
   */
  variance() {
    return tf.square(this.sigma);
  }

  /**
   * Get the distribution's parameters.
   * @returns {{mu: number, sigma: number}}
   */
  getParams() {
    return { mu: this.mu.arraySync(), sigma: this.sigma.arraySync() };
  }
}

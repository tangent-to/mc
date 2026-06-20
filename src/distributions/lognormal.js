import * as tf from '@tensorflow/tfjs-node';
import { Distribution } from './base.js';

/**
 * Log-normal distribution
 *
 * A positive random variable whose logarithm is normally distributed:
 * if $\log X \sim \mathcal{N}(\mu, \sigma^2)$ then $X \sim \text{LogNormal}(\mu, \sigma)$.
 *
 * Probability density function (for $x > 0$):
 * $$
 * p(x \mid \mu, \sigma) = \frac{1}{x\,\sigma\sqrt{2\pi}}
 *   \exp\!\left(-\frac{(\log x - \mu)^2}{2\sigma^2}\right)
 * $$
 *
 * Useful as a weakly-informative prior for strictly positive quantities
 * (rates, scales, plateaus).
 *
 * @see {@link https://en.wikipedia.org/wiki/Log-normal_distribution|Log-normal distribution}
 */
export class Lognormal extends Distribution {
  /**
   * @param {number|tf.Tensor} mu - Mean of the underlying normal (log-scale)
   * @param {number|tf.Tensor} sigma - Std-dev of the underlying normal ($\sigma > 0$)
   * @param {string} name - Name of the distribution
   */
  constructor(mu = 0, sigma = 1, name = 'Lognormal') {
    super(name);
    this.mu = typeof mu === 'number' ? tf.scalar(mu) : mu;
    this.sigma = typeof sigma === 'number' ? tf.scalar(sigma) : sigma;
  }

  /**
   * Log probability density function.
   *
   * $$
   * \log p(x) = -\log x - \log \sigma - \tfrac{1}{2}\log(2\pi)
   *            - \frac{(\log x - \mu)^2}{2\sigma^2}, \quad x > 0
   * $$
   *
   * @param {tf.Tensor|number} value - Value to evaluate ($x > 0$)
   * @returns {tf.Tensor} Log probability density
   */
  logProb(value) {
    return tf.tidy(() => {
      const x = typeof value === 'number' ? tf.scalar(value) : value;
      const logX = tf.log(x);
      // Normal(logX | mu, sigma) on the log-scale, minus the Jacobian term log(x).
      const logNormalization = tf.log(tf.mul(this.sigma, Math.sqrt(2 * Math.PI)));
      const logKernel = tf.mul(
        -0.5,
        tf.square(tf.div(tf.sub(logX, this.mu), this.sigma))
      );
      return tf.sub(tf.sub(logKernel, logNormalization), logX);
    });
  }

  /**
   * Sample from the log-normal distribution: $\exp(\mu + \sigma Z)$, $Z \sim \mathcal{N}(0,1)$.
   * @param {number|Array<number>} shape - Shape of samples to generate
   * @returns {tf.Tensor} Samples
   */
  sample(shape = []) {
    return tf.tidy(() => {
      const sampleShape = Array.isArray(shape) ? shape : [shape];
      const z = tf.randomNormal(sampleShape);
      return tf.exp(tf.add(tf.mul(z, this.sigma), this.mu));
    });
  }

  /** Mean of the distribution: $\exp(\mu + \sigma^2/2)$. */
  mean() {
    return tf.tidy(() => tf.exp(tf.add(this.mu, tf.mul(0.5, tf.square(this.sigma)))));
  }
}

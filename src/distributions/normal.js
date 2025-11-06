import * as tf from '@tensorflow/tfjs-node';
import { Distribution } from './base.js';

/**
 * Normal (Gaussian) distribution
 */
export class Normal extends Distribution {
  /**
   * @param {number|tf.Tensor} mu - Mean parameter
   * @param {number|tf.Tensor} sigma - Standard deviation parameter (must be positive)
   * @param {string} name - Name of the distribution
   */
  constructor(mu = 0, sigma = 1, name = 'Normal') {
    super(name);
    this.mu = typeof mu === 'number' ? tf.scalar(mu) : mu;
    this.sigma = typeof sigma === 'number' ? tf.scalar(sigma) : sigma;
  }

  /**
   * Log probability density function
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
}

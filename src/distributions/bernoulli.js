import * as tf from '@tensorflow/tfjs-node';
import { Distribution } from './base.js';

/**
 * Bernoulli distribution for binary outcomes
 */
export class Bernoulli extends Distribution {
  /**
   * @param {number|tf.Tensor} p - Probability of success (must be in [0, 1])
   * @param {string} name - Name of the distribution
   */
  constructor(p = 0.5, name = 'Bernoulli') {
    super(name);
    this.p = typeof p === 'number' ? tf.scalar(p) : p;
  }

  /**
   * Log probability mass function
   * @param {tf.Tensor|number} value - Value to evaluate (0 or 1)
   * @returns {tf.Tensor} Log probability
   */
  logProb(value) {
    return tf.tidy(() => {
      const x = typeof value === 'number' ? tf.scalar(value) : value;

      // log(p(x)) = x * log(p) + (1 - x) * log(1 - p)
      const logP = tf.log(this.p);
      const log1MinusP = tf.log(tf.sub(1, this.p));

      return tf.add(
        tf.mul(x, logP),
        tf.mul(tf.sub(1, x), log1MinusP)
      );
    });
  }

  /**
   * Sample from the Bernoulli distribution
   * @param {number|Array<number>} shape - Shape of samples to generate
   * @returns {tf.Tensor} Samples (0 or 1)
   */
  sample(shape = []) {
    return tf.tidy(() => {
      const sampleShape = Array.isArray(shape) ? shape : [shape];
      const uniformSamples = tf.randomUniform(sampleShape);
      return tf.cast(tf.less(uniformSamples, this.p), 'float32');
    });
  }

  /**
   * Get the mean of the distribution
   */
  mean() {
    return this.p;
  }

  /**
   * Get the variance of the distribution
   */
  variance() {
    return tf.mul(this.p, tf.sub(1, this.p));
  }
}

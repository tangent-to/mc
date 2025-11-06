import * as tf from '@tensorflow/tfjs-node';

/**
 * Base class for probability distributions.
 * Provides common interface for all distributions.
 */
export class Distribution {
  constructor(name = 'Distribution') {
    this.name = name;
    this.observed = null;
  }

  /**
   * Log probability density/mass function
   * @param {tf.Tensor|number} value - Value to evaluate
   * @returns {tf.Tensor} Log probability
   */
  logProb(value) {
    throw new Error('logProb must be implemented by subclass');
  }

  /**
   * Sample from the distribution
   * @param {number|Array<number>} shape - Shape of samples to generate
   * @returns {tf.Tensor} Samples
   */
  sample(shape = []) {
    throw new Error('sample must be implemented by subclass');
  }

  /**
   * Set observed data for this distribution
   * @param {tf.Tensor|number|Array} data - Observed data
   */
  observe(data) {
    this.observed = tf.tensor(data);
    return this;
  }

  /**
   * Get the shape of the distribution
   */
  get shape() {
    return this._shape || [];
  }
}

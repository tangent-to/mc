import * as tf from '@tensorflow/tfjs';

/**
 * Determine whether the first constructor argument is an options object
 * (e.g. `new Normal({ mu, sigma })`) rather than a positional value.
 *
 * Plain objects are treated as options; tensors, arrays and numbers are not.
 * This mirrors the dual positional/options constructor convention used across
 * the sibling `@tangent.to/ds` package.
 *
 * @param {*} value - First constructor argument
 * @returns {boolean} True if `value` should be interpreted as an options object
 */
export function isOptions(value) {
  return (
    value !== null &&
    typeof value === 'object' &&
    !Array.isArray(value) &&
    !(value instanceof tf.Tensor)
  );
}

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
   * Probability density/mass function
   *
   * Computed as `exp(logProb(value))`. Provided for parity with the
   * `@tangent.to/ds` distribution interface (`pdf`/`cdf`/`quantile`).
   *
   * @param {tf.Tensor|number} value - Value to evaluate
   * @returns {tf.Tensor} Probability density/mass
   */
  pdf(value) {
    return tf.tidy(() => tf.exp(this.logProb(value)));
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
   * Get the distribution's parameters as a plain object.
   * Subclasses override to expose their specific parameters.
   * @returns {Object} Parameters
   */
  getParams() {
    return {};
  }
}

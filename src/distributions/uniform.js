import * as tf from '@tensorflow/tfjs-node';
import { Distribution, isOptions } from './base.js';

/**
 * Uniform distribution
 */
export class Uniform extends Distribution {
  /**
   * Accepts either positional arguments or a single options object.
   *
   * @param {number|tf.Tensor|Object} lower - Lower bound, or an options object
   *   `{ lower | min, upper | max, name }`
   * @param {number|tf.Tensor} [upper] - Upper bound
   * @param {string} [name] - Name of the distribution
   *
   * @example
   * new Uniform(0, 1)
   * @example
   * new Uniform({ min: 0, max: 1 })
   */
  constructor(lower = 0, upper = 1, name = 'Uniform') {
    super(name);
    if (isOptions(lower)) {
      const o = lower;
      this.name = o.name ?? 'Uniform';
      lower = o.lower ?? o.min ?? 0;
      upper = o.upper ?? o.max ?? 1;
    }
    this.lower = typeof lower === 'number' ? tf.scalar(lower) : lower;
    this.upper = typeof upper === 'number' ? tf.scalar(upper) : upper;
  }

  /**
   * Log probability density function
   * @param {tf.Tensor|number} value - Value to evaluate
   * @returns {tf.Tensor} Log probability
   */
  logProb(value) {
    return tf.tidy(() => {
      const x = typeof value === 'number' ? tf.scalar(value) : value;
      const range = tf.sub(this.upper, this.lower);

      // Check if value is within bounds
      const inBounds = tf.logicalAnd(
        tf.greaterEqual(x, this.lower),
        tf.lessEqual(x, this.upper)
      );

      // log(p(x)) = -log(upper - lower) if lower <= x <= upper, else -Inf
      const logProb = tf.neg(tf.log(range));
      return tf.where(inBounds, logProb, tf.fill(x.shape, -Infinity));
    });
  }

  /**
   * Sample from the uniform distribution
   * @param {number|Array<number>} shape - Shape of samples to generate
   * @returns {tf.Tensor} Samples
   */
  sample(shape = []) {
    return tf.tidy(() => {
      const sampleShape = Array.isArray(shape) ? shape : [shape];
      const uniformSample = tf.randomUniform(sampleShape);
      const range = tf.sub(this.upper, this.lower);
      return tf.add(tf.mul(uniformSample, range), this.lower);
    });
  }

  /**
   * Get the mean of the distribution
   */
  mean() {
    return tf.div(tf.add(this.lower, this.upper), 2);
  }

  /**
   * Get the variance of the distribution
   */
  variance() {
    const range = tf.sub(this.upper, this.lower);
    return tf.div(tf.square(range), 12);
  }

  /**
   * Get the distribution's parameters.
   * @returns {{lower: number, upper: number}}
   */
  getParams() {
    return { lower: this.lower.arraySync(), upper: this.upper.arraySync() };
  }
}

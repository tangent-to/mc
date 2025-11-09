import * as tf from '@tensorflow/tfjs-node';
import { Distribution } from './base.js';
import jstat from 'jstat';

/**
 * Gamma distribution (useful for modeling positive continuous values)
 */
export class Gamma extends Distribution {
  /**
   * @param {number|tf.Tensor} alpha - Shape parameter (must be > 0)
   * @param {number|tf.Tensor} beta - Rate parameter (must be > 0)
   * @param {string} name - Name of the distribution
   */
  constructor(alpha = 1, beta = 1, name = 'Gamma') {
    super(name);
    this.alpha = typeof alpha === 'number' ? tf.scalar(alpha) : alpha;
    this.beta = typeof beta === 'number' ? tf.scalar(beta) : beta;
  }

  /**
   * Log probability density function
   * @param {tf.Tensor|number} value - Value to evaluate (must be > 0)
   * @returns {tf.Tensor} Log probability
   */
  logProb(value) {
    return tf.tidy(() => {
      const x = typeof value === 'number' ? tf.scalar(value) : value;

      // Gamma distribution is only defined for x > 0
      // Return -Infinity for x <= 0
      const xVal = x.arraySync();
      if (xVal <= 0) {
        return tf.scalar(-Infinity);
      }

      // log(p(x)) = α * log(β) + (α - 1) * log(x) - β * x - log(Γ(α))
      const alphaVal = this.alpha.arraySync();

      const logGamma = Math.log(jstat.gammafn(alphaVal));

      const logProb = tf.add(
        tf.add(
          tf.mul(this.alpha, tf.log(this.beta)),
          tf.mul(tf.sub(this.alpha, 1), tf.log(x))
        ),
        tf.sub(
          tf.mul(tf.neg(this.beta), x),
          logGamma
        )
      );

      return logProb;
    });
  }

  /**
   * Sample from the gamma distribution
   * @param {number|Array<number>} shape - Shape of samples to generate
   * @returns {tf.Tensor} Samples
   */
  sample(shape = []) {
    return tf.tidy(() => {
      const sampleShape = Array.isArray(shape) ? shape : [shape];
      const n = sampleShape.reduce((a, b) => a * b, 1) || 1;

      const alphaVal = this.alpha.arraySync();
      const betaVal = this.beta.arraySync();

      // Use jstat to sample from gamma distribution
      const samples = [];
      for (let i = 0; i < n; i++) {
        samples.push(jstat.gamma.sample(alphaVal, 1 / betaVal));
      }

      return tf.tensor(samples).reshape(sampleShape);
    });
  }

  /**
   * Get the mean of the distribution
   */
  mean() {
    return tf.div(this.alpha, this.beta);
  }

  /**
   * Get the variance of the distribution
   */
  variance() {
    return tf.div(this.alpha, tf.square(this.beta));
  }
}

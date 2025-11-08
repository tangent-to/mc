import * as tf from '@tensorflow/tfjs-node';
import { Distribution } from './base.js';
import jstat from 'jstat';

/**
 * Beta distribution (useful for modeling probabilities)
 */
export class Beta extends Distribution {
  /**
   * @param {number|tf.Tensor} alpha - Shape parameter (must be > 0)
   * @param {number|tf.Tensor} beta - Shape parameter (must be > 0)
   * @param {string} name - Name of the distribution
   */
  constructor(alpha = 1, beta = 1, name = 'Beta') {
    super(name);
    this.alpha = typeof alpha === 'number' ? tf.scalar(alpha) : alpha;
    this.beta = typeof beta === 'number' ? tf.scalar(beta) : beta;
  }

  /**
   * Log probability density function
   * @param {tf.Tensor|number} value - Value to evaluate (must be in [0, 1])
   * @returns {tf.Tensor} Log probability
   */
  logProb(value) {
    return tf.tidy(() => {
      const x = typeof value === 'number' ? tf.scalar(value) : value;

      // log(p(x)) = (α - 1) * log(x) + (β - 1) * log(1 - x) - log(B(α, β))
      // where B(α, β) is the beta function

      const logX = tf.log(x);
      const log1MinusX = tf.log(tf.sub(1, x));

      // Log beta function: log(B(α, β)) = log(Γ(α)) + log(Γ(β)) - log(Γ(α + β))
      const alphaVal = this.alpha.arraySync();
      const betaVal = this.beta.arraySync();
      const logBetaFunc = Math.log(jstat.betafn(alphaVal, betaVal));

      const logKernel = tf.add(
        tf.mul(tf.sub(this.alpha, 1), logX),
        tf.mul(tf.sub(this.beta, 1), log1MinusX)
      );

      return tf.sub(logKernel, logBetaFunc);
    });
  }

  /**
   * Sample from the beta distribution
   * Uses the relationship: if X ~ Gamma(α) and Y ~ Gamma(β), then X/(X+Y) ~ Beta(α, β)
   * @param {number|Array<number>} shape - Shape of samples to generate
   * @returns {tf.Tensor} Samples
   */
  sample(shape = []) {
    return tf.tidy(() => {
      const sampleShape = Array.isArray(shape) ? shape : [shape];
      const n = sampleShape.reduce((a, b) => a * b, 1) || 1;

      const alphaVal = this.alpha.arraySync();
      const betaVal = this.beta.arraySync();

      // Use jstat to sample from beta distribution
      const samples = [];
      for (let i = 0; i < n; i++) {
        samples.push(jstat.beta.sample(alphaVal, betaVal));
      }

      return tf.tensor(samples).reshape(sampleShape);
    });
  }

  /**
   * Get the mean of the distribution
   */
  mean() {
    return tf.div(this.alpha, tf.add(this.alpha, this.beta));
  }

  /**
   * Get the variance of the distribution
   */
  variance() {
    return tf.tidy(() => {
      const sum = tf.add(this.alpha, this.beta);
      const numerator = tf.mul(this.alpha, this.beta);
      const denominator = tf.mul(tf.square(sum), tf.add(sum, 1));
      return tf.div(numerator, denominator);
    });
  }
}

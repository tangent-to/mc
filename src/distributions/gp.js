import * as tf from '@tensorflow/tfjs-node';
import { Distribution } from './base.js';
import { RBF } from './kernels.js';

/**
 * Gaussian Process distribution
 * Represents a distribution over functions
 */
export class GaussianProcess extends Distribution {
  /**
   * @param {Function|tf.Tensor} mean - Mean function or constant mean
   * @param {Object} kernel - Kernel function (e.g., RBF, Matern32)
   * @param {number} noiseVariance - Observation noise variance (σ²_noise)
   * @param {string} name - Name of the distribution
   */
  constructor(mean = 0, kernel = null, noiseVariance = 0.01, name = 'GaussianProcess') {
    super(name);
    this.meanFunction = typeof mean === 'function' ? mean : (X) => tf.fill([X.shape[0]], mean);
    this.kernel = kernel || new RBF(1.0, 1.0);
    this.noiseVariance = noiseVariance;

    // Training data (set via fit method)
    this.X_train = null;
    this.y_train = null;
    this.K_train = null;
    this.K_train_inv = null;
    this.alpha = null; // K⁻¹(y - μ)
  }

  /**
   * Fit the GP to training data (condition on observations)
   * @param {Array|tf.Tensor} X - Training inputs [n, d]
   * @param {Array|tf.Tensor} y - Training outputs [n]
   */
  fit(X, y) {
    return tf.tidy(() => {
      this.X_train = Array.isArray(X) ? tf.tensor2d(X.map(x => Array.isArray(x) ? x : [x])) : X;
      this.y_train = Array.isArray(y) ? tf.tensor1d(y) : y;

      const n = this.X_train.shape[0];

      // Compute kernel matrix K(X, X) + σ²I
      this.K_train = this.kernel.compute(this.X_train);
      const K_with_noise = tf.add(
        this.K_train,
        tf.mul(this.noiseVariance, tf.eye(n))
      );

      // Compute Cholesky decomposition: K = LLᵀ
      // For numerical stability, we use TF's cholesky
      const L = tf.linalg.bandPart(tf.linalg.cholesky(K_with_noise), -1, 0);

      // Compute α = K⁻¹(y - μ)
      const mean_train = this.meanFunction(this.X_train);
      const y_centered = tf.sub(this.y_train, mean_train);

      // Solve Lα' = y_centered
      const alpha_prime = tf.linalg.triangularSolve(L, tf.expandDims(y_centered, 1), true, false);
      // Solve Lᵀα = α'
      this.alpha = tf.squeeze(tf.linalg.triangularSolve(L, alpha_prime, false, true));

      // Keep L for sampling
      this.L_train = L;

      return this;
    });
  }

  /**
   * Predict at new points (posterior mean and variance)
   * @param {Array|tf.Tensor} X_new - Test inputs [n_new, d]
   * @param {boolean} returnStd - Whether to return standard deviation
   * @returns {Object} {mean, std} or just mean
   */
  predict(X_new, returnStd = true) {
    if (this.X_train === null) {
      throw new Error('GP must be fit to data before prediction');
    }

    return tf.tidy(() => {
      const X_test = Array.isArray(X_new) ? tf.tensor2d(X_new.map(x => Array.isArray(x) ? x : [x])) : X_new;

      // Compute k(X_new, X_train)
      const K_star = this.kernel.compute(X_test, this.X_train);

      // Posterior mean: μ(X_new) = μ₀(X_new) + K_star @ α
      const mean_prior = this.meanFunction(X_test);
      const mean = tf.add(
        mean_prior,
        tf.squeeze(tf.matMul(K_star, tf.expandDims(this.alpha, 1)))
      );

      if (!returnStd) {
        return { mean: mean.arraySync() };
      }

      // Posterior variance: K(X_new, X_new) - K_star @ K⁻¹ @ K_starᵀ
      const K_star_star = this.kernel.compute(X_test);

      // Solve L @ V = K_starᵀ
      const V = tf.linalg.triangularSolve(
        this.L_train,
        tf.transpose(K_star),
        true,
        false
      );

      // Variance: diag(K_** - VᵀV)
      const variance = tf.sub(
        tf.diag(K_star_star),
        tf.sum(tf.square(V), 0)
      );

      const std = tf.sqrt(tf.maximum(variance, 1e-10)); // avoid negative variance due to numerical error

      return {
        mean: mean.arraySync(),
        std: std.arraySync()
      };
    });
  }

  /**
   * Sample functions from the GP posterior
   * @param {Array|tf.Tensor} X_new - Test inputs [n_new, d]
   * @param {number} nSamples - Number of function samples
   * @returns {Array} Array of samples, each [n_new]
   */
  samplePosterior(X_new, nSamples = 1) {
    if (this.X_train === null) {
      throw new Error('GP must be fit to data before sampling');
    }

    return tf.tidy(() => {
      const X_test = Array.isArray(X_new) ? tf.tensor2d(X_new.map(x => Array.isArray(x) ? x : [x])) : X_new;

      // Get posterior mean and covariance
      const K_star = this.kernel.compute(X_test, this.X_train);
      const K_star_star = this.kernel.compute(X_test);

      const mean_prior = this.meanFunction(X_test);
      const mean = tf.add(
        mean_prior,
        tf.squeeze(tf.matMul(K_star, tf.expandDims(this.alpha, 1)))
      );

      // Posterior covariance
      const V = tf.linalg.triangularSolve(
        this.L_train,
        tf.transpose(K_star),
        true,
        false
      );
      const cov = tf.sub(K_star_star, tf.matMul(V, V, true, false));

      // Add small jitter for numerical stability
      const n_test = X_test.shape[0];
      const cov_with_jitter = tf.add(cov, tf.mul(1e-6, tf.eye(n_test)));

      // Cholesky decomposition of covariance
      const L_post = tf.linalg.bandPart(tf.linalg.cholesky(cov_with_jitter), -1, 0);

      // Sample: f = μ + L @ z, where z ~ N(0, I)
      const samples = [];
      for (let i = 0; i < nSamples; i++) {
        const z = tf.randomNormal([n_test, 1]);
        const sample = tf.add(
          mean,
          tf.squeeze(tf.matMul(L_post, z))
        );
        samples.push(sample.arraySync());
      }

      return samples;
    });
  }

  /**
   * Log marginal likelihood (for model selection/hyperparameter optimization)
   * log p(y | X) = -½(yᵀK⁻¹y) - ½log|K| - (n/2)log(2π)
   * @returns {number} Log marginal likelihood
   */
  logMarginalLikelihood() {
    if (this.X_train === null) {
      throw new Error('GP must be fit to data before computing likelihood');
    }

    return tf.tidy(() => {
      const n = this.X_train.shape[0];

      // Compute yᵀK⁻¹y = yᵀα
      const mean_train = this.meanFunction(this.X_train);
      const y_centered = tf.sub(this.y_train, mean_train);
      const fit_term = tf.sum(tf.mul(y_centered, this.alpha));

      // Compute log|K| = 2 * Σ log(diag(L))
      const log_det = tf.mul(2, tf.sum(tf.log(tf.diag(this.L_train))));

      // Constant term
      const const_term = n * Math.log(2 * Math.PI);

      const log_likelihood = tf.mul(-0.5, tf.add(tf.add(fit_term, log_det), const_term));

      return log_likelihood.arraySync();
    });
  }

  /**
   * Log probability density (for MCMC integration)
   * @param {tf.Tensor|Array} value - Function values at training points
   * @returns {tf.Tensor} Log probability
   */
  logProb(value) {
    if (this.X_train === null) {
      throw new Error('GP must be fit to training data before computing logProb');
    }

    return tf.tidy(() => {
      const y = Array.isArray(value) ? tf.tensor1d(value) : value;
      const n = this.X_train.shape[0];

      // Build covariance matrix
      const K = tf.add(
        this.kernel.compute(this.X_train),
        tf.mul(this.noiseVariance, tf.eye(n))
      );

      const L = tf.linalg.bandPart(tf.linalg.cholesky(K), -1, 0);

      // Center y
      const mean_train = this.meanFunction(this.X_train);
      const y_centered = tf.sub(y, mean_train);

      // Compute α = K⁻¹y
      const alpha_prime = tf.linalg.triangularSolve(L, tf.expandDims(y_centered, 1), true, false);
      const alpha = tf.squeeze(tf.linalg.triangularSolve(L, alpha_prime, false, true));

      // Log likelihood
      const fit_term = tf.sum(tf.mul(y_centered, alpha));
      const log_det = tf.mul(2, tf.sum(tf.log(tf.diag(L))));
      const const_term = n * Math.log(2 * Math.PI);

      return tf.mul(-0.5, tf.add(tf.add(fit_term, log_det), const_term));
    });
  }

  /**
   * Sample from the GP prior
   * @param {Array|tf.Tensor} X - Input locations [n, d]
   * @param {number} nSamples - Number of function samples
   * @returns {Array} Array of samples
   */
  sample(X, nSamples = 1) {
    return tf.tidy(() => {
      const X_tensor = Array.isArray(X) ? tf.tensor2d(X.map(x => Array.isArray(x) ? x : [x])) : X;
      const n = X_tensor.shape[0];

      // Compute covariance matrix
      const K = this.kernel.compute(X_tensor);
      const K_with_jitter = tf.add(K, tf.mul(1e-6, tf.eye(n)));

      // Cholesky decomposition
      const L = tf.linalg.bandPart(tf.linalg.cholesky(K_with_jitter), -1, 0);

      // Mean
      const mean = this.meanFunction(X_tensor);

      // Sample: f = μ + L @ z
      const samples = [];
      for (let i = 0; i < nSamples; i++) {
        const z = tf.randomNormal([n, 1]);
        const sample = tf.add(
          mean,
          tf.squeeze(tf.matMul(L, z))
        );
        samples.push(sample.arraySync());
      }

      return samples;
    });
  }
}

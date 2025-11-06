import * as tf from '@tensorflow/tfjs-node';
import { Matrix, CholeskyDecomposition } from 'ml-matrix';
import { Distribution } from './base.js';
import { RBF } from './kernels.js';

/**
 * Gaussian Process distribution
 * Represents a distribution over functions
 *
 * Note: This implementation uses ml-matrix for linear algebra operations
 * that are not available in TensorFlow.js (matrix inversion, Cholesky).
 * TensorFlow.js is used for kernel computations and automatic differentiation.
 */
export class GaussianProcess extends Distribution {
  /**
   * @param {Function|number} mean - Mean function or constant mean
   * @param {Object} kernel - Kernel function (e.g., RBF, Matern32)
   * @param {number} noiseVariance - Observation noise variance (σ²_noise)
   * @param {string} name - Name of the distribution
   */
  constructor(mean = 0, kernel = null, noiseVariance = 0.01, name = 'GaussianProcess') {
    super(name);
    this.meanValue = mean;
    this.kernel = kernel || new RBF(1.0, 1.0);
    this.noiseVariance = noiseVariance;

    // Training data
    this.X_train = null;
    this.y_train = null;
    this.alpha = null;  // K⁻¹(y - μ)
    this.L = null;      // Cholesky factor
  }

  /**
   * Fit the GP to training data (condition on observations)
   * @param {Array|tf.Tensor} X - Training inputs [n, d]
   * @param {Array|tf.Tensor} y - Training outputs [n]
   */
  fit(X, y) {
    // Convert to arrays if needed
    const X_array = Array.isArray(X) ? X : X.arraySync();
    const y_array = Array.isArray(y) ? y : y.arraySync();

    this.X_train = X_array;
    this.y_train = y_array;

    const n = X_array.length;

    // Compute kernel matrix using TensorFlow
    const X_tensor = tf.tensor2d(X_array.map(x => Array.isArray(x) ? x : [x]));
    const K_tf = this.kernel.compute(X_tensor);
    const K_array = K_tf.arraySync();
    K_tf.dispose();
    X_tensor.dispose();

    // Convert to ml-matrix and add noise
    const K = new Matrix(K_array);
    for (let i = 0; i < n; i++) {
      K.set(i, i, K.get(i, i) + this.noiseVariance);
    }

    // Cholesky decomposition
    try {
      const chol = new CholeskyDecomposition(K);
      this.L = chol.lowerTriangularMatrix;
    } catch (e) {
      // If Cholesky fails, add more jitter
      console.warn('Cholesky decomposition failed, adding jitter');
      for (let i = 0; i < n; i++) {
        K.set(i, i, K.get(i, i) + 1e-6);
      }
      const chol = new CholeskyDecomposition(K);
      this.L = chol.lowerTriangularMatrix;
    }

    // Compute α = K⁻¹(y - μ)
    const y_centered = y_array.map(yi => yi - this.meanValue);

    // Solve L * L^T * α = y_centered
    // First solve L * v = y_centered
    const v = this._forwardSubstitution(this.L, y_centered);
    // Then solve L^T * α = v
    this.alpha = this._backwardSubstitution(this.L.transpose(), v);

    return this;
  }

  /**
   * Forward substitution for lower triangular matrix
   */
  _forwardSubstitution(L, b) {
    const n = b.length;
    const x = new Array(n);

    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let j = 0; j < i; j++) {
        sum += L.get(i, j) * x[j];
      }
      x[i] = (b[i] - sum) / L.get(i, i);
    }

    return x;
  }

  /**
   * Backward substitution for upper triangular matrix
   */
  _backwardSubstitution(U, b) {
    const n = b.length;
    const x = new Array(n);

    for (let i = n - 1; i >= 0; i--) {
      let sum = 0;
      for (let j = i + 1; j < n; j++) {
        sum += U.get(i, j) * x[j];
      }
      x[i] = (b[i] - sum) / U.get(i, i);
    }

    return x;
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

    const X_test_array = Array.isArray(X_new) ? X_new : X_new.arraySync();

    // Compute k(X_new, X_train) using TensorFlow
    const X_train_tensor = tf.tensor2d(this.X_train.map(x => Array.isArray(x) ? x : [x]));
    const X_test_tensor = tf.tensor2d(X_test_array.map(x => Array.isArray(x) ? x : [x]));

    const K_star_tf = this.kernel.compute(X_test_tensor, X_train_tensor);
    const K_star_array = K_star_tf.arraySync();

    // Posterior mean: μ(X_new) = μ₀ + K_star @ α
    const mean = [];
    for (let i = 0; i < X_test_array.length; i++) {
      let sum = this.meanValue;
      for (let j = 0; j < this.alpha.length; j++) {
        sum += K_star_array[i][j] * this.alpha[j];
      }
      mean.push(sum);
    }

    if (!returnStd) {
      K_star_tf.dispose();
      X_train_tensor.dispose();
      X_test_tensor.dispose();
      return { mean };
    }

    // Posterior variance: K(X_new, X_new) - K_star @ K⁻¹ @ K_starᵀ
    const K_star_star_tf = this.kernel.compute(X_test_tensor);
    const K_star_star_array = K_star_star_tf.arraySync();

    K_star_tf.dispose();
    K_star_star_tf.dispose();
    X_train_tensor.dispose();
    X_test_tensor.dispose();

    const std = [];
    for (let i = 0; i < X_test_array.length; i++) {
      // Solve L * v = K_star[i, :]
      const v = this._forwardSubstitution(this.L, K_star_array[i]);

      // Variance = K_star_star[i, i] - ||v||²
      let v_squared_sum = 0;
      for (let j = 0; j < v.length; j++) {
        v_squared_sum += v[j] * v[j];
      }

      const variance = Math.max(K_star_star_array[i][i] - v_squared_sum, 1e-10);
      std.push(Math.sqrt(variance));
    }

    return { mean, std };
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

    const X_test_array = Array.isArray(X_new) ? X_new : X_new.arraySync();

    // Get posterior mean and covariance
    const X_train_tensor = tf.tensor2d(this.X_train.map(x => Array.isArray(x) ? x : [x]));
    const X_test_tensor = tf.tensor2d(X_test_array.map(x => Array.isArray(x) ? x : [x]));

    const K_star_tf = this.kernel.compute(X_test_tensor, X_train_tensor);
    const K_star_star_tf = this.kernel.compute(X_test_tensor);

    const K_star_array = K_star_tf.arraySync();
    const K_star_star_array = K_star_star_tf.arraySync();

    K_star_tf.dispose();
    K_star_star_tf.dispose();
    X_train_tensor.dispose();
    X_test_tensor.dispose();

    const n_test = X_test_array.length;

    // Compute posterior mean
    const mean = [];
    for (let i = 0; i < n_test; i++) {
      let sum = this.meanValue;
      for (let j = 0; j < this.alpha.length; j++) {
        sum += K_star_array[i][j] * this.alpha[j];
      }
      mean.push(sum);
    }

    // Compute posterior covariance
    const cov = new Matrix(K_star_star_array);
    for (let i = 0; i < n_test; i++) {
      const v = this._forwardSubstitution(this.L, K_star_array[i]);
      const v_squared_sum = v.reduce((sum, val) => sum + val * val, 0);
      cov.set(i, i, cov.get(i, i) - v_squared_sum + 1e-6); // Add jitter
    }

    // Cholesky of posterior covariance
    let L_post;
    try {
      const chol = new CholeskyDecomposition(cov);
      L_post = chol.lowerTriangularMatrix;
    } catch (e) {
      // If fails, add more jitter
      for (let i = 0; i < n_test; i++) {
        cov.set(i, i, cov.get(i, i) + 1e-5);
      }
      const chol = new CholeskyDecomposition(cov);
      L_post = chol.lowerTriangularMatrix;
    }

    // Sample: f = μ + L @ z
    const samples = [];
    for (let s = 0; s < nSamples; s++) {
      const z = [];
      for (let i = 0; i < n_test; i++) {
        // Sample from standard normal
        const u1 = Math.random();
        const u2 = Math.random();
        z.push(Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2));
      }

      const sample = [];
      for (let i = 0; i < n_test; i++) {
        let sum = mean[i];
        for (let j = 0; j < n_test; j++) {
          sum += L_post.get(i, j) * z[j];
        }
        sample.push(sum);
      }
      samples.push(sample);
    }

    return samples;
  }

  /**
   * Log marginal likelihood
   * @returns {number} Log marginal likelihood
   */
  logMarginalLikelihood() {
    if (this.X_train === null) {
      throw new Error('GP must be fit to data before computing likelihood');
    }

    const n = this.y_train.length;

    // Compute yᵀK⁻¹y = yᵀα = (y - μ)ᵀα
    const y_centered = this.y_train.map(yi => yi - this.meanValue);
    let fit_term = 0;
    for (let i = 0; i < n; i++) {
      fit_term += y_centered[i] * this.alpha[i];
    }

    // Compute log|K| = 2 * Σ log(diag(L))
    let log_det = 0;
    for (let i = 0; i < n; i++) {
      log_det += 2 * Math.log(this.L.get(i, i));
    }

    const const_term = n * Math.log(2 * Math.PI);

    return -0.5 * (fit_term + log_det + const_term);
  }

  /**
   * Log probability density (for MCMC)
   * @param {Array} value - Function values
   * @returns {number} Log probability
   */
  logProb(value) {
    // For MCMC, just return log marginal likelihood
    // This is a simplification; proper implementation would refit
    return this.logMarginalLikelihood();
  }

  /**
   * Sample from the GP prior
   * @param {Array} X - Input locations
   * @param {number} nSamples - Number of samples
   * @returns {Array} Array of samples
   */
  sample(X, nSamples = 1) {
    const X_array = Array.isArray(X) ? X : X.arraySync();
    const n = X_array.length;

    // Compute covariance matrix
    const X_tensor = tf.tensor2d(X_array.map(x => Array.isArray(x) ? x : [x]));
    const K_tf = this.kernel.compute(X_tensor);
    const K_array = K_tf.arraySync();
    K_tf.dispose();
    X_tensor.dispose();

    const K = Matrix.from(K_array);
    for (let i = 0; i < n; i++) {
      K.set(i, i, K.get(i, i) + 1e-6); // Add jitter
    }

    // Cholesky decomposition
    const chol = new CholeskyDecomposition(K);
    const L = chol.lowerTriangularMatrix;

    // Sample: f = μ + L @ z
    const samples = [];
    for (let s = 0; s < nSamples; s++) {
      const z = [];
      for (let i = 0; i < n; i++) {
        const u1 = Math.random();
        const u2 = Math.random();
        z.push(Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2));
      }

      const sample = [];
      for (let i = 0; i < n; i++) {
        let sum = this.meanValue;
        for (let j = 0; j < n; j++) {
          sum += L.get(i, j) * z[j];
        }
        sample.push(sum);
      }
      samples.push(sample);
    }

    return samples;
  }
}

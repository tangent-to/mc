import * as tf from '@tensorflow/tfjs-node';

/**
 * Kernel functions for Gaussian Processes
 */

/**
 * Radial Basis Function (RBF) / Squared Exponential kernel
 * k(x, x') = σ² * exp(-||x - x'||² / (2 * l²))
 */
export class RBF {
  /**
   * @param {number} lengthscale - Length scale parameter (l)
   * @param {number} variance - Variance parameter (σ²)
   */
  constructor(lengthscale = 1.0, variance = 1.0) {
    this.lengthscale = lengthscale;
    this.variance = variance;
  }

  /**
   * Compute the kernel matrix K[i,j] = k(X[i], X[j])
   * @param {tf.Tensor} X1 - First set of points [n1, d]
   * @param {tf.Tensor} X2 - Second set of points [n2, d] (optional, defaults to X1)
   * @returns {tf.Tensor} Kernel matrix [n1, n2]
   */
  compute(X1, X2 = null) {
    return tf.tidy(() => {
      const X2_actual = X2 === null ? X1 : X2;

      // Compute squared distances: ||x - x'||²
      // Using: ||x - x'||² = ||x||² + ||x'||² - 2*x·x'
      const X1_sqnorms = tf.sum(tf.square(X1), 1, true); // [n1, 1]
      const X2_sqnorms = tf.sum(tf.square(X2_actual), 1, true); // [n2, 1]
      const cross = tf.matMul(X1, X2_actual, false, true); // [n1, n2]

      const sq_dists = tf.add(
        X1_sqnorms,
        tf.sub(tf.transpose(X2_sqnorms), tf.mul(2, cross))
      );

      // k(x, x') = σ² * exp(-sq_dist / (2 * l²))
      const K = tf.mul(
        this.variance,
        tf.exp(tf.div(tf.neg(sq_dists), 2 * this.lengthscale ** 2))
      );

      return K;
    });
  }
}

/**
 * Matérn 3/2 kernel
 * k(x, x') = σ² * (1 + √3*r/l) * exp(-√3*r/l)
 * where r = ||x - x'||
 */
export class Matern32 {
  /**
   * @param {number} lengthscale - Length scale parameter (l)
   * @param {number} variance - Variance parameter (σ²)
   */
  constructor(lengthscale = 1.0, variance = 1.0) {
    this.lengthscale = lengthscale;
    this.variance = variance;
  }

  /**
   * Compute the kernel matrix
   * @param {tf.Tensor} X1 - First set of points [n1, d]
   * @param {tf.Tensor} X2 - Second set of points [n2, d] (optional)
   * @returns {tf.Tensor} Kernel matrix [n1, n2]
   */
  compute(X1, X2 = null) {
    return tf.tidy(() => {
      const X2_actual = X2 === null ? X1 : X2;

      // Compute distances
      const X1_sqnorms = tf.sum(tf.square(X1), 1, true);
      const X2_sqnorms = tf.sum(tf.square(X2_actual), 1, true);
      const cross = tf.matMul(X1, X2_actual, false, true);

      const sq_dists = tf.add(
        X1_sqnorms,
        tf.sub(tf.transpose(X2_sqnorms), tf.mul(2, cross))
      );
      const dists = tf.sqrt(tf.maximum(sq_dists, 1e-10)); // avoid sqrt(0)

      // Matérn 3/2: σ² * (1 + √3*r/l) * exp(-√3*r/l)
      const sqrt3_r_over_l = tf.div(tf.mul(Math.sqrt(3), dists), this.lengthscale);

      const K = tf.mul(
        this.variance,
        tf.mul(
          tf.add(1, sqrt3_r_over_l),
          tf.exp(tf.neg(sqrt3_r_over_l))
        )
      );

      return K;
    });
  }
}

/**
 * Matérn 5/2 kernel
 * k(x, x') = σ² * (1 + √5*r/l + 5*r²/(3*l²)) * exp(-√5*r/l)
 */
export class Matern52 {
  /**
   * @param {number} lengthscale - Length scale parameter (l)
   * @param {number} variance - Variance parameter (σ²)
   */
  constructor(lengthscale = 1.0, variance = 1.0) {
    this.lengthscale = lengthscale;
    this.variance = variance;
  }

  /**
   * Compute the kernel matrix
   * @param {tf.Tensor} X1 - First set of points [n1, d]
   * @param {tf.Tensor} X2 - Second set of points [n2, d] (optional)
   * @returns {tf.Tensor} Kernel matrix [n1, n2]
   */
  compute(X1, X2 = null) {
    return tf.tidy(() => {
      const X2_actual = X2 === null ? X1 : X2;

      // Compute distances
      const X1_sqnorms = tf.sum(tf.square(X1), 1, true);
      const X2_sqnorms = tf.sum(tf.square(X2_actual), 1, true);
      const cross = tf.matMul(X1, X2_actual, false, true);

      const sq_dists = tf.add(
        X1_sqnorms,
        tf.sub(tf.transpose(X2_sqnorms), tf.mul(2, cross))
      );
      const dists = tf.sqrt(tf.maximum(sq_dists, 1e-10));

      // Matérn 5/2
      const sqrt5_r_over_l = tf.div(tf.mul(Math.sqrt(5), dists), this.lengthscale);
      const five_rsq_over_3lsq = tf.div(
        tf.mul(5, sq_dists),
        3 * this.lengthscale ** 2
      );

      const K = tf.mul(
        this.variance,
        tf.mul(
          tf.add(tf.add(1, sqrt5_r_over_l), five_rsq_over_3lsq),
          tf.exp(tf.neg(sqrt5_r_over_l))
        )
      );

      return K;
    });
  }
}

/**
 * Periodic kernel
 * k(x, x') = σ² * exp(-2 * sin²(π*|x - x'|/p) / l²)
 */
export class Periodic {
  /**
   * @param {number} period - Period parameter (p)
   * @param {number} lengthscale - Length scale parameter (l)
   * @param {number} variance - Variance parameter (σ²)
   */
  constructor(period = 1.0, lengthscale = 1.0, variance = 1.0) {
    this.period = period;
    this.lengthscale = lengthscale;
    this.variance = variance;
  }

  /**
   * Compute the kernel matrix
   * @param {tf.Tensor} X1 - First set of points [n1, 1]
   * @param {tf.Tensor} X2 - Second set of points [n2, 1] (optional)
   * @returns {tf.Tensor} Kernel matrix [n1, n2]
   */
  compute(X1, X2 = null) {
    return tf.tidy(() => {
      const X2_actual = X2 === null ? X1 : X2;

      // Compute distances
      const X1_expanded = tf.expandDims(X1, 1); // [n1, 1, d]
      const X2_expanded = tf.expandDims(X2_actual, 0); // [1, n2, d]
      const diffs = tf.sub(X1_expanded, X2_expanded); // [n1, n2, d]
      const dists = tf.norm(diffs, 2, 2); // [n1, n2]

      // Periodic kernel: exp(-2 * sin²(π*r/p) / l²)
      const sin_term = tf.sin(tf.div(tf.mul(Math.PI, dists), this.period));
      const K = tf.mul(
        this.variance,
        tf.exp(
          tf.div(
            tf.mul(-2, tf.square(sin_term)),
            this.lengthscale ** 2
          )
        )
      );

      return K;
    });
  }
}

/**
 * Linear kernel
 * k(x, x') = σ² * (x - c)ᵀ(x' - c)
 */
export class Linear {
  /**
   * @param {number} variance - Variance parameter (σ²)
   * @param {number} offset - Offset parameter (c)
   */
  constructor(variance = 1.0, offset = 0.0) {
    this.variance = variance;
    this.offset = offset;
  }

  /**
   * Compute the kernel matrix
   * @param {tf.Tensor} X1 - First set of points [n1, d]
   * @param {tf.Tensor} X2 - Second set of points [n2, d] (optional)
   * @returns {tf.Tensor} Kernel matrix [n1, n2]
   */
  compute(X1, X2 = null) {
    return tf.tidy(() => {
      const X2_actual = X2 === null ? X1 : X2;

      const X1_centered = tf.sub(X1, this.offset);
      const X2_centered = tf.sub(X2_actual, this.offset);

      const K = tf.mul(
        this.variance,
        tf.matMul(X1_centered, X2_centered, false, true)
      );

      return K;
    });
  }
}

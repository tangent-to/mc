import * as tf from '@tensorflow/tfjs-node';

/**
 * Model class for defining Bayesian probabilistic models
 *
 * Similar to PyMC's Model context manager, this class represents a probabilistic model
 * as a Directed Acyclic Graph (DAG) of random variables.
 *
 * **Joint probability**:
 * $$
 * p(\theta, y) = p(y|\theta)p(\theta)
 * $$
 * where $\theta$ are parameters (latent variables) and $y$ is observed data.
 *
 * **Posterior** (via Bayes' theorem):
 * $$
 * p(\theta|y) = \frac{p(y|\theta)p(\theta)}{p(y)} \propto p(y|\theta)p(\theta)
 * $$
 *
 * @see {@link https://www.pymc.io/|PyMC Documentation}
 */
export class Model {
  /**
   * Accepts either a positional name or a single options object `{ name }`.
   *
   * @param {string|Object} name - Model name, or an options object `{ name }`
   *
   * @example
   * new Model('linear_regression')
   * @example
   * new Model({ name: 'linear_regression' })
   */
  constructor(name = 'model') {
    if (name !== null && typeof name === 'object' && !Array.isArray(name)) {
      name = name.name ?? 'model';
    }
    this.name = name;
    this.variables = new Map(); // Random variables in the model
    this.observedVars = new Map(); // Observed data
    this.potentials = new Map(); // Generic log-density terms (factors / likelihoods)
    this.deterministics = new Map(); // Named transforms recorded in the trace
    this.logProbFn = null; // Compiled log probability function
  }

  /**
   * Register a generic log-density term (a "potential" / factor) contributing to
   * the joint log-probability. `fn(params)` receives the current free-variable
   * values as tf tensors keyed by name and must return a tf.Tensor of
   * log-density values (which are summed into the total).
   *
   * This is the general mechanism for likelihoods whose parameters are arbitrary
   * deterministic functions of the latent variables and data — the deterministic
   * expression is computed inside `fn`, so it is not specific to any one model:
   *
   * ```js
   * model.potential('y', (v) =>
   *   new Normal(tf.add(tf.mul(v.slope, xData), v.intercept), v.sigma).logProb(yData));
   * ```
   *
   * @param {string} name - Identifier for the term
   * @param {(params: Object) => tf.Tensor} fn - Returns a log-density tensor
   * @returns {Model} this
   */
  potential(name, fn) {
    this.potentials.set(name, fn);
    return this;
  }

  /**
   * Register a named deterministic transform of the parameters for recording in
   * the trace (computed post-hoc from posterior draws). Deterministics do NOT
   * affect the log-probability — use {@link Model#potential} for likelihood or
   * factor terms.
   *
   * @param {string} name - Identifier for the transform
   * @param {(params: Object) => (tf.Tensor|number|Array)} fn - The transform
   * @returns {Model} this
   */
  deterministic(name, fn) {
    this.deterministics.set(name, fn);
    return this;
  }

  /**
   * Add a random variable to the model
   * @param {string} name - Name of the variable
   * @param {Distribution} distribution - Distribution of the variable
   * @param {*} observed - Observed data (optional)
   * @returns {Distribution} The distribution
   */
  addVariable(name, distribution, observed = null) {
    this.variables.set(name, distribution);

    if (observed !== null) {
      distribution.observe(observed);
      this.observedVars.set(name, observed);
    }

    return distribution;
  }

  /**
   * Get a variable from the model
   * @param {string} name - Name of the variable
   * @returns {Distribution} The distribution
   */
  getVariable(name) {
    return this.variables.get(name);
  }

  /**
   * Compute the log probability of the model given parameter values
   * @param {Object} params - Parameter values as {name: value} pairs
   * @returns {tf.Tensor} Log probability (scalar)
   */
  logProb(params) {
    return tf.tidy(() => {
      let logProb = tf.scalar(0);

      // Compute log probability for each variable
      for (const [name, distribution] of this.variables.entries()) {
        const value = params[name];

        if (value !== undefined) {
          const varLogProb = distribution.logProb(value);
          logProb = tf.add(logProb, tf.sum(varLogProb));
        } else if (distribution.observed !== null) {
          // For observed variables, compute log likelihood
          const varLogProb = distribution.logProb(distribution.observed);
          logProb = tf.add(logProb, tf.sum(varLogProb));
        }
      }

      // Generic potential / likelihood terms (deterministic-mean factors).
      for (const fn of this.potentials.values()) {
        logProb = tf.add(logProb, tf.sum(fn(params)));
      }

      return logProb;
    });
  }

  /**
   * Compute the log probability and its gradient with respect to parameters
   * @param {Object} params - Parameter values as {name: tf.Tensor} pairs
   * @returns {Object} {logProb: number, gradients: Object}
   */
  logProbAndGradient(params) {
    const paramNames = Object.keys(params);

    // Inputs as tensors (track the ones we create so we can free them).
    const created = [];
    const inputs = paramNames.map((name) => {
      const v = params[name];
      if (v instanceof tf.Tensor) return v;
      const t = tf.tensor(v);
      created.push(t);
      return t;
    });

    // tf.valueAndGrads differentiates w.r.t. the positional inputs and returns
    // gradients in the SAME order — robust regardless of variable naming.
    const f = (...args) => {
      const dict = {};
      paramNames.forEach((name, i) => { dict[name] = args[i]; });
      return this.logProb(dict);
    };
    const { value, grads } = tf.valueAndGrads(f)(inputs);

    const gradients = {};
    paramNames.forEach((name, i) => { gradients[name] = grads[i]; });

    const logProbValue = value.arraySync();
    value.dispose();
    created.forEach((t) => t.dispose());

    return { logProb: logProbValue, gradients };
  }

  /**
   * Sample from the prior distributions
   * @param {number} nSamples - Number of samples to generate
   * @returns {Object} Samples as {name: Array} pairs
   */
  samplePrior(nSamples = 1) {
    const samples = {};

    for (const [name, distribution] of this.variables.entries()) {
      if (distribution.observed === null) {
        const sample = distribution.sample([nSamples]);
        samples[name] = sample.arraySync();
        sample.dispose();
      }
    }

    return samples;
  }

  /**
   * Get list of unobserved variable names
   * @returns {Array<string>} Variable names
   */
  getFreeVariableNames() {
    const names = [];
    for (const [name, distribution] of this.variables.entries()) {
      if (distribution.observed === null) {
        names.push(name);
      }
    }
    return names;
  }

  /**
   * Posterior predictive sampling
   * Generate predictions by sampling from the posterior
   * @param {Object} trace - Trace object from MCMC sampling
   * @param {Function} predictFn - Function that takes params and returns predictions
   * @param {number} nSamples - Number of posterior samples to use (null = use all)
   * @returns {Array} Array of predictions from each posterior sample
   */
  predictPosterior(trace, predictFn, nSamples = null) {
    const traceData = trace.trace || trace;
    const nTraces = traceData[Object.keys(traceData)[0]].length;
    const nToUse = nSamples === null ? nTraces : Math.min(nSamples, nTraces);

    const predictions = [];

    for (let i = 0; i < nToUse; i++) {
      // Extract parameters for this sample
      const params = {};
      for (const [name, samples] of Object.entries(traceData)) {
        params[name] = samples[i];
      }

      // Generate prediction
      const pred = predictFn(params);
      predictions.push(pred);
    }

    return predictions;
  }

  /**
   * Compute posterior predictive mean and credible intervals
   * @param {Object} trace - Trace object from MCMC sampling
   * @param {Function} predictFn - Function that takes params and returns predictions
   * @param {number} credibleInterval - Credible interval (e.g., 0.95 for 95%)
   * @returns {Object} {mean, lower, upper} predictions
   */
  predictPosteriorSummary(trace, predictFn, credibleInterval = 0.95) {
    const predictions = this.predictPosterior(trace, predictFn);

    // Assume predictions are arrays of numbers or single numbers
    const isArray = Array.isArray(predictions[0]);

    if (!isArray) {
      // Single value predictions
      const sorted = [...predictions].sort((a, b) => a - b);
      const n = sorted.length;
      const lowerPercentile = (1 - credibleInterval) / 2;
      const upperPercentile = 1 - lowerPercentile;
      const lowerIdx = Math.floor(n * lowerPercentile);
      const upperIdx = Math.min(Math.floor(n * upperPercentile), n - 1);

      return {
        mean: predictions.reduce((a, b) => a + b, 0) / n,
        lower: sorted[lowerIdx],
        upper: sorted[upperIdx]
      };
    } else {
      // Array predictions - compute element-wise statistics
      const nPoints = predictions[0].length;
      const mean = new Array(nPoints).fill(0);
      const lower = new Array(nPoints);
      const upper = new Array(nPoints);

      for (let i = 0; i < nPoints; i++) {
        const values = predictions.map(p => p[i]);
        const sorted = [...values].sort((a, b) => a - b);
        const n = sorted.length;
        const lowerPercentile = (1 - credibleInterval) / 2;
        const upperPercentile = 1 - lowerPercentile;
        const lowerIdx = Math.floor(n * lowerPercentile);
        const upperIdx = Math.min(Math.floor(n * upperPercentile), n - 1);

        mean[i] = values.reduce((a, b) => a + b, 0) / n;
        lower[i] = sorted[lowerIdx];
        upper[i] = sorted[upperIdx];
      }

      return { mean, lower, upper };
    }
  }

  /**
   * Create a summary of the model
   * @returns {string} Model summary
   */
  summary() {
    let summary = `Model: ${this.name}\n`;
    summary += `Variables:\n`;

    for (const [name, distribution] of this.variables.entries()) {
      const observed = distribution.observed !== null ? ' (observed)' : '';
      summary += `  - ${name}: ${distribution.name}${observed}\n`;
    }

    return summary;
  }
}

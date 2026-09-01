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
 * Since 0.5.0 the model runs on plain numbers/arrays (no tensors).
 * Gradients of the joint log-probability are ANALYTIC for the prior terms
 * (via @tangent.to/proba's dlogpdf) and central finite differences for
 * {@link Model#potential} terms (arbitrary user functions).
 *
 * @see {@link https://www.pymc.io/|PyMC Documentation}
 */

import { valueAndGradFns } from '@tangent.to/grad';

/** Sum a number or an array of numbers. */
function sumOf(v) {
  if (Array.isArray(v)) {
    let s = 0;
    for (let i = 0; i < v.length; i++) s += v[i];
    return s;
  }
  return v;
}

/**
 * Bayesian probabilistic model: a DAG of random variables (priors), observed
 * likelihoods, generic {@link Model#potential} log-density terms, and named
 * {@link Model#deterministic} transforms, exposing the joint log-probability
 * and its gradient for the MCMC samplers.
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
    this.potentialGrads = new Map(); // Optional analytic gradients for potentials
    this.deterministics = new Map(); // Named transforms recorded in the trace
  }

  /**
   * Register a generic log-density term (a "potential" / factor) contributing to
   * the joint log-probability. `fn(params)` receives the current free-variable
   * values as plain numbers (or arrays) keyed by name and must return a number
   * or an array of log-density values (which are summed into the total).
   *
   * This is the general mechanism for likelihoods whose parameters are arbitrary
   * deterministic functions of the latent variables and data - the deterministic
   * expression is computed inside `fn` with ordinary JavaScript math:
   *
   * ```js
   * model.potential('y', (v) =>
   *   new Normal(xData.map((x) => v.slope * x + v.intercept), v.sigma).logProb(yData));
   * ```
   *
   * Gradients of potentials are estimated by central finite differences by
   * default; priors added with {@link Model#addVariable} get analytic gradients.
   * For a large data term this finite-difference cost (2·(#free params) extra
   * evaluations of `fn` per gradient) dominates NUTS/HMC — and, more seriously,
   * finite-difference error costs the leapfrog integrator its symplectic
   * property, degrading the acceptance rate. Two ways to avoid it:
   *
   * {@link Model#autoPotential} writes the term in `@tangent.to/grad` ops and
   * differentiates it exactly, with no derivation by hand. Prefer it.
   *
   * Otherwise pass an explicit `gradFn` returning the analytic gradient:
   *
   * ```js
   * model.potential('y', (v) => new Normal(mu(v), v.sigma).logProb(yData),
   *   (v) => ({ slope: dSlope(v), intercept: dIntercept(v), sigma: dSigma(v) }));
   * ```
   *
   * `gradFn(params)` must return an object mapping each free-variable name to the
   * partial derivative of THIS term's log-density with respect to it (a number,
   * or an array for a vector-valued variable). It is added to the analytic prior
   * gradients; omit an entry whose partial is zero.
   *
   * @param {string} name - Identifier for the term
   * @param {(params: Object) => (number|Array<number>)} fn - Returns log-density value(s)
   * @param {(params: Object) => Object} [gradFn] - Optional analytic gradient of `fn`
   * @returns {Model} this
   */
  potential(name, fn, gradFn = null) {
    this.potentials.set(name, fn);
    if (gradFn) this.potentialGrads.set(name, gradFn);
    else this.potentialGrads.delete(name);
    return this;
  }

  /**
   * Register a potential written in `@tangent.to/grad` ops, differentiated
   * exactly by reverse-mode autodiff.
   *
   * The same term as {@link Model#potential}, but `fn` builds its log-density
   * from grad's ops instead of plain arithmetic, and returns that expression
   * rather than a number. No gradient is derived by hand and none is
   * approximated:
   *
   * ```js
   * import { add, mul, sub, div, log, square, sum, matmul } from '@tangent.to/grad';
   *
   * model.autoPotential('y', (v) => {
   *   const z = div(sub(yData, matmul(X, v.beta)), v.sigma);
   *   return sub(mul(-0.5, sum(square(z))), mul(yData.length, log(v.sigma)));
   * });
   * ```
   *
   * Against the finite-difference fallback on a 21-parameter regression with
   * 300 observations: one likelihood evaluation per gradient instead of 2·P,
   * NUTS 7.7× faster end to end, and the same posterior. The gradient matches
   * a hand-derived closed form to ~1e-13, where central differences are off by
   * ~2e-7.
   *
   * The value and gradient share one evaluation, so the sampler's
   * value-and-gradient path sweeps the data once rather than twice.
   *
   * @param {string} name - Identifier for the term
   * @param {(params: Object) => Object} fn - Builds the log-density as a grad
   *   expression; receives the free variables as grad `Var`s keyed by name
   * @returns {Model} this
   */
  autoPotential(name, fn) {
    const { value, gradient } = valueAndGradFns(fn);
    return this.potential(name, value, gradient);
  }

  /**
   * Register a named deterministic transform of the parameters for recording in
   * the trace (computed post-hoc from posterior draws). Deterministics do NOT
   * affect the log-probability - use {@link Model#potential} for likelihood or
   * factor terms.
   *
   * @param {string} name - Identifier for the transform
   * @param {(params: Object) => (number|Array)} fn - The transform
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

  /** Sum of all potential terms at the given parameter values. */
  _potentialSum(params) {
    let s = 0;
    for (const fn of this.potentials.values()) {
      s += sumOf(fn(params));
    }
    return s;
  }

  /**
   * Compute the log probability of the model given parameter values
   * @param {Object} params - Parameter values as {name: number|Array} pairs
   * @returns {number} Log probability (scalar)
   */
  logProb(params) {
    let logProb = 0;

    // Log probability for each variable
    for (const [name, distribution] of this.variables.entries()) {
      const value = params[name];

      if (value !== undefined) {
        logProb += sumOf(distribution.logProb(value));
      } else if (distribution.observed !== null) {
        // For observed variables, compute log likelihood
        logProb += sumOf(distribution.logProb(distribution.observed));
      }
    }

    // Generic potential / likelihood terms (deterministic-mean factors).
    if (this.potentials.size) {
      logProb += this._potentialSum(params);
    }

    return logProb;
  }

  /**
   * Compute the log probability and its gradient with respect to parameters.
   *
   * Prior terms are differentiated analytically (proba dlogpdf); potential
   * terms by central finite differences with step h = 1e-6 * max(1, |x|)
   * per scalar component.
   *
   * @param {Object} params - Parameter values as {name: number|Array} pairs
   * @returns {{logProb: number, gradients: Object}} The scalar log probability
   *   and a `{name: number|Array}` map of gradients, one per parameter
   */
  logProbAndGradient(params) {
    let logProb = 0;
    const gradients = {};

    // Analytic prior gradients + observed-variable likelihood (constant in params)
    for (const [name, distribution] of this.variables.entries()) {
      const value = params[name];

      if (value !== undefined) {
        logProb += sumOf(distribution.logProb(value));
        gradients[name] = distribution.dlogProbDx(value);
      } else if (distribution.observed !== null) {
        logProb += sumOf(distribution.logProb(distribution.observed));
      }
    }

    // Potentials: value plus gradient. Analytic gradients skip the
    // 2·(#free params) extra evaluations per gradient that finite differences
    // require, which dominates NUTS/HMC on a large data term.
    if (this.potentials.size) {
      logProb += this._potentialSum(params);
      this._potentialGradients(params, gradients);
    }

    return { logProb, gradients };
  }

  /**
   * Gradient of the joint log-probability WITHOUT its value — exactly
   * `logProbAndGradient(params).gradients`, skipping the potential-value pass.
   *
   * Samplers' leapfrog steps only consume the gradient, but for a model with
   * an analytic potential gradient, computing the discarded value costs a
   * full extra pass over the data at every leapfrog step. This method
   * omits it; the returned gradients are identical.
   *
   * @param {Object} params - Parameter values as {name: number|Array} pairs
   * @returns {Object} `{name: number|Array}` map of gradients
   */
  gradientsOnly(params) {
    const gradients = {};

    for (const [name, distribution] of this.variables.entries()) {
      const value = params[name];
      if (value !== undefined) {
        gradients[name] = distribution.dlogProbDx(value);
      }
    }

    if (this.potentials.size) {
      this._potentialGradients(params, gradients);
    }

    return gradients;
  }

  /**
   * Accumulate every potential's gradient into `gradients` (mutated).
   * A potential registered with an analytic gradient function contributes it
   * directly (one pass); the rest fall back to central finite differences on
   * their pooled sum. Shared by logProbAndGradient and gradientsOnly.
   * @private
   */
  _potentialGradients(params, gradients) {
    const fdPotentials = [];
    for (const [pname, fn] of this.potentials.entries()) {
      const gradFn = this.potentialGrads.get(pname);
      if (!gradFn) {
        fdPotentials.push(fn);
        continue;
      }
      const g = gradFn(params);
      for (const name of Object.keys(g)) {
        const val = g[name];
        const cur = gradients[name];
        if (Array.isArray(val)) {
          if (Array.isArray(cur)) {
            for (let i = 0; i < val.length; i++) cur[i] += val[i];
          } else {
            const base = cur ?? 0;
            gradients[name] = val.map((v) => v + base);
          }
        } else {
          gradients[name] = (cur ?? 0) + val;
        }
      }
    }

    // Finite-difference the pooled sum of any potentials without an analytic
    // gradient (no-op when every term supplied one).
    if (fdPotentials.length) {
      const potSum = (work) => {
        let s = 0;
        for (const fn of fdPotentials) s += sumOf(fn(work));
        return s;
      };
      for (const name of Object.keys(params)) {
        const v = params[name];
        if (Array.isArray(v)) {
          const g = Array.isArray(gradients[name])
            ? gradients[name]
            : new Array(v.length).fill(gradients[name] ?? 0);
          const work = { ...params, [name]: v.slice() };
          for (let i = 0; i < v.length; i++) {
            const h = 1e-6 * Math.max(1, Math.abs(v[i]));
            work[name][i] = v[i] + h;
            const fPlus = potSum(work);
            work[name][i] = v[i] - h;
            const fMinus = potSum(work);
            work[name][i] = v[i];
            g[i] += (fPlus - fMinus) / (2 * h);
          }
          gradients[name] = g;
        } else {
          const h = 1e-6 * Math.max(1, Math.abs(v));
          const fPlus = potSum({ ...params, [name]: v + h });
          const fMinus = potSum({ ...params, [name]: v - h });
          gradients[name] = (gradients[name] ?? 0) + (fPlus - fMinus) / (2 * h);
        }
      }
    }
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
        samples[name] = distribution.sample([nSamples]);
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
   * Evaluate registered {@link Model#deterministic} transforms on each posterior
   * draw and append them to the trace as extra columns. Computed post-hoc - they
   * do not affect sampling - and the MCMC samplers call this automatically before
   * returning their trace. Each `fn(params)` receives a `{name: number}` map of
   * the free-variable values for one draw and may return a number or an array
   * (legacy tensor-like returns with `arraySync` are read out too).
   *
   * @param {Object} trace - Trace map `{ name: [...] }` or a `{ trace }` wrapper.
   * @returns {Object} The same trace, with one column per deterministic.
   */
  computeDeterministics(trace) {
    if (!this.deterministics.size || !trace) return trace;
    const cols = trace.trace || trace;
    const freeNames = this.getFreeVariableNames();
    const anchor = freeNames.find((n) => Array.isArray(cols[n]));
    const nDraws = anchor ? cols[anchor].length : 0;
    for (const [name, fn] of this.deterministics.entries()) {
      const out = new Array(nDraws);
      for (let i = 0; i < nDraws; i++) {
        const params = {};
        for (const fv of freeNames) params[fv] = cols[fv] ? cols[fv][i] : undefined;
        let v = fn(params);
        if (v && typeof v.arraySync === 'function') {
          const arr = v.arraySync();
          if (typeof v.dispose === 'function') v.dispose();
          v = arr;
        }
        out[i] = v;
      }
      cols[name] = out;
    }
    return trace;
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

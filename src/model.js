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
import { Distribution } from './distributions/base.js';
import { makeTransform, supportOf } from './transforms.js';

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
    // Terms that exist as compiled grad plans, by name: what can be written
    // out as data and sent to a worker. Filled by autoPotential and observe.
    this.compiledTerms = new Map();
    // observe() terms, name -> data. Kept apart from observedVars, which is
    // keyed by VARIABLE name and read when deciding which variables to
    // transform; an observed term is a potential, not a variable.
    this.observedTerms = new Map();
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
   * `add` and `mul` take any number of operands, which is what keeps a mean
   * with several terms readable. JavaScript cannot overload `+`, so this is as
   * close as the language gets to PyMC's `mu0 + tau * z + gamma`:
   *
   * ```js
   * const mu = add(v.mu0, mul(tau, matmul(Z, v.z)), matmul(C, v.cyc));
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
   * The tape is COMPILED by default: built once and replayed at each new set of
   * parameters, rather than reconstructed per call. That is worth roughly 6x on
   * a real model, and it is safe here because of the contract above. `fn`
   * builds an expression out of grad's ops, so its graph is fixed by the way it
   * is written; and a sampler holds every parameter's shape constant for the
   * length of a run, so nothing can change underneath the plan.
   *
   * Stepping outside that contract breaks the assumption, and the two ways to
   * do it both take deliberate effort: branching on a parameter's numeric value
   * by reaching into `.data`, so different draws take different paths through
   * `fn`, or closing over data that is mutated while the sampler runs. Neither
   * is an expression built from ops, which is why the default is what it is.
   * Pass `{ compile: false }` if you need one of them.
   *
   * @param {string} name - Identifier for the term
   * @param {(params: Object) => Object} fn - Builds the log-density as a grad
   *   expression; receives the free variables as grad `Var`s keyed by name
   * @param {Object} [options]
   * @param {boolean} [options.compile=true] - reuse the tape across calls
   * @returns {Model} this
   */
  autoPotential(name, fn, options = {}) {
    const { compile = true } = options;
    const fns = valueAndGradFns(fn, { compile });
    if (compile) this.compiledTerms.set(name, fns.compiled);
    else this.compiledTerms.delete(name);
    return this.potential(name, fns.value, fns.gradient);
  }

  /**
   * Declare an observed random variable: the likelihood, derived from a
   * distribution instead of written out.
   *
   * `factory` receives the free variables as grad `Var`s and returns a
   * distribution whose parameters are expressions in them. The term added to
   * the model is that distribution's `logDensity` at `observed`, differentiated
   * exactly and compiled, so this is {@link Model#autoPotential} with the
   * density supplied by the distribution rather than by you. What that
   * removes from a model is everything a PyMC user never writes: the kernel,
   * the `-n log sigma`, the normalizing constant.
   *
   * ```js
   * const { add, mul } = mc.ops;
   * model.addVariable('a', new Normal(0, 5));
   * model.addVariable('b', new Normal(0, 5));
   * model.addVariable('sigma', new HalfNormal(2));
   * model.observe('y', (v) => new Normal(add(v.a, mul(v.b, xData)), v.sigma), yData);
   * ```
   *
   * The seven built-in distributions can be observed. A user-defined one
   * cannot be differentiated and is refused here; write its term with
   * `autoPotential`.
   *
   * @param {string} name - Identifier for the term
   * @param {(v: Object) => Distribution} factory - Builds the observation
   *   distribution from the free variables
   * @param {number|Array} observed - The data
   * @param {Object} [options] - As for {@link Model#autoPotential}
   * @returns {Model} this
   */
  observe(name, factory, observed, options = {}) {
    if (typeof factory !== 'function') {
      throw new Error(`observe("${name}"): expected a function returning a distribution`);
    }
    if (observed === undefined || observed === null) {
      throw new Error(`observe("${name}"): observed data is required`);
    }
    const term = (v) => {
      const dist = factory(v);
      if (!(dist instanceof Distribution)) {
        throw new Error(
          `observe("${name}"): the factory must return one of mc's distributions, got ` +
            `${dist === null ? 'null' : typeof dist}`,
        );
      }
      return dist.logDensity(observed);
    };
    this.observedTerms.set(name, observed);
    return this.autoPotential(name, term, options);
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
    this._transformCache = null;

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
   * The transform for each free variable, built from its prior's support and
   * cached. Rebuilt when a variable is added.
   * @private
   */
  _transforms() {
    if (this._transformCache && this._transformCache.size === this.variables.size) {
      return this._transformCache;
    }
    const map = new Map();
    for (const [name, dist] of this.variables.entries()) {
      map.set(name, makeTransform(supportOf(dist)));
    }
    this._transformCache = map;
    return map;
  }

  /**
   * Does any free variable have a bounded support?
   *
   * When nothing is constrained the unconstrained space IS the constrained one
   * and every transform is the identity, so a sampler can skip the mapping
   * entirely and behave exactly as it did before this existed.
   *
   * @returns {boolean}
   */
  hasConstrainedVariables() {
    for (const [name, T] of this._transforms()) {
      if (!T.isIdentity && this.observedVars.get(name) === undefined) return true;
    }
    return false;
  }

  /**
   * Map constrained parameter values into the unconstrained space a gradient
   * sampler should move through.
   *
   * @param {Object} params - `{name: number|Array}` in the model's own units
   * @returns {Object} the same shape, unconstrained
   */
  toUnconstrained(params) {
    const T = this._transforms();
    const out = {};
    for (const [name, value] of Object.entries(params)) {
      const t = T.get(name);
      if (!t || t.isIdentity) { out[name] = value; continue; }
      out[name] = Array.isArray(value) ? value.map(t.toUnconstrained) : t.toUnconstrained(value);
    }
    return out;
  }

  /**
   * Map unconstrained values back into the model's units.
   *
   * @param {Object} uparams - `{name: number|Array}`, unconstrained
   * @returns {Object} the same shape, constrained
   */
  toConstrained(uparams) {
    const T = this._transforms();
    const out = {};
    for (const [name, value] of Object.entries(uparams)) {
      const t = T.get(name);
      if (!t || t.isIdentity) { out[name] = value; continue; }
      out[name] = Array.isArray(value) ? value.map(t.toConstrained) : t.toConstrained(value);
    }
    return out;
  }

  /**
   * Joint log-probability and gradient in UNCONSTRAINED space.
   *
   * The change of variables adds Σ log|dx/du| to the log-density, which is
   * what keeps the posterior invariant: without it the sampler would explore
   * the transformed density, not the one you wrote. The gradient is chained
   * through the same derivative, plus the d/du of that Jacobian term.
   *
   * For a lower-bounded parameter x = a + eᵘ the Jacobian term is just u, so
   * its derivative is 1 — the "+1" below. For a doubly-bounded one it is
   * 1 − 2σ(u).
   *
   * @param {Object} uparams - `{name: number|Array}`, unconstrained
   * @returns {{logProb: number, gradients: Object}} both in unconstrained terms
   */
  logProbAndGradientUnconstrained(uparams) {
    const T = this._transforms();
    const params = this.toConstrained(uparams);
    const { logProb, gradients } = this.logProbAndGradient(params);

    let logJacobian = 0;
    const out = {};
    for (const [name, uvalue] of Object.entries(uparams)) {
      const t = T.get(name);
      const g = gradients[name];
      if (!t || t.isIdentity) { out[name] = g; continue; }
      if (Array.isArray(uvalue)) {
        const acc = new Array(uvalue.length);
        for (let i = 0; i < uvalue.length; i++) {
          logJacobian += t.logDetJacobian(uvalue[i]);
          acc[i] = (Array.isArray(g) ? g[i] : (g ?? 0)) * t.dxdu(uvalue[i])
            + t.dLogDetJacobian(uvalue[i]);
        }
        out[name] = acc;
      } else {
        logJacobian += t.logDetJacobian(uvalue);
        out[name] = (g ?? 0) * t.dxdu(uvalue) + t.dLogDetJacobian(uvalue);
      }
    }
    return { logProb: logProb + logJacobian, gradients: out };
  }

  /**
   * Gradient only, in unconstrained space — the leapfrog hot path.
   * @param {Object} uparams
   * @returns {Object} `{name: number|Array}`
   */
  gradientsOnlyUnconstrained(uparams) {
    return this.logProbAndGradientUnconstrained(uparams).gradients;
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

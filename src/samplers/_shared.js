/**
 * Internal helpers shared by the MCMC samplers (Metropolis-Hastings, HMC, NUTS).
 * Not part of the public API.
 */

/**
 * Compute the Hamiltonian (total energy) H = -logProb(position) + ½·pᵀp.
 * @param {Model} model - The probabilistic model
 * @param {Object} position - Current position (parameters)
 * @param {Object} momentum - Current momentum
 * @returns {number} Hamiltonian value
 */
export function computeHamiltonian(model, position, momentum) {
  const logProb = model.logProb(position).arraySync();

  let kineticEnergy = 0;
  for (const name of Object.keys(momentum)) {
    const p = momentum[name];
    kineticEnergy += 0.5 * p * p;
  }

  return -logProb + kineticEnergy;
}

/**
 * Create an empty trace object with one array per free variable.
 * @param {Array<string>} variableNames - Free variable names
 * @returns {Object} Trace object `{ name: [] }`
 */
export function initTrace(variableNames) {
  const trace = {};
  for (const name of variableNames) {
    trace[name] = [];
  }
  return trace;
}

/**
 * Append the current parameter values to the trace.
 * @param {Object} trace - Trace object created by {@link initTrace}
 * @param {Object} params - Current parameter values
 * @param {Array<string>} variableNames - Free variable names
 */
export function recordSample(trace, params, variableNames) {
  for (const name of variableNames) {
    trace[name].push(params[name]);
  }
}

/**
 * Internal helpers shared by the MCMC samplers (Metropolis-Hastings, HMC, NUTS).
 * Not part of the public API. Positions/momenta are plain numbers or arrays
 * of numbers keyed by variable name.
 */

/**
 * Kinetic energy ½·pᵀp of a scalar-or-array momentum dict.
 * @param {Object} momentum - Momentum values keyed by variable name
 * @returns {number} Kinetic energy
 */
export function kineticEnergy(momentum) {
  let ke = 0;
  for (const p of Object.values(momentum)) {
    if (Array.isArray(p)) {
      for (let i = 0; i < p.length; i++) ke += 0.5 * p[i] * p[i];
    } else {
      ke += 0.5 * p * p;
    }
  }
  return ke;
}

/**
 * Compute the Hamiltonian (total energy) H = -logProb(position) + ½·pᵀp.
 * @param {Model} model - The probabilistic model
 * @param {Object} position - Current position (parameters)
 * @param {Object} momentum - Current momentum
 * @returns {number} Hamiltonian value
 */
export function computeHamiltonian(model, position, momentum) {
  const logProb = model.logProb(position);
  return -logProb + kineticEnergy(momentum);
}

/**
 * Elementwise a + s*b for scalars or arrays (shapes must match).
 * @param {number|Array<number>} a
 * @param {number} s
 * @param {number|Array<number>} b
 * @returns {number|Array<number>}
 */
export function axpy(a, s, b) {
  if (Array.isArray(a)) {
    const out = new Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = a[i] + s * b[i];
    return out;
  }
  return a + s * b;
}

/**
 * Dot product of two scalar-or-array values.
 * @param {number|Array<number>} a
 * @param {number|Array<number>} b
 * @returns {number}
 */
export function dotValue(a, b) {
  if (Array.isArray(a)) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }
  return a * b;
}

/**
 * Draw a standard-normal momentum with the same shape as the position.
 * @param {Object} position - Position dict used as the shape template
 * @param {Object} rng - RNG with .normal()
 * @returns {Object} Momentum dict
 */
export function sampleMomentum(position, rng) {
  const momentum = {};
  for (const [name, v] of Object.entries(position)) {
    momentum[name] = Array.isArray(v) ? v.map(() => rng.normal()) : rng.normal();
  }
  return momentum;
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

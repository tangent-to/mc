import { isOptions } from '../distributions/base.js';
import { sampleModelChains } from '../parallel.js';
import { getRng } from '../rng.js';
import { axpy, computeHamiltonian, initTrace, kineticEnergy, recordSample, sampleMomentum } from './_shared.js';

/**
 * Hamiltonian Monte Carlo (HMC) sampler
 *
 * Uses gradient information for efficient exploration of the posterior.
 * HMC simulates Hamiltonian dynamics to propose distant states with high acceptance probability.
 *
 * **Hamiltonian**:
 * $$
 * H(\theta, p) = -\log p(\theta|y) + \frac{1}{2}p^T p
 * $$
 * where $\theta$ is position (parameters), $p$ is momentum.
 *
 * **Leapfrog integrator** preserves volume and is reversible:
 * 1. Half-step momentum: $p_{i+1/2} = p_i + \frac{\epsilon}{2}\nabla_\theta \log p(\theta_i|y)$
 * 2. Full-step position: $\theta_{i+1} = \theta_i + \epsilon p_{i+1/2}$
 * 3. Half-step momentum: $p_{i+1} = p_{i+1/2} + \frac{\epsilon}{2}\nabla_\theta \log p(\theta_{i+1}|y)$
 *
 * @see {@link https://arxiv.org/abs/1701.02434|A Conceptual Introduction to HMC}
 */
export class HamiltonianMC {
  /**
   * Accepts either positional arguments or a single options object.
   *
   * @param {number|Object} stepSize - Leapfrog step size (epsilon), or an options
   *   object `{ stepSize, nSteps }`
   * @param {number} [nSteps] - Number of leapfrog steps (L)
   *
   * @example
   * new HamiltonianMC(0.01, 10)
   * @example
   * new HamiltonianMC({ stepSize: 0.01, nSteps: 10 })
   */
  constructor(stepSize = 0.01, nSteps = 10) {
    if (isOptions(stepSize)) {
      const o = stepSize;
      nSteps = o.nSteps ?? 10;
      stepSize = o.stepSize ?? 0.01;
    }
    this.stepSize = stepSize;
    this.nSteps = nSteps;
  }

  /**
   * Get the sampler's configuration.
   * @returns {{stepSize: number, nSteps: number}}
   */
  getParams() {
    return { stepSize: this.stepSize, nSteps: this.nSteps };
  }

  /**
   * Leapfrog integrator for Hamiltonian dynamics
   * @param {Object} position - Current position (parameters)
   * @param {Object} momentum - Current momentum
   * @param {Model} model - The probabilistic model
   * @returns {Object} New position and momentum
   */
  leapfrog(position, momentum, model) {
    const variableNames = Object.keys(position);

    // Only the gradient is consumed here — gradientsOnly skips the potential
    // value pass that logProbAndGradient computes and this method discards.
    const gradOf = (q) =>
      typeof model.gradientsOnly === 'function'
        ? model.gradientsOnly(q)
        : model.logProbAndGradient(q).gradients;

    // Half step for momentum
    const grad = gradOf(position);
    let pNew = {};
    for (const name of variableNames) {
      pNew[name] = axpy(momentum[name], this.stepSize / 2, grad[name]);
    }

    // Full steps for position and momentum
    let qNew = { ...position };

    for (let i = 0; i < this.nSteps; i++) {
      // Full step for position
      for (const name of variableNames) {
        qNew[name] = axpy(qNew[name], this.stepSize, pNew[name]);
      }

      // Full step for momentum (except at end)
      if (i < this.nSteps - 1) {
        const gradNew = gradOf(qNew);
        for (const name of variableNames) {
          pNew[name] = axpy(pNew[name], this.stepSize, gradNew[name]);
        }
      }
    }

    // Half step for momentum at end
    const gradFinal = gradOf(qNew);
    for (const name of variableNames) {
      pNew[name] = axpy(pNew[name], this.stepSize / 2, gradFinal[name]);
    }

    return { position: qNew, momentum: pNew };
  }

  /**
   * Compute Hamiltonian (total energy)
   * @param {Object} position - Current position
   * @param {Object} momentum - Current momentum
   * @param {Model} model - The probabilistic model
   * @returns {number} Hamiltonian value
   */
  hamiltonian(position, momentum, model) {
    return computeHamiltonian(model, position, momentum);
  }

  /**
   * Run HMC sampling.
   *
   * The sampling controls may be passed positionally or as a single options
   * object. When an options object is supplied as the third argument, the
   * `burnIn` and `thin` positional arguments are ignored in favour of the
   * object's fields.
   *
   * @param {Model} model - The probabilistic model
   * @param {Object} initialValues - Initial parameter values
   * @param {Object|number} [nSamples=1000] - Number of samples, or an options object
   * @param {number} [nSamples.nSamples=1000] - Number of samples (options-object form)
   * @param {number} [nSamples.burnIn=500] - Number of burn-in samples to discard (options-object form)
   * @param {number} [nSamples.nWarmup] - Alias for `burnIn` (the name used by `sampleChains`); takes precedence when both are given (options-object form)
   * @param {number} [nSamples.thin=1] - Thinning interval (options-object form)
   * @param {number} [burnIn=500] - Number of burn-in samples to discard (positional form)
   * @param {number} [thin=1] - Thinning interval (positional form)
   * @returns {Object} Trace object with samples and diagnostics
   *
   * @example
   * hmc.sample(model, { mu: 0 }, 1000, 500, 1)
   * @example
   * hmc.sample(model, { mu: 0 }, { nSamples: 1000, burnIn: 500, thin: 1 })
   */
  sample(model, initialValues, nSamples = 1000, burnIn = 500, thin = 1) {
    if (isOptions(nSamples) && nSamples.chains > 1) {
      return sampleModelChains(model, initialValues, nSamples, 'hamiltonian', this.getParams());
    }
    let verbose = false;
    if (isOptions(nSamples)) {
      const o = nSamples;
      // Accept `nWarmup` (the name sampleChains/NUTS use) with `burnIn` kept as
      // a back-compat alias, so a user's warmup count isn't silently dropped.
      burnIn = o.nWarmup ?? o.burnIn ?? 500;
      thin = o.thin ?? 1;
      verbose = o.verbose ?? false;
      nSamples = o.nSamples ?? 1000;
    }
    const log = verbose ? console.log : () => {};
    const variableNames = model.getFreeVariableNames();
    const trace = initTrace(variableNames);
    const accepted = { count: 0, total: 0 };

    // Current state
    let currentParams = { ...initialValues };

    const totalIterations = burnIn + (nSamples * thin);

    log(`Starting Hamiltonian Monte Carlo sampling...`);
    log(`Step size: ${this.stepSize}, Steps: ${this.nSteps}`);
    log(`Burn-in: ${burnIn}, Samples: ${nSamples}, Thin: ${thin}`);
    log(`Total iterations: ${totalIterations}`);

    const rng = getRng();

    // Cache the current position's log-density: the potential energy only
    // changes when a proposal is accepted, so recomputing model.logProb over
    // the full data every iteration (as this.hamiltonian would) is wasted work.
    // Only the kinetic term, which depends on the freshly drawn momentum, is
    // recomputed each iteration.
    let currentLogProb = model.logProb(currentParams);

    for (let i = 0; i < totalIterations; i++) {
      // Sample momentum (matched to each variable's shape)
      const momentum = sampleMomentum(
        Object.fromEntries(variableNames.map((n) => [n, currentParams[n]])),
        rng,
      );

      // Current Hamiltonian, from the cached potential + this step's kinetic
      const currentH = -currentLogProb + kineticEnergy(momentum);

      // Leapfrog integration
      const { position: proposedParams, momentum: proposedMomentum } = this.leapfrog(
        currentParams,
        momentum,
        model
      );

      // Proposed Hamiltonian (this logProb pass is unavoidable, but reused on
      // acceptance as the next iteration's cached current value)
      const proposedLogProb = model.logProb(proposedParams);
      const proposedH = -proposedLogProb + kineticEnergy(proposedMomentum);

      // Accept or reject
      const logAcceptanceRatio = currentH - proposedH;
      const acceptanceRatio = Math.exp(Math.min(0, logAcceptanceRatio));

      accepted.total++;
      if (rng.float() < acceptanceRatio) {
        currentParams = proposedParams;
        currentLogProb = proposedLogProb;
        accepted.count++;
      }

      // Store samples after burn-in and according to thinning
      if (i >= burnIn && (i - burnIn) % thin === 0) {
        recordSample(trace, currentParams, variableNames);
      }

      // Progress logging
      if ((i + 1) % Math.max(1, Math.floor(totalIterations / 10)) === 0) {
        const progress = ((i + 1) / totalIterations * 100).toFixed(0);
        const acceptanceRate = (accepted.count / accepted.total * 100).toFixed(1);
        log(`Progress: ${progress}% | Acceptance rate: ${acceptanceRate}%`);
      }
    }

    const finalAcceptanceRate = (accepted.count / accepted.total * 100).toFixed(1);
    log(`Sampling complete! Final acceptance rate: ${finalAcceptanceRate}%`);

    model.computeDeterministics(trace); // append post-hoc deterministic columns

    return {
      trace,
      acceptanceRate: accepted.count / accepted.total,
      nSamples: nSamples
    };
  }
}

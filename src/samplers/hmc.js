import * as tf from '@tensorflow/tfjs-node';
import { isOptions } from '../distributions/base.js';

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

    // Convert to tensors
    const q = {};
    const p = {};

    for (const name of variableNames) {
      q[name] = tf.tensor(position[name]);
      p[name] = tf.tensor(momentum[name]);
    }

    return tf.tidy(() => {
      // Half step for momentum
      const grad = model.logProbAndGradient(q);
      const pHalf = {};
      for (const name of variableNames) {
        pHalf[name] = tf.add(p[name], tf.mul(this.stepSize / 2, grad.gradients[name]));
      }

      // Full steps for position and momentum
      let qNew = { ...q };
      let pNew = { ...pHalf };

      for (let i = 0; i < this.nSteps; i++) {
        // Full step for position
        for (const name of variableNames) {
          qNew[name] = tf.add(qNew[name], tf.mul(this.stepSize, pNew[name]));
        }

        // Full step for momentum (except at end)
        if (i < this.nSteps - 1) {
          const gradNew = model.logProbAndGradient(qNew);
          for (const name of variableNames) {
            pNew[name] = tf.add(pNew[name], tf.mul(this.stepSize, gradNew.gradients[name]));
          }
        }
      }

      // Half step for momentum at end
      const gradFinal = model.logProbAndGradient(qNew);
      for (const name of variableNames) {
        pNew[name] = tf.add(pNew[name], tf.mul(this.stepSize / 2, gradFinal.gradients[name]));
      }

      // Convert back to numbers
      const positionNew = {};
      const momentumNew = {};
      for (const name of variableNames) {
        positionNew[name] = qNew[name].arraySync();
        momentumNew[name] = pNew[name].arraySync();
      }

      return { position: positionNew, momentum: momentumNew };
    });
  }

  /**
   * Compute Hamiltonian (total energy)
   * @param {Object} position - Current position
   * @param {Object} momentum - Current momentum
   * @param {Model} model - The probabilistic model
   * @returns {number} Hamiltonian value
   */
  hamiltonian(position, momentum, model) {
    const logProb = model.logProb(position).arraySync();
    const variableNames = Object.keys(momentum);

    let kineticEnergy = 0;
    for (const name of variableNames) {
      const p = momentum[name];
      kineticEnergy += 0.5 * p * p;
    }

    return -logProb + kineticEnergy;
  }

  /**
   * Run HMC sampling
   * The sampling controls may be passed positionally or as a single options
   * object `{ nSamples, burnIn, thin }`.
   *
   * @param {Model} model - The probabilistic model
   * @param {Object} initialValues - Initial parameter values
   * @param {number|Object} nSamples - Number of samples, or an options object
   * @param {number} burnIn - Number of burn-in samples to discard
   * @param {number} thin - Thinning interval
   * @returns {Object} Trace object with samples and diagnostics
   */
  sample(model, initialValues, nSamples = 1000, burnIn = 500, thin = 1) {
    if (isOptions(nSamples)) {
      const o = nSamples;
      burnIn = o.burnIn ?? 500;
      thin = o.thin ?? 1;
      nSamples = o.nSamples ?? 1000;
    }
    const variableNames = model.getFreeVariableNames();
    const trace = {};
    const accepted = { count: 0, total: 0 };

    // Initialize trace arrays
    for (const name of variableNames) {
      trace[name] = [];
    }

    // Current state
    let currentParams = { ...initialValues };

    const totalIterations = burnIn + (nSamples * thin);

    console.log(`Starting Hamiltonian Monte Carlo sampling...`);
    console.log(`Step size: ${this.stepSize}, Steps: ${this.nSteps}`);
    console.log(`Burn-in: ${burnIn}, Samples: ${nSamples}, Thin: ${thin}`);
    console.log(`Total iterations: ${totalIterations}`);

    for (let i = 0; i < totalIterations; i++) {
      // Sample momentum
      const momentum = {};
      for (const name of variableNames) {
        momentum[name] = tf.randomNormal([]).arraySync();
      }

      // Current Hamiltonian
      const currentH = this.hamiltonian(currentParams, momentum, model);

      // Leapfrog integration
      const { position: proposedParams, momentum: proposedMomentum } = this.leapfrog(
        currentParams,
        momentum,
        model
      );

      // Proposed Hamiltonian
      const proposedH = this.hamiltonian(proposedParams, proposedMomentum, model);

      // Accept or reject
      const logAcceptanceRatio = currentH - proposedH;
      const acceptanceRatio = Math.exp(Math.min(0, logAcceptanceRatio));

      accepted.total++;
      if (Math.random() < acceptanceRatio) {
        currentParams = proposedParams;
        accepted.count++;
      }

      // Store samples after burn-in and according to thinning
      if (i >= burnIn && (i - burnIn) % thin === 0) {
        for (const name of variableNames) {
          trace[name].push(currentParams[name]);
        }
      }

      // Progress logging
      if ((i + 1) % Math.max(1, Math.floor(totalIterations / 10)) === 0) {
        const progress = ((i + 1) / totalIterations * 100).toFixed(0);
        const acceptanceRate = (accepted.count / accepted.total * 100).toFixed(1);
        console.log(`Progress: ${progress}% | Acceptance rate: ${acceptanceRate}%`);
      }
    }

    const finalAcceptanceRate = (accepted.count / accepted.total * 100).toFixed(1);
    console.log(`Sampling complete! Final acceptance rate: ${finalAcceptanceRate}%`);

    return {
      trace,
      acceptanceRate: accepted.count / accepted.total,
      nSamples: nSamples
    };
  }
}

import jstat from 'jstat';
import { isOptions } from '../distributions/base.js';
import { initTrace, recordSample } from './_shared.js';

/**
 * Metropolis-Hastings MCMC sampler
 *
 * A simple but effective MCMC algorithm for sampling from posterior distributions.
 *
 * **Algorithm**: At each iteration, a proposal $\theta'$ is generated from a symmetric
 * proposal distribution $q(\theta'|\theta) = \mathcal{N}(\theta, \sigma^2)$.
 * The proposal is accepted with probability:
 * $$
 * \alpha = \min\left(1, \frac{p(\theta'|y)}{p(\theta|y)}\right)
 * $$
 *
 * **Optimal acceptance rate**: Target ~23.4% for high-dimensional problems, 44% for 1D.
 *
 * @see {@link https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm|Metropolis-Hastings}
 */
export class MetropolisHastings {
  /**
   * Accepts either a positional argument or a single options object.
   *
   * @param {number|Object} proposalStd - Standard deviation for the Gaussian
   *   proposal distribution, or an options object `{ proposalStd }`
   *
   * @example
   * new MetropolisHastings(0.5)
   * @example
   * new MetropolisHastings({ proposalStd: 0.5 })
   */
  constructor(proposalStd = 0.1) {
    if (isOptions(proposalStd)) {
      proposalStd = proposalStd.proposalStd ?? 0.1;
    }
    this.proposalStd = proposalStd;
  }

  /**
   * Get the sampler's configuration.
   * @returns {{proposalStd: number}}
   */
  getParams() {
    return { proposalStd: this.proposalStd };
  }

  /**
   * Run Metropolis-Hastings sampling
   * The sampling controls may be passed positionally or as a single options
   * object `{ nSamples, burnIn, thin }`.
   *
   * @param {Model} model - The probabilistic model
   * @param {Object} initialValues - Initial parameter values
   * @param {number|Object} nSamples - Number of samples, or an options object
   * @param {number} burnIn - Number of burn-in samples to discard
   * @param {number} thin - Thinning interval (keep every nth sample)
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
    const trace = initTrace(variableNames);
    const accepted = { count: 0, total: 0 };

    // Current state
    let currentParams = { ...initialValues };
    let currentLogProb = model.logProb(currentParams).arraySync();

    const totalIterations = burnIn + (nSamples * thin);

    console.log(`Starting Metropolis-Hastings sampling...`);
    console.log(`Burn-in: ${burnIn}, Samples: ${nSamples}, Thin: ${thin}`);
    console.log(`Total iterations: ${totalIterations}`);

    for (let i = 0; i < totalIterations; i++) {
      // Propose new parameters
      const proposedParams = {};
      for (const name of variableNames) {
        const current = currentParams[name];
        const currentValue = typeof current === 'number' ? current : current.arraySync();
        const proposal = currentValue + jstat.normal.sample(0, this.proposalStd);
        proposedParams[name] = proposal;
      }

      // Compute acceptance probability
      const proposedLogProb = model.logProb(proposedParams).arraySync();
      const logAcceptanceRatio = proposedLogProb - currentLogProb;
      const acceptanceRatio = Math.exp(logAcceptanceRatio);

      // Accept or reject
      accepted.total++;
      if (Math.random() < acceptanceRatio) {
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

  /**
   * Tune the proposal standard deviation to achieve target acceptance rate
   * @param {number} currentAcceptanceRate - Current acceptance rate
   * @returns {number} New proposal standard deviation
   */
  tuneProposal(currentAcceptanceRate) {
    const targetRate = 0.234; // Optimal for high dimensions
    if (currentAcceptanceRate > targetRate) {
      this.proposalStd *= 1.1; // Increase step size
    } else {
      this.proposalStd *= 0.9; // Decrease step size
    }
    return this.proposalStd;
  }
}

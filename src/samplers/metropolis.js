import { isOptions } from '../distributions/base.js';
import { getRng } from '../rng.js';
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
   * Run Metropolis-Hastings sampling.
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
   * @param {number} [nSamples.thin=1] - Thinning interval, keep every nth sample (options-object form)
   * @param {number} [burnIn=500] - Number of burn-in samples to discard (positional form)
   * @param {number} [thin=1] - Thinning interval, keep every nth sample (positional form)
   * @returns {Object} Trace object with samples and diagnostics
   *
   * @example
   * mh.sample(model, { mu: 0 }, 1000, 500, 1)
   * @example
   * mh.sample(model, { mu: 0 }, { nSamples: 1000, burnIn: 500, thin: 1 })
   */
  sample(model, initialValues, nSamples = 1000, burnIn = 500, thin = 1) {
    let verbose = false;
    if (isOptions(nSamples)) {
      const o = nSamples;
      burnIn = o.burnIn ?? 500;
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
    let currentLogProb = model.logProb(currentParams);
    const rng = getRng();

    const totalIterations = burnIn + (nSamples * thin);

    log(`Starting Metropolis-Hastings sampling...`);
    log(`Burn-in: ${burnIn}, Samples: ${nSamples}, Thin: ${thin}`);
    log(`Total iterations: ${totalIterations}`);

    for (let i = 0; i < totalIterations; i++) {
      // Propose new parameters (Gaussian random walk, elementwise for arrays)
      const proposedParams = {};
      for (const name of variableNames) {
        const current = currentParams[name];
        proposedParams[name] = Array.isArray(current)
          ? current.map((c) => c + this.proposalStd * rng.normal())
          : current + this.proposalStd * rng.normal();
      }

      // Compute acceptance probability
      const proposedLogProb = model.logProb(proposedParams);
      const logAcceptanceRatio = proposedLogProb - currentLogProb;
      const acceptanceRatio = Math.exp(logAcceptanceRatio);

      // Accept or reject
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

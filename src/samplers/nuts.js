import * as tf from '@tensorflow/tfjs-node';

/**
 * No-U-Turn Sampler (NUTS)
 *
 * An extension of Hamiltonian Monte Carlo that automatically tunes the trajectory length.
 * NUTS eliminates the need to manually set the number of leapfrog steps by running
 * until the trajectory makes a "U-turn" (starts coming back).
 *
 * **Algorithm**: Uses recursive tree doubling to adaptively determine path length.
 * The trajectory is stopped when:
 * $$
 * (p^+ - p^-) \cdot \theta^+ < 0 \quad \text{or} \quad (p^+ - p^-) \cdot \theta^- < 0
 * $$
 * where $\theta^+, p^+$ are the forward endpoint and $\theta^-, p^-$ are the backward endpoint.
 *
 * **Advantages over HMC:**
 * - No manual tuning of trajectory length
 * - Better exploration of complex posteriors
 * - State-of-the-art MCMC performance
 *
 * **Dual averaging** is used to automatically tune step size during warm-up.
 *
 * @see {@link https://arxiv.org/abs/1111.4246|The No-U-Turn Sampler (Hoffman & Gelman, 2014)}
 */
export class NUTS {
  /**
   * @param {number} stepSize - Initial leapfrog step size (will be adapted during warmup)
   * @param {number} maxTreeDepth - Maximum tree depth (default 10, gives up to 2^10 = 1024 steps)
   * @param {number} targetAcceptance - Target acceptance rate for step size adaptation (default 0.8)
   */
  constructor(stepSize = 0.01, maxTreeDepth = 10, targetAcceptance = 0.8) {
    this.stepSize = stepSize;
    this.maxTreeDepth = maxTreeDepth;
    this.targetAcceptance = targetAcceptance;

    // Dual averaging parameters for step size adaptation
    this.mu = Math.log(10 * stepSize); // Log step size
    this.gamma = 0.05;
    this.t0 = 10;
    this.kappa = 0.75;
  }

  /**
   * Single leapfrog step
   * @param {Object} position - Current position (parameters)
   * @param {Object} momentum - Current momentum
   * @param {number} stepSize - Step size for this step
   * @param {Model} model - The probabilistic model
   * @returns {Object} New position and momentum
   */
  leapfrog(position, momentum, stepSize, model) {
    const variableNames = Object.keys(position);

    return tf.tidy(() => {
      // Convert to tensors
      const q = {};
      const p = {};

      for (const name of variableNames) {
        q[name] = typeof position[name] === 'number' ? tf.scalar(position[name]) : position[name];
        p[name] = typeof momentum[name] === 'number' ? tf.scalar(momentum[name]) : momentum[name];
      }

      // Half step for momentum
      const grad1 = model.logProbAndGradient(q);
      const pHalf = {};
      for (const name of variableNames) {
        pHalf[name] = tf.add(p[name], tf.mul(stepSize / 2, grad1.gradients[name]));
      }

      // Full step for position
      const qNew = {};
      for (const name of variableNames) {
        qNew[name] = tf.add(q[name], tf.mul(stepSize, pHalf[name]));
      }

      // Half step for momentum
      const grad2 = model.logProbAndGradient(qNew);
      const pNew = {};
      for (const name of variableNames) {
        pNew[name] = tf.add(pHalf[name], tf.mul(stepSize / 2, grad2.gradients[name]));
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
   * Check if trajectory is making a U-turn
   * @param {Object} positionMinus - Backward endpoint position
   * @param {Object} positionPlus - Forward endpoint position
   * @param {Object} momentumMinus - Backward endpoint momentum
   * @param {Object} momentumPlus - Forward endpoint momentum
   * @returns {boolean} True if trajectory is making a U-turn
   */
  isUTurn(positionMinus, positionPlus, momentumMinus, momentumPlus) {
    const variableNames = Object.keys(positionMinus);

    // Compute (theta_plus - theta_minus) . p_plus
    let dotPlus = 0;
    for (const name of variableNames) {
      const deltaTheta = positionPlus[name] - positionMinus[name];
      dotPlus += deltaTheta * momentumPlus[name];
    }

    // Compute (theta_plus - theta_minus) . p_minus
    let dotMinus = 0;
    for (const name of variableNames) {
      const deltaTheta = positionPlus[name] - positionMinus[name];
      dotMinus += deltaTheta * momentumMinus[name];
    }

    // U-turn if either dot product is negative
    return dotPlus < 0 || dotMinus < 0;
  }

  /**
   * Build tree recursively (doubling procedure)
   * @param {Object} position - Starting position
   * @param {Object} momentum - Starting momentum
   * @param {number} slice - Slice variable for acceptance
   * @param {number} direction - Direction (+1 forward, -1 backward)
   * @param {number} depth - Current tree depth
   * @param {number} stepSize - Step size
   * @param {Model} model - The probabilistic model
   * @param {number} H0 - Initial Hamiltonian
   * @returns {Object} Tree information
   */
  buildTree(position, momentum, slice, direction, depth, stepSize, model, H0) {
    const deltaMax = 1000; // Maximum energy change

    if (depth === 0) {
      // Base case: single leapfrog step
      const { position: positionNew, momentum: momentumNew } =
        this.leapfrog(position, momentum, direction * stepSize, model);

      const H = this.hamiltonian(positionNew, momentumNew, model);

      // Check if proposal is valid (energy difference not too large)
      const valid = (slice <= Math.exp(H0 - H));

      // Metropolis acceptance criterion
      const accept = (slice <= Math.exp(H0 - H)) ? 1 : 0;

      return {
        positionMinus: positionNew,
        positionPlus: positionNew,
        momentumMinus: momentumNew,
        momentumPlus: momentumNew,
        positionPrime: positionNew,
        nValid: valid ? 1 : 0,
        stop: !valid || (H0 - H) > deltaMax,
        alpha: Math.min(1, Math.exp(H0 - H)),
        nAlpha: 1
      };
    }

    // Recursion: build left and right subtrees
    const tree1 = this.buildTree(position, momentum, slice, direction, depth - 1, stepSize, model, H0);

    if (tree1.stop) {
      return tree1;
    }

    // Build second half of tree
    const position2 = direction === 1 ? tree1.positionPlus : tree1.positionMinus;
    const momentum2 = direction === 1 ? tree1.momentumPlus : tree1.momentumMinus;

    const tree2 = this.buildTree(position2, momentum2, slice, direction, depth - 1, stepSize, model, H0);

    // Combine trees
    const positionMinus = direction === 1 ? tree1.positionMinus : tree2.positionMinus;
    const positionPlus = direction === 1 ? tree2.positionPlus : tree1.positionPlus;
    const momentumMinus = direction === 1 ? tree1.momentumMinus : tree2.momentumMinus;
    const momentumPlus = direction === 1 ? tree2.momentumPlus : tree1.momentumPlus;

    // Check for U-turn
    const uTurn = this.isUTurn(positionMinus, positionPlus, momentumMinus, momentumPlus);

    // Sample from combined tree (with probability proportional to valid nodes)
    let positionPrime = tree1.positionPrime;
    const acceptProb = tree2.nValid / Math.max(tree1.nValid + tree2.nValid, 1);
    if (Math.random() < acceptProb) {
      positionPrime = tree2.positionPrime;
    }

    return {
      positionMinus,
      positionPlus,
      momentumMinus,
      momentumPlus,
      positionPrime,
      nValid: tree1.nValid + tree2.nValid,
      stop: tree1.stop || tree2.stop || uTurn,
      alpha: tree1.alpha + tree2.alpha,
      nAlpha: tree1.nAlpha + tree2.nAlpha
    };
  }

  /**
   * Run NUTS sampling
   * @param {Model} model - The probabilistic model
   * @param {Object} initialValues - Initial parameter values
   * @param {number} nSamples - Number of samples to generate
   * @param {number} nWarmup - Number of warmup samples (for step size adaptation)
   * @param {number} thin - Thinning interval
   * @returns {Object} Trace object with samples and diagnostics
   */
  sample(model, initialValues, nSamples = 1000, nWarmup = 500, thin = 1) {
    const variableNames = model.getFreeVariableNames();
    const trace = {};
    const accepted = { count: 0, total: 0 };

    // Initialize trace arrays
    for (const name of variableNames) {
      trace[name] = [];
    }

    // Current state
    let currentParams = { ...initialValues };

    const totalIterations = nWarmup + (nSamples * thin);

    console.log(`Starting NUTS sampling...`);
    console.log(`Warmup: ${nWarmup}, Samples: ${nSamples}, Thin: ${thin}`);
    console.log(`Total iterations: ${totalIterations}`);
    console.log(`Max tree depth: ${this.maxTreeDepth} (up to ${Math.pow(2, this.maxTreeDepth)} leapfrog steps)`);

    // Dual averaging state
    let logStepSize = Math.log(this.stepSize);
    let logStepSizeBar = 0;
    let hBar = 0;

    for (let i = 0; i < totalIterations; i++) {
      // Sample momentum
      const momentum = {};
      for (const name of variableNames) {
        momentum[name] = tf.randomNormal([]).arraySync();
      }

      // Compute current Hamiltonian
      const H0 = this.hamiltonian(currentParams, momentum, model);

      // Sample slice variable
      const slice = Math.random() * Math.exp(-H0);

      // Initialize tree
      let positionMinus = { ...currentParams };
      let positionPlus = { ...currentParams };
      let momentumMinus = { ...momentum };
      let momentumPlus = { ...momentum };
      let proposedParams = { ...currentParams };

      let depth = 0;
      let stop = false;
      let nValid = 1;
      let alpha = 0;
      let nAlpha = 0;

      // Build tree by doubling
      while (!stop && depth < this.maxTreeDepth) {
        // Choose direction randomly
        const direction = Math.random() < 0.5 ? -1 : 1;

        // Build subtree
        let tree;
        if (direction === 1) {
          tree = this.buildTree(
            positionPlus, momentumPlus, slice, direction, depth,
            this.stepSize, model, H0
          );
          positionPlus = tree.positionPlus;
          momentumPlus = tree.momentumPlus;
        } else {
          tree = this.buildTree(
            positionMinus, momentumMinus, slice, direction, depth,
            this.stepSize, model, H0
          );
          positionMinus = tree.positionMinus;
          momentumMinus = tree.momentumMinus;
        }

        // Sample from tree
        if (!tree.stop) {
          const acceptProb = tree.nValid / nValid;
          if (Math.random() < acceptProb) {
            proposedParams = tree.positionPrime;
          }
        }

        // Check for U-turn or divergence
        stop = tree.stop || this.isUTurn(positionMinus, positionPlus, momentumMinus, momentumPlus);

        nValid += tree.nValid;
        alpha += tree.alpha;
        nAlpha += tree.nAlpha;
        depth++;
      }

      // Update current state
      currentParams = proposedParams;

      // Compute acceptance rate for this iteration
      const iterAcceptRate = alpha / Math.max(nAlpha, 1);
      accepted.total++;
      if (iterAcceptRate > 0.5) { // Simplified acceptance tracking
        accepted.count++;
      }

      // Adapt step size during warmup using dual averaging
      if (i < nWarmup) {
        const eta = 1.0 / (i + 1 + this.t0);
        hBar = (1 - eta) * hBar + eta * (this.targetAcceptance - iterAcceptRate);
        logStepSize = this.mu - Math.sqrt(i + 1) / this.gamma * hBar;

        const logEta = Math.pow(i + 1, -this.kappa);
        logStepSizeBar = logEta * logStepSize + (1 - logEta) * logStepSizeBar;

        this.stepSize = Math.exp(logStepSize);
      } else if (i === nWarmup) {
        // End of warmup: set final step size
        this.stepSize = Math.exp(logStepSizeBar);
        console.log(`Warmup complete. Final step size: ${this.stepSize.toFixed(6)}`);
      }

      // Store samples after warmup and according to thinning
      if (i >= nWarmup && (i - nWarmup) % thin === 0) {
        for (const name of variableNames) {
          trace[name].push(currentParams[name]);
        }
      }

      // Progress logging
      if ((i + 1) % Math.max(1, Math.floor(totalIterations / 10)) === 0) {
        const progress = ((i + 1) / totalIterations * 100).toFixed(0);
        const avgAcceptRate = (accepted.count / accepted.total * 100).toFixed(1);
        const stepSizeStr = this.stepSize.toFixed(6);
        const phase = i < nWarmup ? 'Warmup' : 'Sampling';
        console.log(`Progress: ${progress}% | ${phase} | Step size: ${stepSizeStr} | Avg accept: ${avgAcceptRate}%`);
      }
    }

    const finalAcceptanceRate = (accepted.count / accepted.total * 100).toFixed(1);
    console.log(`Sampling complete! Final acceptance rate: ${finalAcceptanceRate}%`);
    console.log(`Adapted step size: ${this.stepSize.toFixed(6)}`);

    return {
      trace,
      acceptanceRate: accepted.count / accepted.total,
      nSamples: nSamples,
      stepSize: this.stepSize
    };
  }
}

import * as tf from '@tensorflow/tfjs-node';

/**
 * Model class for defining Bayesian probabilistic models.
 * Similar to PyMC's Model context manager.
 */
export class Model {
  constructor(name = 'model') {
    this.name = name;
    this.variables = new Map(); // Random variables in the model
    this.observedVars = new Map(); // Observed data
    this.logProbFn = null; // Compiled log probability function
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

      return logProb;
    });
  }

  /**
   * Compute the log probability and its gradient with respect to parameters
   * @param {Object} params - Parameter values as {name: tf.Tensor} pairs
   * @returns {Object} {logProb: number, gradients: Object}
   */
  logProbAndGradient(params) {
    // Convert params to tf.Variables for gradient computation
    const tfParams = {};
    const paramNames = Object.keys(params);

    for (const name of paramNames) {
      tfParams[name] = tf.variable(params[name]);
    }

    let logProbValue;
    const gradients = {};

    // Compute gradients
    const grads = tf.variableGrads(() => {
      logProbValue = this.logProb(tfParams);
      return logProbValue;
    });

    // Extract gradient values
    for (const name of paramNames) {
      if (grads.grads[tfParams[name].id]) {
        gradients[name] = grads.grads[tfParams[name].id];
      }
    }

    // Clean up variables
    for (const name of paramNames) {
      tfParams[name].dispose();
    }

    return {
      logProb: logProbValue.arraySync(),
      gradients: gradients
    };
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

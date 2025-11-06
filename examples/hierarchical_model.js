/**
 * Hierarchical Model Example
 *
 * This example demonstrates a hierarchical Bayesian model (multilevel model).
 * We model measurements from multiple groups, where each group has its own mean
 * but these means are drawn from a common distribution (partial pooling).
 *
 * Model structure (DAG):
 *   μ_global ~ Normal(0, 10)
 *   σ_global ~ Uniform(0.1, 10)
 *   μ_group[j] ~ Normal(μ_global, σ_global) for each group j
 *   σ_within ~ Uniform(0.1, 10)
 *   y[i] ~ Normal(μ_group[group[i]], σ_within)
 */

import { Model, Normal, Uniform, MetropolisHastings, printSummary } from '../src/index.js';
import * as tf from '@tensorflow/tfjs-node';

async function main() {
  console.log('=== Hierarchical Bayesian Model with JSMC ===\n');

  // Generate synthetic hierarchical data
  const nGroups = 5;
  const nPerGroup = 20;
  const trueGlobalMu = 10.0;
  const trueGlobalSigma = 2.0;
  const trueWithinSigma = 1.0;

  const groupMeans = [];
  const data = [];
  const groups = [];

  console.log('Generating synthetic hierarchical data...');
  console.log(`True global mean: ${trueGlobalMu}`);
  console.log(`True global std: ${trueGlobalSigma}`);
  console.log(`True within-group std: ${trueWithinSigma}\n`);

  // Generate group means from global distribution
  for (let j = 0; j < nGroups; j++) {
    const groupMean = trueGlobalMu + (Math.random() - 0.5) * 2 * trueGlobalSigma;
    groupMeans.push(groupMean);
    console.log(`Group ${j} true mean: ${groupMean.toFixed(2)}`);

    // Generate observations within group
    for (let i = 0; i < nPerGroup; i++) {
      const yi = groupMean + (Math.random() - 0.5) * 2 * trueWithinSigma;
      data.push(yi);
      groups.push(j);
    }
  }

  console.log(`\nGenerated ${data.length} observations across ${nGroups} groups\n`);

  // Define the hierarchical model
  const model = new Model('hierarchical_model');

  // Hyperpriors (global parameters)
  const muGlobal = new Normal(0, 10, 'mu_global');
  const sigmaGlobal = new Uniform(0.1, 10, 'sigma_global');
  const sigmaWithin = new Uniform(0.1, 10, 'sigma_within');

  model.addVariable('mu_global', muGlobal);
  model.addVariable('sigma_global', sigmaGlobal);
  model.addVariable('sigma_within', sigmaWithin);

  // Group-level parameters (one mean per group)
  for (let j = 0; j < nGroups; j++) {
    const muGroup = new Normal(0, 10, `mu_group_${j}`);
    model.addVariable(`mu_group_${j}`, muGroup);
  }

  // Override logProb to implement the hierarchical structure
  const originalLogProb = model.logProb.bind(model);
  model.logProb = function(params) {
    return tf.tidy(() => {
      let logProb = tf.scalar(0);

      // Hyperpriors
      logProb = tf.add(logProb, muGlobal.logProb(params.mu_global));
      logProb = tf.add(logProb, sigmaGlobal.logProb(params.sigma_global));
      logProb = tf.add(logProb, sigmaWithin.logProb(params.sigma_within));

      // Group-level priors: μ_group[j] ~ Normal(μ_global, σ_global)
      for (let j = 0; j < nGroups; j++) {
        const groupPrior = new Normal(params.mu_global, params.sigma_global);
        const groupLogProb = groupPrior.logProb(params[`mu_group_${j}`]);
        logProb = tf.add(logProb, groupLogProb);
      }

      // Likelihood: y[i] ~ Normal(μ_group[group[i]], σ_within)
      for (let i = 0; i < data.length; i++) {
        const groupIdx = groups[i];
        const groupMean = params[`mu_group_${groupIdx}`];
        const likelihood = new Normal(groupMean, params.sigma_within);
        const logLik = likelihood.logProb(data[i]);
        logProb = tf.add(logProb, logLik);
      }

      return logProb;
    });
  };

  console.log('Model structure:');
  console.log(model.summary());

  // Initialize parameters
  const initialValues = {
    mu_global: 0,
    sigma_global: 1,
    sigma_within: 1
  };

  for (let j = 0; j < nGroups; j++) {
    initialValues[`mu_group_${j}`] = 0;
  }

  // Run MCMC sampling
  console.log('Running MCMC sampling...');
  const sampler = new MetropolisHastings(0.3);

  const trace = sampler.sample(
    model,
    initialValues,
    2000,  // samples
    1000,  // burn-in
    2      // thin
  );

  // Analyze results
  printSummary(trace);

  console.log('\nComparing with true values:');
  console.log(`True global μ: ${trueGlobalMu.toFixed(4)}`);
  console.log(`True global σ: ${trueGlobalSigma.toFixed(4)}`);
  console.log(`True within σ: ${trueWithinSigma.toFixed(4)}`);
  console.log('\nTrue group means:');
  for (let j = 0; j < nGroups; j++) {
    console.log(`  Group ${j}: ${groupMeans[j].toFixed(4)}`);
  }
}

main();

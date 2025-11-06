/**
 * Linear Regression Example
 *
 * This example demonstrates how to use JSMC for Bayesian linear regression.
 * The model structure is: y = α + β*x + ε, where ε ~ N(0, σ)
 */

import { Model, Normal, Uniform, MetropolisHastings, printSummary } from '../src/index.js';
import * as tf from '@tensorflow/tfjs-node';

async function main() {
  console.log('=== Bayesian Linear Regression with JSMC ===\n');

  // Generate synthetic data
  const trueAlpha = 2.0;
  const trueBeta = 3.0;
  const trueSigma = 0.5;
  const n = 50;

  const x = [];
  const y = [];

  console.log('Generating synthetic data...');
  console.log(`True parameters: α=${trueAlpha}, β=${trueBeta}, σ=${trueSigma}\n`);

  for (let i = 0; i < n; i++) {
    const xi = Math.random() * 10;
    const yi = trueAlpha + trueBeta * xi + (Math.random() - 0.5) * 2 * trueSigma;
    x.push(xi);
    y.push(yi);
  }

  // Define the model
  const model = new Model('linear_regression');

  // Prior distributions (PyMC-like DAG structure)
  const alpha = new Normal(0, 10, 'alpha');
  const beta = new Normal(0, 10, 'beta');
  const sigma = new Uniform(0.01, 5, 'sigma');

  // Add variables to model
  model.addVariable('alpha', alpha);
  model.addVariable('beta', beta);
  model.addVariable('sigma', sigma);

  // Likelihood: y ~ Normal(alpha + beta*x, sigma)
  // Note: We need to define a custom likelihood function
  // Since distributions depend on each other (DAG structure)

  // Override the model's logProb to include the likelihood with dependencies
  const originalLogProb = model.logProb.bind(model);
  model.logProb = function(params) {
    return tf.tidy(() => {
      // Prior log probabilities
      let logProb = originalLogProb(params);

      // Likelihood: for each observation, compute p(y_i | x_i, alpha, beta, sigma)
      const alphaVal = params.alpha;
      const betaVal = params.beta;
      const sigmaVal = params.sigma;

      for (let i = 0; i < n; i++) {
        const mu = typeof alphaVal === 'number'
          ? alphaVal + betaVal * x[i]
          : tf.add(alphaVal, tf.mul(betaVal, x[i]));

        const likelihood = new Normal(mu, sigmaVal);
        const logLik = likelihood.logProb(y[i]);
        logProb = tf.add(logProb, logLik);
      }

      return logProb;
    });
  };

  console.log('Model structure:');
  console.log(model.summary());

  // Run MCMC sampling
  const sampler = new MetropolisHastings(0.5); // proposal std

  const initialValues = {
    alpha: 0,
    beta: 0,
    sigma: 1
  };

  const trace = sampler.sample(
    model,
    initialValues,
    1000,  // samples
    500,   // burn-in
    1      // thin
  );

  // Analyze results
  printSummary(trace);

  console.log('\nComparing with true values:');
  console.log(`True α: ${trueAlpha.toFixed(4)}`);
  console.log(`True β: ${trueBeta.toFixed(4)}`);
  console.log(`True σ: ${trueSigma.toFixed(4)}`);
}

main();

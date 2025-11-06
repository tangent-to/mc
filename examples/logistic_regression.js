/**
 * Logistic Regression Example
 *
 * This example demonstrates Bayesian logistic regression for binary classification.
 * The model uses a logistic link function: p(y=1) = σ(α + β*x)
 */

import { Model, Normal, Bernoulli, MetropolisHastings, printSummary } from '../src/index.js';
import * as tf from '@tensorflow/tfjs-node';

// Sigmoid function
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

async function main() {
  console.log('=== Bayesian Logistic Regression with JSMC ===\n');

  // Generate synthetic binary classification data
  const trueAlpha = -1.0;
  const trueBeta = 2.0;
  const n = 100;

  const x = [];
  const y = [];

  console.log('Generating synthetic binary data...');
  console.log(`True parameters: α=${trueAlpha}, β=${trueBeta}\n`);

  for (let i = 0; i < n; i++) {
    const xi = (Math.random() - 0.5) * 4;  // x in [-2, 2]
    const p = sigmoid(trueAlpha + trueBeta * xi);
    const yi = Math.random() < p ? 1 : 0;
    x.push(xi);
    y.push(yi);
  }

  console.log(`Generated ${n} observations (${y.filter(yi => yi === 1).length} positive, ${y.filter(yi => yi === 0).length} negative)`);

  // Define the model
  const model = new Model('logistic_regression');

  // Prior distributions
  const alpha = new Normal(0, 5, 'alpha');
  const beta = new Normal(0, 5, 'beta');

  model.addVariable('alpha', alpha);
  model.addVariable('beta', beta);

  // Override logProb to include logistic likelihood
  const originalLogProb = model.logProb.bind(model);
  model.logProb = function(params) {
    return tf.tidy(() => {
      // Prior log probabilities
      let logProb = originalLogProb(params);

      const alphaVal = params.alpha;
      const betaVal = params.beta;

      // Likelihood: y_i ~ Bernoulli(σ(α + β*x_i))
      for (let i = 0; i < n; i++) {
        const logit = typeof alphaVal === 'number'
          ? alphaVal + betaVal * x[i]
          : tf.add(alphaVal, tf.mul(betaVal, x[i]));

        // Convert logit to probability using sigmoid
        const p = typeof logit === 'number'
          ? sigmoid(logit)
          : tf.sigmoid(logit);

        const likelihood = new Bernoulli(p);
        const logLik = likelihood.logProb(y[i]);
        logProb = tf.add(logProb, logLik);
      }

      return logProb;
    });
  };

  console.log('\nModel structure:');
  console.log(model.summary());

  // Run MCMC sampling
  const sampler = new MetropolisHastings(0.2);

  const initialValues = {
    alpha: 0,
    beta: 0
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
}

main();

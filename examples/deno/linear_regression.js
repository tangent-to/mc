#!/usr/bin/env -S deno run --allow-read --allow-env

/**
 * Bayesian Linear Regression Example for Deno/Zed REPL
 *
 * This example demonstrates how to use @tangent.to/mc in Deno
 * to perform Bayesian linear regression.
 *
 * To run in Zed: Open this file and use the REPL mode
 * To run from command line: deno run --allow-read --allow-env linear_regression.js
 */

import { Model, Normal, Uniform, MetropolisHastings, printSummary } from "npm:@tangent.to/mc@0.2.0";

console.log("=== Bayesian Linear Regression with Deno ===\n");

// Generate synthetic data: y = 2 + 3*x + noise
console.log("Generating synthetic data...");
const n = 50;
const trueAlpha = 2.0;
const trueBeta = 3.0;
const trueSigma = 0.5;

const x = [];
const y = [];

for (let i = 0; i < n; i++) {
  const xi = Math.random() * 10;
  const noise = (Math.random() - 0.5) * 2 * trueSigma;
  const yi = trueAlpha + trueBeta * xi + noise;
  x.push(xi);
  y.push(yi);
}

console.log(`Generated ${n} data points`);
console.log(`True parameters: α=${trueAlpha}, β=${trueBeta}, σ=${trueSigma}\n`);

// Create Bayesian model
console.log("Building Bayesian model...");
const model = new Model('linear_regression');

// Define priors
const alpha = new Normal(0, 10, 'alpha');
const beta = new Normal(0, 10, 'beta');
const sigma = new Uniform(0.01, 5, 'sigma');

model.addVariable('alpha', alpha);
model.addVariable('beta', beta);
model.addVariable('sigma', sigma);

// Define log probability function (likelihood + priors)
model.logProb = function(params) {
  let logProb = alpha.logProb(params.alpha)
    .add(beta.logProb(params.beta))
    .add(sigma.logProb(params.sigma));

  // Add likelihood for each observation
  for (let i = 0; i < x.length; i++) {
    const mu = params.alpha + params.beta * x[i];
    const likelihood = new Normal(mu, params.sigma);
    logProb = logProb.add(likelihood.logProb(y[i]));
  }

  return logProb;
};

console.log("Model structure:");
console.log(model.summary());

// Run MCMC sampling
console.log("\nRunning Metropolis-Hastings sampler...");
const sampler = new MetropolisHastings(0.5);

const initialValues = {
  alpha: 0,
  beta: 0,
  sigma: 1
};

const trace = sampler.sample(
  model,
  initialValues,
  1000,  // nSamples
  500,   // burnIn
  1      // thin
);

// Analyze results
console.log("\n=== Posterior Summary ===\n");
printSummary(trace);

console.log("\n=== Comparison with True Values ===");
console.log(`True α: ${trueAlpha.toFixed(4)}`);
console.log(`True β: ${trueBeta.toFixed(4)}`);
console.log(`True σ: ${trueSigma.toFixed(4)}`);

console.log("\n✓ Bayesian inference complete!");

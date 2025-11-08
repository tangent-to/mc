#!/usr/bin/env -S deno run --allow-read --allow-env

/**
 * Hierarchical Bayesian Model Example for Deno/Zed REPL
 *
 * This example demonstrates a hierarchical model with partial pooling
 * across groups. This is useful when you have grouped data and want
 * to borrow strength across groups.
 *
 * To run in Zed: Open this file and use the REPL mode
 * To run from command line: deno run --allow-read --allow-env hierarchical_model.js
 */

import { Model, Normal, Uniform, MetropolisHastings, printSummary } from "npm:@tangent.to/mc@0.2.0";

console.log("=== Hierarchical Bayesian Model with Deno ===\n");

// Generate hierarchical data
// 3 groups with different means but similar variance
console.log("Generating hierarchical data (3 groups)...");

const groups = [
  { name: "Group A", trueMean: 10.0, data: [] },
  { name: "Group B", trueMean: 12.0, data: [] },
  { name: "Group C", trueMean: 11.0, data: [] }
];

const trueGlobalMean = 11.0;
const trueSigma = 1.5;

// Generate data for each group
for (const group of groups) {
  for (let i = 0; i < 10; i++) {
    const noise = (Math.random() - 0.5) * 2 * trueSigma;
    group.data.push(group.trueMean + noise);
  }
  console.log(`${group.name}: n=${group.data.length}, true mean=${group.trueMean}`);
}

console.log(`\nTrue global mean: ${trueGlobalMean}`);
console.log(`True σ: ${trueSigma}\n`);

// Build hierarchical model
console.log("Building hierarchical model...");
const model = new Model('hierarchical');

// Hyperpriors (top level)
const muGlobal = new Normal(0, 20, 'mu_global');
const sigmaGlobal = new Uniform(0.1, 10, 'sigma_global');

// Group-level parameters (middle level)
const muA = new Normal(0, 10, 'mu_a');
const muB = new Normal(0, 10, 'mu_b');
const muC = new Normal(0, 10, 'mu_c');

// Observation noise (bottom level)
const sigma = new Uniform(0.1, 5, 'sigma');

model.addVariable('mu_global', muGlobal);
model.addVariable('sigma_global', sigmaGlobal);
model.addVariable('mu_a', muA);
model.addVariable('mu_b', muB);
model.addVariable('mu_c', muC);
model.addVariable('sigma', sigma);

// Define log probability
model.logProb = function(params) {
  // Hyperpriors
  let logProb = muGlobal.logProb(params.mu_global)
    .add(sigmaGlobal.logProb(params.sigma_global))
    .add(sigma.logProb(params.sigma));

  // Group means depend on global mean (hierarchical structure)
  const muAPrior = new Normal(params.mu_global, params.sigma_global);
  const muBPrior = new Normal(params.mu_global, params.sigma_global);
  const muCPrior = new Normal(params.mu_global, params.sigma_global);

  logProb = logProb
    .add(muAPrior.logProb(params.mu_a))
    .add(muBPrior.logProb(params.mu_b))
    .add(muCPrior.logProb(params.mu_c));

  // Likelihood for Group A
  for (const y of groups[0].data) {
    const likelihood = new Normal(params.mu_a, params.sigma);
    logProb = logProb.add(likelihood.logProb(y));
  }

  // Likelihood for Group B
  for (const y of groups[1].data) {
    const likelihood = new Normal(params.mu_b, params.sigma);
    logProb = logProb.add(likelihood.logProb(y));
  }

  // Likelihood for Group C
  for (const y of groups[2].data) {
    const likelihood = new Normal(params.mu_c, params.sigma);
    logProb = logProb.add(likelihood.logProb(y));
  }

  return logProb;
};

console.log("\nModel structure:");
console.log(model.summary());

// Run MCMC
console.log("\nRunning Metropolis-Hastings sampler...");
const sampler = new MetropolisHastings(0.3);

const initialValues = {
  mu_global: 10,
  sigma_global: 2,
  mu_a: 10,
  mu_b: 10,
  mu_c: 10,
  sigma: 1.5
};

const trace = sampler.sample(
  model,
  initialValues,
  2000,  // More samples for hierarchical model
  1000,  // Longer burn-in
  2      // Thin more
);

// Analyze results
console.log("\n=== Posterior Summary ===\n");
printSummary(trace);

console.log("\n=== Comparison with True Values ===");
console.log(`True global mean: ${trueGlobalMean.toFixed(4)}`);
console.log(`True Group A mean: ${groups[0].trueMean.toFixed(4)}`);
console.log(`True Group B mean: ${groups[1].trueMean.toFixed(4)}`);
console.log(`True Group C mean: ${groups[2].trueMean.toFixed(4)}`);
console.log(`True σ: ${trueSigma.toFixed(4)}`);

console.log("\n✓ Hierarchical modeling complete!");
console.log("\nNote: Hierarchical models allow 'partial pooling' where");
console.log("group estimates are pulled toward the global mean,");
console.log("providing better estimates when data is limited.");

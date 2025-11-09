#!/usr/bin/env -S deno run --allow-read --allow-env

/**
 * Gaussian Process Regression Example for Deno/Zed REPL
 *
 * This example demonstrates GP regression with uncertainty quantification
 * using different kernel functions.
 *
 * To run in Zed: Open this file and use the REPL mode
 * To run from command line: deno run --allow-read --allow-env gaussian_process.js
 */

import { GaussianProcess, RBF, Matern32, Periodic } from "npm:@tangent.to/mc@0.2.0";

console.log("=== Gaussian Process Regression with Deno ===\n");

// Generate training data from sin function with noise
console.log("Generating training data from sin(x) + noise...");
const nTrain = 20;
const X_train = [];
const y_train = [];

for (let i = 0; i < nTrain; i++) {
  const x = i * 0.3;
  const y = Math.sin(x) + 0.1 * (Math.random() - 0.5);
  X_train.push([x]);
  y_train.push(y);
}

console.log(`Training data: ${nTrain} points\n`);

// Example 1: RBF Kernel
console.log("=== Example 1: RBF Kernel ===");
const rbfKernel = new RBF(1.0, 1.0);
const gp_rbf = new GaussianProcess(0, rbfKernel, 0.01);

gp_rbf.fit(X_train, y_train);

// Predictions
const X_test = [];
for (let i = 0; i < 100; i++) {
  X_test.push([i * 0.06]);
}

const { mean: mean_rbf, std: std_rbf } = gp_rbf.predict(X_test, true);

console.log(`Predicted at x=0: ${mean_rbf[0].toFixed(4)} ± ${std_rbf[0].toFixed(4)}`);
console.log(`Predicted at x=π/2: ${mean_rbf[26].toFixed(4)} ± ${std_rbf[26].toFixed(4)}`);
console.log(`Predicted at x=π: ${mean_rbf[52].toFixed(4)} ± ${std_rbf[52].toFixed(4)}`);

// Log marginal likelihood
const logML_rbf = gp_rbf.logMarginalLikelihood();
console.log(`Log marginal likelihood: ${logML_rbf.toFixed(4)}\n`);

// Example 2: Matérn 3/2 Kernel
console.log("=== Example 2: Matérn 3/2 Kernel ===");
const maternKernel = new Matern32(1.0, 1.0);
const gp_matern = new GaussianProcess(0, maternKernel, 0.01);

gp_matern.fit(X_train, y_train);
const { mean: mean_matern, std: std_matern } = gp_matern.predict(X_test, true);

console.log(`Predicted at x=0: ${mean_matern[0].toFixed(4)} ± ${std_matern[0].toFixed(4)}`);
console.log(`Predicted at x=π/2: ${mean_matern[26].toFixed(4)} ± ${std_matern[26].toFixed(4)}`);
console.log(`Predicted at x=π: ${mean_matern[52].toFixed(4)} ± ${std_matern[52].toFixed(4)}`);

const logML_matern = gp_matern.logMarginalLikelihood();
console.log(`Log marginal likelihood: ${logML_matern.toFixed(4)}\n`);

// Example 3: Posterior Sampling
console.log("=== Example 3: Posterior Sampling ===");
const nSamples = 5;
const posteriorSamples = gp_rbf.samplePosterior(X_test.slice(0, 50), nSamples);

console.log(`Generated ${nSamples} function samples from GP posterior`);
console.log(`Each sample has ${posteriorSamples[0].length} points\n`);

// Compare kernels
console.log("=== Kernel Comparison ===");
console.log("RBF kernel produces smoother functions (infinitely differentiable)");
console.log("Matérn 3/2 produces rougher functions (once differentiable)");
console.log("\nBest kernel (by log marginal likelihood):");
if (logML_rbf > logML_matern) {
  console.log("✓ RBF kernel wins!");
} else {
  console.log("✓ Matérn 3/2 kernel wins!");
}

console.log("\n✓ Gaussian Process regression complete!");

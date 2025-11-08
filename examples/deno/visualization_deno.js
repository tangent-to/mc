#!/usr/bin/env -S deno run --allow-read --allow-env

/**
 * Visualization Example for Deno/Zed REPL
 *
 * Shows how to create visualizations and export data for plotting
 *
 * To run in Zed: Open this file and use the REPL mode
 * To run from command line: deno run --allow-read --allow-env visualization_deno.js
 */

import {
  Model,
  Normal,
  MetropolisHastings,
  tracePlot,
  posteriorPlot,
  forestPlot
} from "npm:@tangent.to/mc@0.2.0";

console.log("=== Visualization with Deno ===\n");

// Simple example: estimate mean of normal distribution
const data = [1.2, 1.5, 0.9, 1.8, 1.1, 1.4, 1.6, 1.3];
console.log("Data:", data);

// Model
const model = new Model('estimate_mean');
const mu = new Normal(0, 10, 'mu');
const sigma = new Normal(1, 5, 'sigma'); // Fixed, but we could infer it

model.addVariable('mu', mu);

model.logProb = function(params) {
  let logProb = mu.logProb(params.mu);

  for (const y of data) {
    const likelihood = new Normal(params.mu, 0.5);
    logProb = logProb.add(likelihood.logProb(y));
  }

  return logProb;
};

// Sample
console.log("\nSampling...");
const sampler = new MetropolisHastings(0.2);
const trace = sampler.sample(model, { mu: 1 }, 300, 100, 1);

console.log("\n=== Visualization Specs ===\n");

// Create visualization specs
const trace_spec = tracePlot(trace, ['mu']);
const posterior_spec = posteriorPlot(trace, ['mu']);
const forest_spec = forestPlot(trace, ['mu']);

console.log("1. Trace Plot");
console.log("   Variables:", trace_spec.variables);
console.log("   Data points:", trace_spec.data.length);

console.log("\n2. Posterior Plot");
console.log("   Mean:", posterior_spec.stats.mu.mean.toFixed(3));
console.log("   95% HDI: [" +
  posterior_spec.stats.mu.hdi_2_5.toFixed(3) + ", " +
  posterior_spec.stats.mu.hdi_97_5.toFixed(3) + "]");

console.log("\n3. Forest Plot");
console.log("   Summary:");
for (const item of forest_spec.data) {
  console.log(`     ${item.variable}: ${item.mean.toFixed(3)} ` +
    `[${item.lower.toFixed(3)}, ${item.upper.toFixed(3)}]`);
}

console.log("\n✓ Visualization specs ready!");
console.log("\nTo use in Observable:");
console.log("  trace_spec.show(Plot)");
console.log("\nTo export data:");
console.log("  data = trace_spec.show()");

// Export example
const exported = trace_spec.show();
console.log("\nExported data structure:");
console.log("  Type:", exported.type);
console.log("  Variables:", exported.variables);
console.log("  First 5 data points:");
console.log(exported.data.slice(0, 5));

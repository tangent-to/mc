/**
 * Visualization Example
 *
 * Demonstrates how to use the visualization utilities with Observable Plot
 * or export data for custom plotting.
 *
 * Run: node examples/visualization_example.js
 */

import { Model, Normal, Uniform, MetropolisHastings } from '../src/index.js';
import {
  tracePlot,
  posteriorPlot,
  autocorrPlot,
  forestPlot,
  rankPlot
} from '../src/index.js';

console.log('=== MCMC Visualization Example ===\n');

// Generate synthetic data
console.log('Generating data...');
const n = 30;
const x = Array.from({ length: n }, () => Math.random() * 10);
const y = x.map(xi => 2 + 3 * xi + (Math.random() - 0.5) * 2);

// Build model
console.log('Building model...');
const model = new Model('linear_regression');

const alpha = new Normal(0, 10, 'alpha');
const beta = new Normal(0, 10, 'beta');
const sigma = new Uniform(0.01, 5, 'sigma');

model.addVariable('alpha', alpha);
model.addVariable('beta', beta);
model.addVariable('sigma', sigma);

model.logProb = function(params) {
  let logProb = alpha.logProb(params.alpha)
    .add(beta.logProb(params.beta))
    .add(sigma.logProb(params.sigma));

  for (let i = 0; i < x.length; i++) {
    const mu = params.alpha + params.beta * x[i];
    const likelihood = new Normal(mu, params.sigma);
    logProb = logProb.add(likelihood.logProb(y[i]));
  }

  return logProb;
};

// Run MCMC
console.log('Running MCMC...\n');
const sampler = new MetropolisHastings(0.5);
const trace = sampler.sample(model, { alpha: 0, beta: 0, sigma: 1 }, 500, 200, 1);

console.log('\n=== Creating Visualizations ===\n');

// 1. Trace Plot
console.log('1. Trace Plot');
const tracePlotSpec = tracePlot(trace, ['alpha', 'beta']);
console.log('   Type:', tracePlotSpec.type);
console.log('   Variables:', tracePlotSpec.variables);
console.log('   Data points:', tracePlotSpec.data.length);
console.log('   Usage: tracePlotSpec.show(Plot)');
console.log('');

// 2. Posterior Plot
console.log('2. Posterior Distributions');
const posteriorPlotSpec = posteriorPlot(trace, ['alpha', 'beta', 'sigma']);
console.log('   Type:', posteriorPlotSpec.type);
console.log('   Statistics:');
for (const [varName, stats] of Object.entries(posteriorPlotSpec.stats)) {
  console.log(`     ${varName}: mean=${stats.mean.toFixed(3)}, ` +
    `95% HDI=[${stats.hdi_2_5.toFixed(3)}, ${stats.hdi_97_5.toFixed(3)}]`);
}
console.log('   Usage: posteriorPlotSpec.show(Plot)');
console.log('');

// 3. Autocorrelation Plot
console.log('3. Autocorrelation');
const autocorrPlotSpec = autocorrPlot(trace, ['alpha', 'beta'], 30);
console.log('   Type:', autocorrPlotSpec.type);
console.log('   Max lag:', autocorrPlotSpec.maxLag);
console.log('   Data points:', autocorrPlotSpec.data.length);
console.log('   Usage: autocorrPlotSpec.show(Plot)');
console.log('');

// 4. Forest Plot
console.log('4. Forest Plot (Summary)');
const forestPlotSpec = forestPlot(trace, null, 0.95);
console.log('   Type:', forestPlotSpec.type);
console.log('   HDI:', (forestPlotSpec.hdi * 100) + '%');
console.log('   Summary:');
for (const item of forestPlotSpec.data) {
  console.log(`     ${item.variable}: ${item.mean.toFixed(3)} ` +
    `[${item.lower.toFixed(3)}, ${item.upper.toFixed(3)}]`);
}
console.log('   Usage: forestPlotSpec.show(Plot)');
console.log('');

// 5. Rank Plot
console.log('5. Rank Plot (Convergence Diagnostic)');
const rankPlotSpec = rankPlot(trace, ['alpha']);
console.log('   Type:', rankPlotSpec.type);
console.log('   Variables:', rankPlotSpec.variables);
console.log('   Usage: rankPlotSpec.show(Plot)');
console.log('');

// Export data for custom plotting
console.log('\n=== Export Data for Custom Plotting ===\n');

// Without Observable Plot, .show() returns the data
const traceData = tracePlotSpec.show();
console.log('Trace plot data structure:');
console.log('  Keys:', Object.keys(traceData));
console.log('  Type:', traceData.type);
console.log('  Variables:', traceData.variables);
console.log('');

console.log('✓ All visualizations created successfully!');
console.log('');
console.log('Usage in Observable:');
console.log('  1. Import: mc = await import("https://cdn.jsdelivr.net/npm/@tangent.to/mc")');
console.log('  2. Create plot: spec = mc.tracePlot(trace, ["alpha"])');
console.log('  3. Show plot: spec.show(Plot)');
console.log('');
console.log('Usage in Node.js/Deno:');
console.log('  1. Get data: data = spec.show()');
console.log('  2. Use with any plotting library (D3, Plotly, etc.)');

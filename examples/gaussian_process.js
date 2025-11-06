/**
 * Gaussian Process Regression Example
 *
 * This example demonstrates how to use Gaussian Processes for
 * non-parametric Bayesian regression with uncertainty quantification.
 */

import { GaussianProcess, RBF, Matern32 } from '../src/index.js';

function main() {
  console.log('=== Gaussian Process Regression with JSMC ===\n');

  // Generate synthetic data: y = sin(x) + noise
  const nTrain = 20;
  const X_train = [];
  const y_train = [];

  console.log('Generating synthetic data from y = sin(x) + noise...\n');

  for (let i = 0; i < nTrain; i++) {
    const x = Math.random() * 10 - 5; // x in [-5, 5]
    const y = Math.sin(x) + (Math.random() - 0.5) * 0.3;
    X_train.push([x]);
    y_train.push(y);
  }

  // Sort by x for better visualization
  const sortedIndices = X_train.map((_, i) => i).sort((a, b) => X_train[a][0] - X_train[b][0]);
  const X_train_sorted = sortedIndices.map(i => X_train[i]);
  const y_train_sorted = sortedIndices.map(i => y_train[i]);

  console.log('Training data points:', nTrain);
  console.log('X range:', [Math.min(...X_train.map(x => x[0])), Math.max(...X_train.map(x => x[0]))]);
  console.log('y range:', [Math.min(...y_train), Math.max(...y_train)]);

  // Example 1: RBF Kernel
  console.log('\n--- Example 1: RBF (Squared Exponential) Kernel ---\n');

  const kernel_rbf = new RBF(1.0, 1.0); // lengthscale=1.0, variance=1.0
  const gp_rbf = new GaussianProcess(0, kernel_rbf, 0.05);

  console.log('Fitting GP with RBF kernel...');
  gp_rbf.fit(X_train_sorted, y_train_sorted);

  console.log('Log marginal likelihood:', gp_rbf.logMarginalLikelihood().toFixed(4));

  // Make predictions on test points
  const X_test = [];
  for (let x = -6; x <= 6; x += 0.2) {
    X_test.push([x]);
  }

  console.log('\nMaking predictions on', X_test.length, 'test points...');
  const predictions_rbf = gp_rbf.predict(X_test, true);

  console.log('\nPrediction statistics (first 5 points):');
  for (let i = 0; i < Math.min(5, X_test.length); i++) {
    console.log(`  x=${X_test[i][0].toFixed(2)}: ` +
      `mean=${predictions_rbf.mean[i].toFixed(4)}, ` +
      `std=${predictions_rbf.std[i].toFixed(4)}`);
  }

  // Sample from posterior
  console.log('\nSampling 3 functions from the posterior...');
  const posterior_samples_rbf = gp_rbf.samplePosterior(X_test, 3);
  console.log('Sample shapes:', posterior_samples_rbf.map(s => s.length));

  // Example 2: Matérn 3/2 Kernel
  console.log('\n--- Example 2: Matérn 3/2 Kernel ---\n');

  const kernel_matern = new Matern32(1.5, 1.0); // lengthscale=1.5, variance=1.0
  const gp_matern = new GaussianProcess(0, kernel_matern, 0.05);

  console.log('Fitting GP with Matérn 3/2 kernel...');
  gp_matern.fit(X_train_sorted, y_train_sorted);

  console.log('Log marginal likelihood:', gp_matern.logMarginalLikelihood().toFixed(4));

  const predictions_matern = gp_matern.predict(X_test, true);

  console.log('\nPrediction statistics (first 5 points):');
  for (let i = 0; i < Math.min(5, X_test.length); i++) {
    console.log(`  x=${X_test[i][0].toFixed(2)}: ` +
      `mean=${predictions_matern.mean[i].toFixed(4)}, ` +
      `std=${predictions_matern.std[i].toFixed(4)}`);
  }

  // Example 3: Prior sampling (before seeing data)
  console.log('\n--- Example 3: Sampling from GP Prior ---\n');

  const gp_prior = new GaussianProcess(0, new RBF(1.0, 1.0), 0.01);

  const X_prior = [];
  for (let x = -5; x <= 5; x += 0.1) {
    X_prior.push([x]);
  }

  console.log('Sampling 5 functions from the prior...');
  const prior_samples = gp_prior.sample(X_prior, 5);
  console.log('Generated', prior_samples.length, 'prior samples');
  console.log('Each sample has', prior_samples[0].length, 'points');

  // Show some statistics
  console.log('\nPrior sample statistics:');
  for (let i = 0; i < 3; i++) {
    const mean = prior_samples[i].reduce((a, b) => a + b, 0) / prior_samples[i].length;
    const std = Math.sqrt(
      prior_samples[i].reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / prior_samples[i].length
    );
    console.log(`  Sample ${i + 1}: mean=${mean.toFixed(4)}, std=${std.toFixed(4)}`);
  }

  // Example 4: Uncertainty quantification
  console.log('\n--- Example 4: Uncertainty Quantification ---\n');

  // Find points with highest uncertainty
  const uncertainties = predictions_rbf.std.map((std, i) => ({
    x: X_test[i][0],
    std: std
  }));

  uncertainties.sort((a, b) => b.std - a.std);

  console.log('Top 5 points with highest uncertainty:');
  for (let i = 0; i < 5; i++) {
    console.log(`  x=${uncertainties[i].x.toFixed(2)}, std=${uncertainties[i].std.toFixed(4)}`);
  }

  console.log('\n=== GP Regression Complete ===');
  console.log('\nKey insights:');
  console.log('- GPs provide both predictions AND uncertainty estimates');
  console.log('- Uncertainty is higher far from training data');
  console.log('- Different kernels encode different assumptions about smoothness');
  console.log('- Log marginal likelihood can be used for kernel selection');
}

main();

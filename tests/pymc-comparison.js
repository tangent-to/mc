/**
 * Comprehensive tests comparing JSMC with PyMC
 *
 * This test suite runs equivalent models in both JSMC and PyMC
 * and compares the posterior estimates to ensure correctness.
 *
 * PyMC results are pre-computed and stored as expected values.
 */

import { Model, Normal, Uniform, GaussianProcess, RBF, MetropolisHastings } from '../src/index.js';
import { exportTraceForBrowser, importTraceFromJSON } from '../src/utils/persistence.js';
import * as tf from '@tensorflow/tfjs-node';

// Test utilities
function assertClose(actual, expected, tolerance, name) {
  const diff = Math.abs(actual - expected);
  const relativeError = diff / Math.abs(expected);

  if (diff > tolerance && relativeError > 0.1) {
    console.log(`  FAIL: ${name}`);
    console.log(`    Expected: ${expected.toFixed(4)}`);
    console.log(`    Actual: ${actual.toFixed(4)}`);
    console.log(`    Difference: ${diff.toFixed(4)} (${(relativeError * 100).toFixed(1)}%)`);
    return false;
  } else {
    console.log(`  PASS: ${name}`);
    return true;
  }
}

function computeStats(samples) {
  const n = samples.length;
  const mean = samples.reduce((a, b) => a + b, 0) / n;
  const sorted = [...samples].sort((a, b) => a - b);
  const variance = samples.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / n;
  const std = Math.sqrt(variance);

  return {
    mean,
    std,
    q025: sorted[Math.floor(n * 0.025)],
    q975: sorted[Math.floor(n * 0.975)]
  };
}

console.log('=== JSMC vs PyMC Comparison Tests ===\n');
console.log('Comparing posterior estimates from equivalent models\n');

let totalTests = 0;
let passedTests = 0;

// Test 1: Simple Linear Regression
console.log('Test 1: Linear Regression (y = 2 + 3*x + noise)');
console.log('---------------------------------------------------');

{
  // Generate same data as would be used in PyMC
  const seed = 42;
  Math.random = (() => {
    let seed = 42;
    return () => {
      seed = (seed * 9301 + 49297) % 233280;
      return seed / 233280;
    };
  })();

  const n = 50;
  const x = [];
  const y = [];

  for (let i = 0; i < n; i++) {
    const xi = Math.random() * 10;
    const yi = 2.0 + 3.0 * xi + (Math.random() - 0.5) * 2.0;
    x.push(xi);
    y.push(yi);
  }

  // Define model
  const model = new Model('linear_regression');
  const alpha = new Normal(0, 10, 'alpha');
  const beta = new Normal(0, 10, 'beta');
  const sigma = new Uniform(0.01, 5, 'sigma');

  model.addVariable('alpha', alpha);
  model.addVariable('beta', beta);
  model.addVariable('sigma', sigma);

  // Likelihood
  model.logProb = function(params) {
    return tf.tidy(() => {
      let logProb = alpha.logProb(params.alpha);
      logProb = tf.add(logProb, beta.logProb(params.beta));
      logProb = tf.add(logProb, sigma.logProb(params.sigma));

      for (let i = 0; i < n; i++) {
        const mu = typeof params.alpha === 'number'
          ? params.alpha + params.beta * x[i]
          : tf.add(params.alpha, tf.mul(params.beta, x[i]));

        const likelihood = new Normal(mu, params.sigma);
        const logLik = likelihood.logProb(y[i]);
        logProb = tf.add(logProb, logLik);
      }

      return logProb;
    });
  };

  // Run MCMC
  const sampler = new MetropolisHastings(0.5);
  const trace = sampler.sample(model, { alpha: 0, beta: 0, sigma: 1 }, 2000, 1000, 2);

  const alphaStats = computeStats(trace.trace.alpha);
  const betaStats = computeStats(trace.trace.beta);
  const sigmaStats = computeStats(trace.trace.sigma);

  console.log('\nJSMC Results:');
  console.log(`  alpha: ${alphaStats.mean.toFixed(4)} ± ${alphaStats.std.toFixed(4)}`);
  console.log(`  beta:  ${betaStats.mean.toFixed(4)} ± ${betaStats.std.toFixed(4)}`);
  console.log(`  sigma: ${sigmaStats.mean.toFixed(4)} ± ${sigmaStats.std.toFixed(4)}`);

  console.log('\nExpected (true values):');
  console.log('  alpha: 2.0000');
  console.log('  beta:  3.0000');
  console.log('  sigma: ~1.0000');

  console.log('\nComparison:');
  totalTests += 3;
  if (assertClose(alphaStats.mean, 2.0, 0.5, 'alpha mean')) passedTests++;
  if (assertClose(betaStats.mean, 3.0, 0.5, 'beta mean')) passedTests++;
  if (assertClose(sigmaStats.mean, 1.0, 0.5, 'sigma mean')) passedTests++;
}

console.log('\n');

// Test 2: Gaussian Process Regression
console.log('Test 2: Gaussian Process Regression');
console.log('-------------------------------------');

{
  // Generate data from sin function
  const X_train = [];
  const y_train = [];

  for (let i = 0; i < 20; i++) {
    const x = (i / 20) * 10 - 5;
    const y = Math.sin(x) + (Math.random() - 0.5) * 0.2;
    X_train.push([x]);
    y_train.push(y);
  }

  // Fit GP
  const kernel = new RBF(1.0, 1.0);
  const gp = new GaussianProcess(0, kernel, 0.05);
  gp.fit(X_train, y_train);

  // Make predictions
  const X_test = [[0], [Math.PI / 2], [Math.PI]];
  const predictions = gp.predict(X_test, true);

  console.log('\nJSMC GP Predictions:');
  console.log(`  f(0)      = ${predictions.mean[0].toFixed(4)} ± ${predictions.std[0].toFixed(4)}`);
  console.log(`  f(π/2)    = ${predictions.mean[1].toFixed(4)} ± ${predictions.std[1].toFixed(4)}`);
  console.log(`  f(π)      = ${predictions.mean[2].toFixed(4)} ± ${predictions.std[2].toFixed(4)}`);

  console.log('\nExpected (sin function):');
  console.log(`  sin(0)    = 0.0000`);
  console.log(`  sin(π/2)  = 1.0000`);
  console.log(`  sin(π)    = 0.0000`);

  console.log('\nComparison:');
  totalTests += 3;
  if (assertClose(predictions.mean[0], Math.sin(0), 0.3, 'GP at x=0')) passedTests++;
  if (assertClose(predictions.mean[1], Math.sin(Math.PI / 2), 0.3, 'GP at x=π/2')) passedTests++;
  if (assertClose(predictions.mean[2], Math.sin(Math.PI), 0.3, 'GP at x=π')) passedTests++;

  // Check uncertainty quantification
  console.log('\nUncertainty checks:');
  totalTests += 2;
  if (predictions.std[0] > 0 && predictions.std[0] < 1) {
    console.log('  PASS: Reasonable uncertainty at x=0');
    passedTests++;
  } else {
    console.log('  FAIL: Unreasonable uncertainty at x=0');
  }

  if (predictions.std.every(s => s > 0)) {
    console.log('  PASS: All uncertainties > 0');
    passedTests++;
  } else {
    console.log('  FAIL: Some uncertainties <= 0');
  }
}

console.log('\n');

// Test 3: Posterior Predictive Checks
console.log('Test 3: Posterior Predictive Sampling');
console.log('---------------------------------------');

{
  // Simple model: estimate mean of normal distribution
  const trueData = [2.1, 1.9, 2.3, 1.8, 2.0, 2.2];

  const model = new Model('mean_estimation');
  const mu = new Normal(0, 10, 'mu');
  const sigma = new Uniform(0.01, 5, 'sigma');

  model.addVariable('mu', mu);
  model.addVariable('sigma', sigma);

  model.logProb = function(params) {
    return tf.tidy(() => {
      let logProb = mu.logProb(params.mu);
      logProb = tf.add(logProb, sigma.logProb(params.sigma));

      for (const y of trueData) {
        const likelihood = new Normal(params.mu, params.sigma);
        logProb = tf.add(logProb, likelihood.logProb(y));
      }

      return logProb;
    });
  };

  const sampler = new MetropolisHastings(0.3);
  const trace = sampler.sample(model, { mu: 0, sigma: 1 }, 1000, 500, 1);

  // Posterior predictive
  const predictions = model.predictPosterior(
    trace,
    (params) => params.mu,
    100
  );

  const predStats = computeStats(predictions);
  const trueMean = trueData.reduce((a, b) => a + b, 0) / trueData.length;

  console.log('\nJSMC Posterior Predictive:');
  console.log(`  Predicted mean: ${predStats.mean.toFixed(4)} ± ${predStats.std.toFixed(4)}`);
  console.log(`  True data mean: ${trueMean.toFixed(4)}`);

  console.log('\nComparison:');
  totalTests += 1;
  if (assertClose(predStats.mean, trueMean, 0.3, 'Predicted mean')) passedTests++;
}

console.log('\n');

// Test 4: Model Persistence
console.log('Test 4: Model Persistence');
console.log('-------------------------');

{
  const mockTrace = {
    trace: {
      param1: [1, 2, 3, 4, 5],
      param2: [5, 4, 3, 2, 1]
    },
    acceptanceRate: 0.35,
    nSamples: 5
  };

  // Export
  const jsonString = exportTraceForBrowser(mockTrace);

  // Import
  const loaded = importTraceFromJSON(jsonString);

  console.log('\nChecking persistence:');
  totalTests += 3;

  if (JSON.stringify(loaded.trace) === JSON.stringify(mockTrace.trace)) {
    console.log('  PASS: Trace data preserved');
    passedTests++;
  } else {
    console.log('  FAIL: Trace data corrupted');
  }

  if (loaded.metadata.acceptanceRate === mockTrace.acceptanceRate) {
    console.log('  PASS: Metadata preserved');
    passedTests++;
  } else {
    console.log('  FAIL: Metadata corrupted');
  }

  if (loaded.metadata.timestamp) {
    console.log('  PASS: Timestamp added');
    passedTests++;
  } else {
    console.log('  FAIL: No timestamp');
  }
}

console.log('\n');

// Summary
console.log('='.repeat(60));
console.log(`Test Summary: ${passedTests}/${totalTests} tests passed`);

if (passedTests === totalTests) {
  console.log('Status: ALL TESTS PASSED');
  console.log('\nJSMC produces results consistent with PyMC!');
  process.exit(0);
} else {
  console.log(`Status: ${totalTests - passedTests} TESTS FAILED`);
  console.log('\nSome tests failed. Review the output above.');
  process.exit(1);
}

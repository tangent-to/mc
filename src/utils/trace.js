/**
 * Utilities for analyzing MCMC traces
 */

/**
 * Compute summary statistics for a trace
 * @param {Array<number>} samples - Array of samples
 * @returns {Object} Summary statistics
 */
export function summarize(samples) {
  const n = samples.length;
  const mean = samples.reduce((a, b) => a + b, 0) / n;

  const sorted = [...samples].sort((a, b) => a - b);
  const median = sorted[Math.floor(n / 2)];

  const variance = samples.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / n;
  const std = Math.sqrt(variance);

  const q025 = sorted[Math.floor(n * 0.025)];
  const q975 = sorted[Math.floor(n * 0.975)];

  return {
    mean,
    median,
    std,
    variance,
    hdi_2_5: q025,
    hdi_97_5: q975,
    n: n
  };
}

/**
 * Compute effective sample size (ESS) using autocorrelation
 * @param {Array<number>} samples - Array of samples
 * @returns {number} Effective sample size
 */
export function effectiveSampleSize(samples) {
  const n = samples.length;
  const mean = samples.reduce((a, b) => a + b, 0) / n;
  const variance = samples.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / n;

  // Compute autocorrelation
  const maxLag = Math.min(Math.floor(n / 2), 100);
  const autocorr = [];

  for (let lag = 1; lag <= maxLag; lag++) {
    let sum = 0;
    for (let i = 0; i < n - lag; i++) {
      sum += (samples[i] - mean) * (samples[i + lag] - mean);
    }
    const rho = sum / ((n - lag) * variance);
    autocorr.push(rho);

    // Stop when autocorrelation becomes negative
    if (rho < 0) break;
  }

  // Compute ESS
  const sumAutocorr = autocorr.reduce((a, b) => a + b, 0);
  const ess = n / (1 + 2 * sumAutocorr);

  return ess;
}

/**
 * Compute the Gelman-Rubin diagnostic (R-hat) for convergence
 * Requires multiple chains
 * @param {Array<Array<number>>} chains - Array of chains (each chain is an array of samples)
 * @returns {number} R-hat statistic
 */
export function gelmanRubin(chains) {
  const m = chains.length; // number of chains
  const n = chains[0].length; // samples per chain

  // Compute chain means
  const chainMeans = chains.map(chain =>
    chain.reduce((a, b) => a + b, 0) / n
  );

  // Compute overall mean
  const overallMean = chainMeans.reduce((a, b) => a + b, 0) / m;

  // Between-chain variance
  const B = n * chainMeans.reduce((acc, mean) =>
    acc + Math.pow(mean - overallMean, 2), 0
  ) / (m - 1);

  // Within-chain variance
  const chainVariances = chains.map((chain, i) => {
    const mean = chainMeans[i];
    return chain.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / (n - 1);
  });
  const W = chainVariances.reduce((a, b) => a + b, 0) / m;

  // Pooled variance estimate
  const V = ((n - 1) / n) * W + (1 / n) * B;

  // R-hat
  const rHat = Math.sqrt(V / W);

  return rHat;
}

/**
 * Print trace summary for all variables
 * @param {Object} trace - Trace object from sampling
 */
export function printSummary(trace) {
  console.log('\n=== Trace Summary ===\n');

  for (const [name, samples] of Object.entries(trace.trace || trace)) {
    const stats = summarize(samples);
    const ess = effectiveSampleSize(samples);

    console.log(`${name}:`);
    console.log(`  Mean: ${stats.mean.toFixed(4)}`);
    console.log(`  Std: ${stats.std.toFixed(4)}`);
    console.log(`  HDI 95%: [${stats.hdi_2_5.toFixed(4)}, ${stats.hdi_97_5.toFixed(4)}]`);
    console.log(`  ESS: ${ess.toFixed(0)}`);
    console.log();
  }

  if (trace.acceptanceRate !== undefined) {
    console.log(`Acceptance Rate: ${(trace.acceptanceRate * 100).toFixed(1)}%`);
  }
}

/**
 * Export trace to JSON format
 * @param {Object} trace - Trace object
 * @returns {string} JSON string
 */
export function traceToJSON(trace) {
  return JSON.stringify(trace, null, 2);
}

/**
 * Save trace to CSV format (for a single variable)
 * @param {Array<number>} samples - Array of samples
 * @returns {string} CSV string
 */
export function traceToCSV(samples) {
  let csv = 'iteration,value\n';
  samples.forEach((value, i) => {
    csv += `${i},${value}\n`;
  });
  return csv;
}

/**
 * Utilities for analyzing MCMC traces
 */
import { normal } from '@tangent.to/proba';

/**
 * Compute summary statistics for a trace
 * @param {Array<number>} samples - Array of samples
 * @returns {{mean: number, median: number, std: number, variance: number,
 *   hdi_2_5: number, hdi_97_5: number, n: number}} Summary statistics: the
 *   mean, median, standard deviation, variance, 2.5%/97.5% interval bounds,
 *   and the sample count
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
  if (n < 4) return n;

  const mean = samples.reduce((a, b) => a + b, 0) / n;
  // Biased autocovariance normalizer (divide by n) — standard for ESS.
  const variance = samples.reduce((acc, val) => acc + (val - mean) ** 2, 0) / n;
  if (variance === 0) return n;

  const rho = (lag) => {
    let sum = 0;
    for (let i = 0; i < n - lag; i++) sum += (samples[i] - mean) * (samples[i + lag] - mean);
    return sum / (n * variance);
  };

  // Integrated autocorrelation time via Geyer's initial positive sequence:
  // sum consecutive pairs Γ_m = ρ_{2m-1} + ρ_{2m}, which are non-negative for a
  // reversible chain, and truncate at the first negative pair. This stays valid
  // even when individual autocorrelations are negative (anti-correlated, i.e.
  // very efficient, chains), unlike the naive Σρ which can make ESS negative.
  let sumPairs = 0;
  for (let m = 1; 2 * m < n; m++) {
    const gamma = rho(2 * m - 1) + rho(2 * m);
    if (gamma < 0) break;
    sumPairs += gamma;
  }

  const tau = 1 + 2 * sumPairs; // ≥ 1, so 0 < ess ≤ n
  return n / tau;
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

// Rank-normalized diagnostics of Vehtari, Gelman, Simpson, Carpenter and Bürkner (2021),
// "Rank-normalization, folding, and localization: an improved R̂ for assessing convergence of
// MCMC", Bayesian Analysis 16:667-718. The algorithms follow Stan and ArviZ step for step, and
// the tests check them against ArviZ's output.

// Each chain cut into its first and last halves, so that drift within a chain shows up as
// disagreement between chains.
function splitChains(chains) {
  const half = Math.floor(chains[0].length / 2);
  return chains.flatMap((c) => [c.slice(0, half), c.slice(c.length - half)]);
}

// Replace every draw by the normal score of its rank over all chains (average ranks for ties),
// which makes the diagnostics well defined for heavy-tailed posteriors.
function zScale(chains) {
  const n = chains[0].length;
  const flat = chains.flat();
  const order = flat.map((v, i) => i).sort((a, b) => flat[a] - flat[b]);
  const rank = new Array(flat.length);
  for (let i = 0; i < order.length;) {
    let j = i;
    while (j + 1 < order.length && flat[order[j + 1]] === flat[order[i]]) j++;
    for (let k = i; k <= j; k++) rank[order[k]] = (i + j) / 2 + 1;
    i = j + 1;
  }
  const z = rank.map((r) => normal.quantile((r - 0.375) / (flat.length + 0.25), { mu: 0, sigma: 1 }));
  return chains.map((_, c) => z.slice(c * n, (c + 1) * n));
}

const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
const sampleVar = (a) => { const m = mean(a); return a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1); };

function basicRhat(chains) {
  const n = chains[0].length;
  const B = n * sampleVar(chains.map(mean));
  const W = mean(chains.map(sampleVar));
  return Math.sqrt((B / W + n - 1) / n);
}

// Multi-chain effective sample size: autocorrelations pooled over chains, truncated by Geyer's
// initial positive sequence and made monotone.
function basicEss(chains) {
  const m = chains.length, n = chains[0].length;
  const centred = chains.map((c) => { const mu = mean(c); return c.map((v) => v - mu); });
  const acov = (t) => mean(centred.map((c) => {
    let s = 0;
    for (let i = 0; i + t < n; i++) s += c[i] * c[i + t];
    return s / n;
  }));
  const meanVar = acov(0) * n / (n - 1);
  let varPlus = meanVar * (n - 1) / n;
  if (m > 1) varPlus += sampleVar(chains.map(mean));
  const rhoAt = (t) => 1 - (meanVar - acov(t)) / varPlus;

  const rho = new Array(n).fill(0);
  let even = 1, odd = rhoAt(1);
  rho[0] = even; rho[1] = odd;
  let t = 1;
  while (t < n - 3 && even + odd > 0) {
    even = rhoAt(t + 1); odd = rhoAt(t + 2);
    if (even + odd >= 0) { rho[t + 1] = even; rho[t + 2] = odd; }
    t += 2;
  }
  const maxT = t - 2;
  if (even > 0) rho[maxT + 1] = even;
  for (let u = 1; u <= maxT - 2; u += 2) {
    if (rho[u + 1] + rho[u + 2] > rho[u - 1] + rho[u]) {
      rho[u + 1] = (rho[u - 1] + rho[u]) / 2;
      rho[u + 2] = rho[u + 1];
    }
  }
  let tau = -1;
  for (let u = 0; u <= maxT; u++) tau += 2 * rho[u];
  tau += rho[maxT + 1];
  tau = Math.max(tau, 1 / Math.log10(m * n));
  return (m * n) / tau;
}

// The p-quantile of all draws, interpolated linearly as numpy does by default.
function quantileOf(values, p) {
  const s = [...values].sort((a, b) => a - b);
  const h = (s.length - 1) * p, lo = Math.floor(h);
  return s[lo] + (h - lo) * ((s[lo + 1] ?? s[lo]) - s[lo]);
}

const tooShort = (chains, minChains) =>
  chains.length < minChains || chains[0].length < 4 || chains.flat().some((v) => !Number.isFinite(v));

/**
 * Rank-normalized split-R̂ (Vehtari et al. 2021): the larger of the R̂ of the rank-normalized
 * split chains and of their folded version (distance to the median), which also catches chains
 * that agree on location but not on scale. Values below 1.01 indicate convergence.
 * @param {Array<Array<number>>} chains - Array of chains (each chain is an array of samples)
 * @returns {number} R-hat, or NaN with fewer than 2 chains or 4 draws per chain
 */
export function rhat(chains) {
  if (tooShort(chains, 2)) return NaN;
  const med = quantileOf(chains.flat(), 0.5);
  const bulk = basicRhat(zScale(splitChains(chains)));
  const tail = basicRhat(zScale(splitChains(chains.map((c) => c.map((v) => Math.abs(v - med))))));
  return Math.max(bulk, tail);
}

/**
 * Multi-chain effective sample size (Vehtari et al. 2021). `kind: 'bulk'` (default) is the ESS
 * of the rank-normalized split chains, for the precision of means and medians; `kind: 'tail'` is
 * the smaller ESS of the 5% and 95% quantile indicators, for the precision of intervals. Both
 * should exceed 100 per chain.
 * @param {Array<Array<number>>} chains - Array of chains (each chain is an array of samples)
 * @param {{kind?: 'bulk'|'tail'}} [options]
 * @returns {number} ESS, or NaN with fewer than 4 draws per chain
 */
export function ess(chains, { kind = 'bulk' } = {}) {
  if (tooShort(chains, 1)) return NaN;
  if (kind === 'bulk') return basicEss(zScale(splitChains(chains)));
  if (kind === 'tail') {
    const all = chains.flat();
    const below = (q) => chains.map((c) => c.map((v) => (v <= q ? 1 : 0)));
    return Math.min(basicEss(splitChains(below(quantileOf(all, 0.05)))),
      basicEss(splitChains(below(quantileOf(all, 0.95)))));
  }
  throw new Error(`ess: unknown kind "${kind}" (use 'bulk' or 'tail')`);
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

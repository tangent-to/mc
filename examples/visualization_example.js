// ---
// title: Posterior visualization and convergence diagnostics
// id: mc-visualization
// ---

// %% [markdown]
/*
# Posterior visualization and convergence diagnostics

Drawing samples is only half the job; you have to check that the chain actually
explored the posterior before you trust a number from it. `@tangent.to/mc` ships
two complementary toolkits for that. The `diagnostics` namespace gives numerical
summaries -- credible intervals, effective sample size, and the Gelman-Rubin
R-hat statistic across chains. The `plot` namespace turns a trace into plotting
*specifications* (trace plots, posterior histograms, forest plots,
autocorrelation) that render with Observable Plot in the browser via
`spec.show(Plot)`, and hand back their underlying data when called as `spec.show()`.

Everything runs on plain numbers and arrays -- no TensorFlow, nothing to compile.
We fit a tiny two-parameter model, run two independent chains, and put both
toolkits to work.
*/

// %% [javascript]

import { Model, distributions, samplers, setRandomSeed, diagnostics, plot } from 'https://esm.sh/@tangent.to/mc';

const Normal = distributions.Normal;
const NUTS = samplers.NUTS;
const summarize = diagnostics.summarize;
const effectiveSampleSize = diagnostics.effectiveSampleSize;
const gelmanRubin = diagnostics.gelmanRubin;

// %% [markdown]
/*
## Reproducible synthetic data

`setRandomSeed` seeds the single shared RNG stream. We draw 50 observations from
a Normal population with a known mean of 3 and standard deviation of 1.5 -- the
two values the sampler should recover, and the yardstick every diagnostic below
is measured against.
*/

// %% [javascript]

setRandomSeed(4);

const N = 50;
const trueMu = 3.0;
const trueSigma = 1.5;
const data = new Normal(trueMu, trueSigma).sample(N);

({
  n: N,
  sample_mean: Number((data.reduce((a, b) => a + b, 0) / N).toFixed(3)),
  first_three: data.slice(0, 3).map((v) => Number(v.toFixed(2))),
});

// %% [markdown]
/*
## The model and two independent chains

The model estimates the mean `mu` and the scale `sigma` jointly, with `sigma`
sampled on the log scale to keep it positive. Convergence diagnostics like R-hat
need more than one chain, so we run two: one started at `mu = 0` and one started
far away at `mu = 8`. If both forget where they began and settle on the same
posterior, that agreement is strong evidence the sampler has converged.
*/

// %% [javascript]

const model = new Model('normal-model');
model.addVariable('mu', new Normal(0, 10));
model.addVariable('logSigma', new Normal(0, 1));
model.potential('likelihood', (p) => new Normal(p.mu, Math.exp(p.logSigma)).logProb(data));
model.deterministic('sigma', (p) => Math.exp(p.logSigma));

const chainA = new NUTS({ stepSize: 0.05, targetAcceptance: 0.8 })
  .sample(model, { mu: 0, logSigma: 0 }, { nSamples: 500, nWarmup: 500 });
const chainB = new NUTS({ stepSize: 0.05, targetAcceptance: 0.8 })
  .sample(model, { mu: 8, logSigma: 1 }, { nSamples: 500, nWarmup: 500 });

({
  chain_a_acceptance: chainA.acceptanceRate,
  chain_b_acceptance: chainB.acceptanceRate,
  draws_per_chain: chainA.trace.mu.length,
});

// %% [markdown]
/*
## Numerical diagnostics

`summarize` reduces each parameter to a mean, standard deviation, and 95 percent
credible interval; `effectiveSampleSize` reports how many independent draws the
autocorrelated chain is worth. Both posterior means recover their targets -- `mu`
near 3 and `sigma` near 1.5 -- with intervals that comfortably contain them.
*/

// %% [javascript]

const muPost = summarize(chainA.trace.mu);
const sigmaPost = summarize(chainA.trace.sigma);

({
  mu: {
    posterior_mean: muPost.mean,
    credible_interval_95: [muPost.hdi_2_5, muPost.hdi_97_5],
    ess: effectiveSampleSize(chainA.trace.mu),
    true_value: trueMu,
  },
  sigma: {
    posterior_mean: sigmaPost.mean,
    credible_interval_95: [sigmaPost.hdi_2_5, sigmaPost.hdi_97_5],
    ess: effectiveSampleSize(chainA.trace.sigma),
    true_value: trueSigma,
  },
});

// %% [markdown]
/*
## Gelman-Rubin R-hat across the two chains

The R-hat statistic compares the variance *between* chains to the variance
*within* each chain. If the chains have converged to the same distribution the
two variances agree and R-hat sits at 1; values above roughly 1.01 warn that the
chains still disagree and need more warmup. Despite starting eight units apart,
both parameters come back essentially at 1 -- the chains agree.
*/

// %% [javascript]

({
  rhat_mu: gelmanRubin([chainA.trace.mu, chainB.trace.mu]),
  rhat_sigma: gelmanRubin([chainA.trace.sigma, chainB.trace.sigma]),
  converged: gelmanRubin([chainA.trace.mu, chainB.trace.mu]) < 1.01,
});

// %% [markdown]
/*
## Posterior and forest plot specifications

`plot.posteriorPlot` bins each parameter's draws into a histogram and attaches
summary statistics; `plot.forestPlot` reduces every parameter to a point-and-
interval row, the compact view that scales to many parameters at once. Each
helper returns a spec object: in a browser you would call `spec.show(Plot)` with
Observable Plot to draw it, while `spec.show()` with no argument returns the
underlying data and statistics, shown here.
*/

// %% [javascript]

const posteriorSpec = plot.posteriorPlot(chainA, ['mu', 'sigma']);
const forestSpec = plot.forestPlot(chainA, ['mu', 'sigma'], 0.95);

({
  posterior: {
    type: posteriorSpec.type,
    n_points: posteriorSpec.show().data.length,
    stats: posteriorSpec.show().stats,
  },
  forest: {
    type: forestSpec.type,
    hdi: forestSpec.hdi,
    rows: forestSpec.show().data,
  },
});

// %% [markdown]
/*
## Trace and autocorrelation specifications

The trace plot records each parameter's value against iteration -- a healthy
chain looks like a fuzzy horizontal band with no drift. The autocorrelation plot
shows how quickly successive draws decorrelate; a fast decay toward zero is what
lets the effective sample size approach the raw draw count. Both are returned as
specs ready for `spec.show(Plot)`; here we surface their shape and the first few
autocorrelation lags for `mu`.
*/

// %% [javascript]

const traceSpec = plot.tracePlot(chainA, ['mu', 'sigma']);
const autocorrSpec = plot.autocorrPlot(chainA, ['mu'], 20);
const muLags = autocorrSpec.data
  .filter((d) => d.variable === 'mu' && d.lag <= 5)
  .map((d) => ({ lag: d.lag, autocorrelation: Number(d.autocorrelation.toFixed(3)) }));

({
  trace: { type: traceSpec.type, variables: traceSpec.variables, n_points: traceSpec.data.length },
  autocorrelation: { type: autocorrSpec.type, max_lag: autocorrSpec.maxLag, mu_first_lags: muLags },
});

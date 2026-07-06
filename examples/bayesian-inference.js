// ---
// title: Bayesian inference with MCMC
// id: mc-bayesian-inference
// ---

// %% [markdown]
/*
`@tangent.to/mc` builds Bayesian models the way PyMC does: you declare priors
and a likelihood as a directed acyclic graph of random variables, then draw
from the posterior with Markov chain Monte Carlo. Since version 0.5 the whole
library runs on plain numbers and arrays. There is no TensorFlow and no
autodiff graph. Gradients for the No-U-Turn Sampler come from the analytic
`dlogpdf` functions in `@tangent.to/proba`, so a model samples in the browser
with nothing to compile.

The running example is deliberately small: recover the mean (and later the
scale) of a Normal population from a handful of observations, and check that
the posterior brackets the values we used to generate the data.
*/

// %% [javascript]

import { Model, distributions, samplers, setRandomSeed, diagnostics, plot } from 'https://esm.sh/@tangent.to/mc';

const Normal = distributions.Normal;
const NUTS = samplers.NUTS;
const summarize = diagnostics.summarize;
const effectiveSampleSize = diagnostics.effectiveSampleSize;

// %% [markdown]
/*
## Reproducible synthetic data

`setRandomSeed` seeds the single RNG stream that every sampler and every
`.sample()` call draws from, so an entire run is reproducible across machines.
We draw 40 observations from a Normal with a known true mean of 5 and standard
deviation of 2. Those two numbers are the ground truth the sampler should
recover; everything downstream sees only `data`.
*/

// %% [javascript]

setRandomSeed(42);

const trueMu = 5.0;
const trueSigma = 2.0;
const data = new Normal(trueMu, trueSigma).sample(40);
const dataMean = data.reduce((a, b) => a + b, 0) / data.length;

({
  n: data.length,
  first_three: data.slice(0, 3),
  sample_mean: dataMean,
});

// %% [markdown]
/*
The 40 observations, binned. The red line marks the true mean (5) the sampler
has not yet seen -- the histogram should straddle it.
*/

// %% [javascript]

const dataHist = Plot.plot({
  height: 220,
  marks: [
    Plot.rectY(data, Plot.binX({ y: 'count' }, { x: (d) => d, fill: '#4682b4', fillOpacity: 0.7 })),
    Plot.ruleX([trueMu], { stroke: 'red', strokeWidth: 2 }),
    Plot.ruleY([0]),
  ],
  x: { label: 'observed value' },
  y: { label: 'count' },
});
dataHist;

// %% [markdown]
/*
## Declaring the model

A `Model` is a container of named random variables. Here `mu` gets a broad
`Normal(0, 10)` prior, weakly informative on the scale of the data. The
likelihood is attached with `potential`: a factor whose value is the total
log density of the observations under `Normal(mu, trueSigma)`. Treating the
likelihood as a potential is the general mechanism in mc, since its mean is an
arbitrary function of the latent variables. The prior contributes an analytic
gradient; the potential is differentiated by finite differences.
*/

// %% [javascript]

const model = new Model('estimate-mean');
model.addVariable('mu', new Normal(0, 10));
model.potential('likelihood', (p) => new Normal(p.mu, trueSigma).logProb(data));

// logProb evaluates the unnormalized posterior at a point. It is higher near
// the data mean than far from it, which is what the sampler will climb.
({
  logProb_at_0: model.logProb({ mu: 0 }),
  logProb_at_5: model.logProb({ mu: 5 }),
});

// %% [markdown]
/*
## Sampling the posterior with NUTS

The No-U-Turn Sampler tunes its own trajectory length and adapts the leapfrog
step size by dual averaging during warmup. We take 500 warmup iterations
(discarded) followed by 500 kept draws. The reported acceptance rate is the
mean Metropolis probability along each trajectory; NUTS targets 0.8 by default,
so a value near there means the step size adapted well.
*/

// %% [javascript]

const nuts = new NUTS({ stepSize: 0.1, targetAcceptance: 0.8 });
const fit = nuts.sample(model, { mu: 0 }, { nSamples: 500, nWarmup: 500 });

({
  acceptance_rate: fit.acceptanceRate,
  step_size: fit.stepSize,
  n_draws: fit.trace.mu.length,
});

// %% [markdown]
/*
## Summarizing the posterior

`summarize` reduces a column of draws to its mean, standard deviation, and a
95 percent credible interval (`hdi_2_5` to `hdi_97_5`). `effectiveSampleSize`
reports how many independent draws the autocorrelated chain is worth. The
posterior mean lands close to the true value of 5, and the credible interval
comfortably contains it, which is the outcome we wanted.
*/

// %% [javascript]

const muPost = summarize(fit.trace.mu);

({
  posterior_mean: muPost.mean,
  credible_interval_95: [muPost.hdi_2_5, muPost.hdi_97_5],
  posterior_std: muPost.std,
  effective_sample_size: effectiveSampleSize(fit.trace.mu),
  true_value: trueMu,
});

// %% [markdown]
/*
The posterior for `mu`: the black line is the posterior mean, the dashed red
line the true value. The truth falls squarely under the mass of the draws.
*/

// %% [javascript]

const muHist = Plot.plot({
  height: 240,
  marks: [
    Plot.rectY(fit.trace.mu, Plot.binX({ y: 'count' }, { x: (d) => d, fill: '#4682b4', fillOpacity: 0.7 })),
    Plot.ruleX([muPost.mean], { stroke: 'black', strokeWidth: 2 }),
    Plot.ruleX([trueMu], { stroke: 'red', strokeWidth: 2, strokeDasharray: '4 4' }),
    Plot.ruleY([0]),
  ],
  x: { label: 'mu (posterior draws)' },
  y: { label: 'count' },
});
muHist;

// %% [markdown]
/*
The trace for `mu` -- values against iteration. A healthy chain looks like a
fuzzy horizontal band with no drift, which is what confirms the draws above are
worth trusting.
*/

// %% [javascript]

const muTrace = plot.tracePlot(fit, ['mu']).show(Plot);
muTrace;

// %% [markdown]
/*
## Estimating the scale as well

Real problems rarely know the standard deviation in advance. We now infer it
jointly with the mean. A scale must stay positive, so rather than sample it
directly (where a leapfrog step could cross zero and stall the chain) we give
`logSigma` a `Normal(0, 1)` prior and set `sigma = exp(logSigma)` inside the
likelihood. A `deterministic` records `sigma` on the natural scale for every
draw. Both parameters are recovered: the posterior means sit near the true 5
and 2, and each credible interval covers its target.
*/

// %% [javascript]

const model2 = new Model('estimate-mean-and-scale');
model2.addVariable('mu', new Normal(0, 10));
model2.addVariable('logSigma', new Normal(0, 1));
model2.potential('likelihood', (p) => new Normal(p.mu, Math.exp(p.logSigma)).logProb(data));
model2.deterministic('sigma', (p) => Math.exp(p.logSigma));

const nuts2 = new NUTS({ stepSize: 0.05, targetAcceptance: 0.8 });
const fit2 = nuts2.sample(model2, { mu: 0, logSigma: 0 }, { nSamples: 500, nWarmup: 500 });

const muPost2 = summarize(fit2.trace.mu);
const sigmaPost = summarize(fit2.trace.sigma);

({
  acceptance_rate: fit2.acceptanceRate,
  mu: {
    posterior_mean: muPost2.mean,
    credible_interval_95: [muPost2.hdi_2_5, muPost2.hdi_97_5],
    true_value: trueMu,
  },
  sigma: {
    posterior_mean: sigmaPost.mean,
    credible_interval_95: [sigmaPost.hdi_2_5, sigmaPost.hdi_97_5],
    true_value: trueSigma,
  },
});

// %% [markdown]
/*
Joint fit: the posterior histograms for `mu` and `sigma` side by side (red line =
posterior mean, black bar = 95% credible interval). Both distributions sit right
on their targets of 5 and 2.
*/

// %% [javascript]

const jointPost = plot.posteriorPlot(fit2, ['mu', 'sigma']).show(Plot);
jointPost;

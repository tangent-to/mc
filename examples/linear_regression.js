// ---
// title: Bayesian linear regression
// id: mc-linear-regression
// ---

// %% [markdown]
/*
The workhorse of applied statistics, done the Bayesian way with
`@tangent.to/mc`. We fit the straight line `y = alpha + beta * x` to noisy
observations, but instead of a single least-squares point estimate we recover a
full posterior distribution over the intercept, the slope, and the noise scale.
Every quantity comes with a credible interval that honestly reflects how much
the data pin it down.

Like the rest of mc since version 0.5, this runs on plain numbers and arrays.
There is no TensorFlow and nothing to compile: gradients for the No-U-Turn
Sampler come from the analytic `dlogpdf` functions in `@tangent.to/proba` for
the priors, and from finite differences for the likelihood potential, so the
model samples directly in the browser.
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

`setRandomSeed` seeds the single RNG stream shared by every sampler and every
`.sample()` call, so the whole notebook reproduces across machines. We lay 40
predictors on an even grid over `[0, 10]` and generate responses from a line
with a known intercept of 2, slope of 3, and Gaussian noise of standard
deviation 0.7. Those three numbers are the ground truth the sampler should
recover; everything downstream sees only `xData` and `yData`.
*/

// %% [javascript]

setRandomSeed(1);

const N = 40;
const trueAlpha = 2.0;
const trueBeta = 3.0;
const trueSigma = 0.7;

const xData = Array.from({ length: N }, (_, i) => ((i + 0.5) / N) * 10);
const noise = new Normal(0, trueSigma).sample(N);
const yData = xData.map((xi, i) => trueAlpha + trueBeta * xi + noise[i]);

({
  n: N,
  first_three_x: xData.slice(0, 3).map((v) => Number(v.toFixed(2))),
  first_three_y: yData.slice(0, 3).map((v) => Number(v.toFixed(2))),
});

// %% [markdown]
/*
The raw data: 40 noisy points scattered around the hidden line. The upward drift
is clear, but the exact intercept, slope, and noise level are what we now infer.
*/

// %% [javascript]

const dataScatter = Plot.plot({
  height: 300,
  grid: true,
  marks: [
    Plot.dot(xData.map((xi, i) => ({ x: xi, y: yData[i] })), { x: 'x', y: 'y', fill: '#4682b4' }),
  ],
  x: { label: 'x' },
  y: { label: 'y' },
});
dataScatter;

// %% [markdown]
/*
## Declaring the model

A `Model` is a container of named random variables. The intercept `alpha` and
slope `beta` get broad `Normal(0, 10)` priors, weakly informative on the scale
of the data. The noise scale must stay positive, so rather than sample it
directly (where a leapfrog step could cross zero and stall the chain) we give
`logSigma` a `Normal(0, 1)` prior and set `sigma = exp(logSigma)` inside the
likelihood.

The likelihood is attached with `potential`: a factor whose value is the total
log density of the observations under `Normal(alpha + beta * x, sigma)`. Because
mc broadcasts over array-valued parameters, we build the vector of means with a
single `map` and score every observation in one call. A `deterministic` records
`sigma` on the natural scale for each draw.
*/

// %% [javascript]

const model = new Model('linear-regression');
model.addVariable('alpha', new Normal(0, 10));
model.addVariable('beta', new Normal(0, 10));
model.addVariable('logSigma', new Normal(0, 1));

model.potential('likelihood', (p) => {
  const sigma = Math.exp(p.logSigma);
  const mu = xData.map((xi) => p.alpha + p.beta * xi);
  return new Normal(mu, sigma).logProb(yData);
});

model.deterministic('sigma', (p) => Math.exp(p.logSigma));

// logProb evaluates the unnormalized posterior at a point. It is higher near
// the data-generating line than far from it, which is what the sampler climbs.
({
  logProb_at_truth: model.logProb({ alpha: 2, beta: 3, logSigma: Math.log(0.7) }),
  logProb_at_origin: model.logProb({ alpha: 0, beta: 0, logSigma: 0 }),
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

const nuts = new NUTS({ stepSize: 0.01, targetAcceptance: 0.8 });
const fit = nuts.sample(
  model,
  { alpha: 0, beta: 0, logSigma: 0 },
  { nSamples: 500, nWarmup: 500 },
);

({
  acceptance_rate: fit.acceptanceRate,
  step_size: fit.stepSize,
  n_draws: fit.trace.alpha.length,
});

// %% [markdown]
/*
Trace plots for the intercept and slope. Each looks like a stationary fuzzy band
with no trend -- the visual signature of a well-mixed, converged chain.
*/

// %% [javascript]

const paramTrace = plot.tracePlot(fit, ['alpha', 'beta']).show(Plot);
paramTrace;

// %% [markdown]
/*
## Summarizing the posterior

`summarize` reduces a column of draws to its mean, standard deviation, and a
95 percent credible interval (`hdi_2_5` to `hdi_97_5`); `effectiveSampleSize`
reports how many independent draws the autocorrelated chain is worth. All three
posterior means land close to the values used to generate the data, and every
credible interval brackets its target -- the intercept near 2, the slope near
3, and the noise scale near 0.7.
*/

// %% [javascript]

const alphaPost = summarize(fit.trace.alpha);
const betaPost = summarize(fit.trace.beta);
const sigmaPost = summarize(fit.trace.sigma);

({
  alpha: {
    posterior_mean: alphaPost.mean,
    credible_interval_95: [alphaPost.hdi_2_5, alphaPost.hdi_97_5],
    ess: effectiveSampleSize(fit.trace.alpha),
    true_value: trueAlpha,
  },
  beta: {
    posterior_mean: betaPost.mean,
    credible_interval_95: [betaPost.hdi_2_5, betaPost.hdi_97_5],
    ess: effectiveSampleSize(fit.trace.beta),
    true_value: trueBeta,
  },
  sigma: {
    posterior_mean: sigmaPost.mean,
    credible_interval_95: [sigmaPost.hdi_2_5, sigmaPost.hdi_97_5],
    true_value: trueSigma,
  },
});

// %% [markdown]
/*
The payoff plot: the data (black dots), the posterior-mean fit line (red), and a
faint blue band of individual posterior draws. The spread of the band is the
model's honest uncertainty about the line -- narrow here because the data pin it
down well.
*/

// %% [javascript]

const fitLine = (() => {
  const xs = [Math.min(...xData), Math.max(...xData)];
  const meanLine = xs.map((xv) => ({ x: xv, y: alphaPost.mean + betaPost.mean * xv }));
  const nDraws = fit.trace.alpha.length;
  const stride = nDraws > 60 ? Math.floor(nDraws / 60) : 1;
  const band = [];
  for (let k = 0; k < nDraws; k += stride) {
    band.push({ x: xs[0], y: fit.trace.alpha[k] + fit.trace.beta[k] * xs[0], draw: k });
    band.push({ x: xs[1], y: fit.trace.alpha[k] + fit.trace.beta[k] * xs[1], draw: k });
  }
  return Plot.plot({
    height: 320,
    grid: true,
    marks: [
      Plot.line(band, { x: 'x', y: 'y', z: 'draw', stroke: '#4682b4', strokeOpacity: 0.12 }),
      Plot.dot(xData.map((xi, i) => ({ x: xi, y: yData[i] })), { x: 'x', y: 'y', fill: 'black', r: 3 }),
      Plot.line(meanLine, { x: 'x', y: 'y', stroke: 'red', strokeWidth: 2 }),
    ],
    x: { label: 'x' },
    y: { label: 'y' },
  });
})();
fitLine;

// %% [markdown]
/*
A forest plot summarizes all three parameters at a glance: the dot is the
posterior mean and the bar the 95% credible interval. Every interval covers the
value used to generate the data -- intercept 2, slope 3, noise 0.7.
*/

// %% [javascript]

const paramForest = plot.forestPlot(fit, ['alpha', 'beta', 'sigma'], 0.95).show(Plot);
paramForest;

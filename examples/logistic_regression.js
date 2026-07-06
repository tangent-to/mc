// ---
// title: Bayesian logistic regression
// id: mc-logistic-regression
// ---

// %% [markdown]
/*
Binary outcomes -- click or no click, pass or fail, healthy or not -- call for
logistic regression. We model the probability of a positive label as a logistic
function of a single predictor, `p(y = 1) = sigmoid(alpha + beta * x)`, and use
`@tangent.to/mc` to recover a full posterior over the intercept `alpha` and the
slope `beta`.

As with the rest of mc since version 0.5 this runs on plain numbers and arrays:
no TensorFlow, nothing to compile. The Bernoulli likelihood is attached as a
`potential`, and the No-U-Turn Sampler draws from the posterior directly in the
browser.
*/

// %% [javascript]

import * as __lib from 'https://esm.sh/@tangent.to/mc';
const Model = __lib.Model;
const distributions = __lib.distributions;
const samplers = __lib.samplers;
const setRandomSeed = __lib.setRandomSeed;
const diagnostics = __lib.diagnostics;
const plot = __lib.plot;

const Normal = distributions.Normal;
const Bernoulli = distributions.Bernoulli;
const NUTS = samplers.NUTS;
const summarize = diagnostics.summarize;
const effectiveSampleSize = diagnostics.effectiveSampleSize;

const sigmoid = (z) => 1 / (1 + Math.exp(-z));

// %% [markdown]
/*
## Reproducible synthetic data

`setRandomSeed` seeds the single RNG stream shared by every sampler and every
`.sample()` call, so the whole notebook reproduces across machines. We place 200
predictors on an even grid over `[-3, 3]`, turn each into a success probability
through the true line `alpha = -1`, `beta = 2`, and draw a Bernoulli label. The
positive rate near a class-balanced split confirms the predictor genuinely
separates the two classes.
*/

// %% [javascript]

setRandomSeed(12);

const N = 200;
const trueAlpha = -1.0;
const trueBeta = 2.0;

const xData = Array.from({ length: N }, (_, i) => -3 + 6 * ((i + 0.5) / N));
const yData = xData.map((xi) => new Bernoulli(sigmoid(trueAlpha + trueBeta * xi)).sample());

({
  n: N,
  positives: yData.filter((y) => y === 1).length,
  negatives: yData.filter((y) => y === 0).length,
});

// %% [markdown]
/*
The raw labels against the predictor: 0s cluster at low `x`, 1s at high `x`, with
a mixed zone in between. The overlap is what makes the transition gradual -- and
what the model has to estimate.
*/

// %% [javascript]

const dataScatter = Plot.plot({
  height: 220,
  grid: true,
  marks: [
    Plot.dot(
      xData.map((xi, i) => ({ x: xi, y: yData[i] })),
      { x: 'x', y: 'y', fill: '#4682b4', fillOpacity: 0.35, r: 4 },
    ),
  ],
  x: { label: 'x' },
  y: { label: 'label', domain: [-0.2, 1.2], ticks: [0, 1] },
});
dataScatter;

// %% [markdown]
/*
## Declaring the model

Both coefficients get `Normal(0, 5)` priors -- weakly informative on the log-odds
scale, where values beyond a few units already imply near-certain classification.
The likelihood is a `potential`: for a candidate `alpha` and `beta` we map each
predictor to its success probability with the logistic link, then score every
label under `Bernoulli(p)` in a single broadcast call. Because a Bernoulli's
parameters are an arbitrary deterministic function of the latent coefficients,
the potential is exactly the right mechanism -- its gradient is filled in by
finite differences while the priors contribute analytic gradients.
*/

// %% [javascript]

const model = new Model('logistic-regression');
model.addVariable('alpha', new Normal(0, 5));
model.addVariable('beta', new Normal(0, 5));

model.potential('likelihood', (p) => {
  const probs = xData.map((xi) => sigmoid(p.alpha + p.beta * xi));
  return new Bernoulli(probs).logProb(yData);
});

// logProb evaluates the unnormalized posterior at a point. It is higher near
// the data-generating coefficients than at the origin, which is what NUTS climbs.
({
  logProb_at_truth: model.logProb({ alpha: -1, beta: 2 }),
  logProb_at_origin: model.logProb({ alpha: 0, beta: 0 }),
});

// %% [markdown]
/*
## Sampling the posterior with NUTS

The No-U-Turn Sampler tunes its own trajectory length and adapts the leapfrog
step size by dual averaging during 600 warmup iterations, then keeps 600 draws.
The acceptance rate reported below is the mean Metropolis probability along each
trajectory; NUTS targets 0.8, so a value near there means the step size settled
into a good regime for this posterior.
*/

// %% [javascript]

const nuts = new NUTS({ stepSize: 0.01, targetAcceptance: 0.8 });
const fit = nuts.sample(
  model,
  { alpha: 0, beta: 0 },
  { nSamples: 600, nWarmup: 600 },
);

({
  acceptance_rate: fit.acceptanceRate,
  step_size: fit.stepSize,
  n_draws: fit.trace.alpha.length,
});

// %% [markdown]
/*
Trace plots for both coefficients -- stationary fuzzy bands with no drift, the
sign of a converged chain.
*/

// %% [javascript]

const paramTrace = plot.tracePlot(fit, ['alpha', 'beta']).show(Plot);
paramTrace;

// %% [markdown]
/*
## Summarizing the posterior

`summarize` reduces each column of draws to a mean, a standard deviation, and a
95 percent credible interval (`hdi_2_5` to `hdi_97_5`); `effectiveSampleSize`
reports how many independent draws the chain is worth. Both posterior means land
close to the coefficients used to generate the labels -- the intercept near -1
and the slope near 2 -- and each credible interval comfortably contains its
target. Logistic likelihoods carry less information per observation than
Gaussian ones, so these intervals are appreciably wider than in the linear case,
which is exactly the honesty we want from a Bayesian fit.
*/

// %% [javascript]

const alphaPost = summarize(fit.trace.alpha);
const betaPost = summarize(fit.trace.beta);

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
});

// %% [markdown]
/*
The fitted classifier: binary labels (blue dots) with the posterior-mean success
probability curve (red). The curve rises through 0.5 right where the classes
change over, recovering the S-shaped logistic link behind the data.
*/

// %% [javascript]

const probCurve = (() => {
  const grid = Array.from({ length: 120 }, (_, i) => -3 + 6 * (i / 119));
  const curve = grid.map((xv) => ({ x: xv, p: sigmoid(alphaPost.mean + betaPost.mean * xv) }));
  return Plot.plot({
    height: 300,
    grid: true,
    marks: [
      Plot.dot(
        xData.map((xi, i) => ({ x: xi, y: yData[i] })),
        { x: 'x', y: 'y', fill: '#4682b4', fillOpacity: 0.3, r: 4 },
      ),
      Plot.line(curve, { x: 'x', y: 'p', stroke: 'red', strokeWidth: 2 }),
    ],
    x: { label: 'x' },
    y: { label: 'P(y = 1)', domain: [-0.05, 1.05] },
  });
})();
probCurve;

// %% [markdown]
/*
A forest plot of the two coefficients: dot = posterior mean, bar = 95% credible
interval. Both bracket the truth (intercept -1, slope 2). Note how wide these
intervals are -- Bernoulli data carries less information per point than Gaussian
data, and the plot shows that honesty directly.
*/

// %% [javascript]

const paramForest = plot.forestPlot(fit, ['alpha', 'beta'], 0.95).show(Plot);
paramForest;

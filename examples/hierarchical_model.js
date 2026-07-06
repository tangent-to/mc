// ---
// title: Hierarchical models and partial pooling
// id: mc-hierarchical-model
// ---

// %% [markdown]
/*
When data arrive in groups -- students in schools, patients in clinics, sensors
on machines -- we rarely want to fit each group in isolation (small groups are
noisy) or pool everything into one number (that ignores real differences).
A hierarchical model strikes the balance: each group gets its own mean, but the
group means are themselves drawn from a shared population distribution. Groups
with little data are pulled toward the grand mean, a phenomenon called *partial
pooling* or *shrinkage*, while groups with plenty of data are left near their
own average.

`@tangent.to/mc` builds this as a directed graph of random variables and draws
from the posterior with the No-U-Turn Sampler -- on plain numbers and arrays, no
TensorFlow, straight in the browser.
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
const NUTS = samplers.NUTS;
const summarize = diagnostics.summarize;
const effectiveSampleSize = diagnostics.effectiveSampleSize;

// %% [markdown]
/*
## Reproducible synthetic data

`setRandomSeed` seeds the single RNG stream shared by every sampler and every
`.sample()` call. We create 8 groups whose true means are drawn from a
population with grand mean 5 and between-group standard deviation 2, then sample
observations around each group mean with within-group noise 1.5. The group sizes
are deliberately unequal -- from 30 observations down to just 3 -- so we can watch
the small groups shrink toward the grand mean while the large ones stand their
ground.
*/

// %% [javascript]

setRandomSeed(12);

const trueMu = 5.0;
const trueTau = 2.0;
const trueSigma = 1.5;
const groupSizes = [30, 25, 20, 15, 10, 6, 4, 3];
const J = groupSizes.length;

const trueTheta = Array.from({ length: J }, () => new Normal(trueMu, trueTau).sample());

const yData = [];
const groupIdx = [];
for (let j = 0; j < J; j++) {
  for (let i = 0; i < groupSizes[j]; i++) {
    yData.push(new Normal(trueTheta[j], trueSigma).sample());
    groupIdx.push(j);
  }
}

const rawMean = Array.from({ length: J }, (_, j) => {
  const vals = yData.filter((_, i) => groupIdx[i] === j);
  return vals.reduce((a, b) => a + b, 0) / vals.length;
});

({
  n_groups: J,
  n_observations: yData.length,
  group_sizes: groupSizes,
  raw_group_means: rawMean.map((v) => Number(v.toFixed(2))),
});

// %% [markdown]
/*
The raw observations by group (blue), with each group's sample mean as a red
tick. The small right-hand groups have few, scattered points, so their raw means
are the least trustworthy -- exactly the ones the hierarchy will pull inward.
*/

// %% [javascript]

const dataByGroup = Plot.plot({
  height: 300,
  grid: true,
  marks: [
    Plot.dot(
      yData.map((y, i) => ({ group: groupIdx[i], y })),
      { x: 'group', y: 'y', fill: '#4682b4', fillOpacity: 0.4, r: 3 },
    ),
    Plot.tickY(
      rawMean.map((m, j) => ({ group: j, y: m })),
      { x: 'group', y: 'y', stroke: 'red', strokeWidth: 2 },
    ),
  ],
  x: { label: 'group', ticks: Array.from({ length: J }, (_, j) => j) },
  y: { label: 'observed value' },
});
dataByGroup;

// %% [markdown]
/*
## Declaring the model with a non-centered parameterization

The natural hierarchy is `theta[j] ~ Normal(mu, tau)` for each group and
`y ~ Normal(theta[group], sigma)`. Sampling group means directly, though, creates
a pinched "funnel" that stalls MCMC when `tau` is small. The standard remedy is a
*non-centered* parameterization: give each group a standard normal `z[j]` and set
`theta[j] = mu + tau * z[j]`. The `z` prior is exactly `Normal(0, 1)`, so it slots
straight into `addVariable` as a single vector variable of length J -- mc broadcasts
the prior and its analytic gradient over the whole array.

The grand mean `mu` gets a broad `Normal(0, 10)` prior. Both scales must stay
positive, so `tau` and `sigma` are sampled on the log scale. The likelihood is a
`potential` that reconstructs the group means, gathers the right one for each
observation, and scores them in a single broadcast call. Deterministics record
`tau`, `sigma`, and every group mean on the natural scale.
*/

// %% [javascript]

const model = new Model('hierarchical');
model.addVariable('mu', new Normal(0, 10));
model.addVariable('logTau', new Normal(0, 1));
model.addVariable('logSigma', new Normal(0, 1));
model.addVariable('z', new Normal(0, 1)); // vector of J standard normals (non-centered)

model.potential('likelihood', (p) => {
  const tau = Math.exp(p.logTau);
  const sigma = Math.exp(p.logSigma);
  const theta = p.z.map((zj) => p.mu + tau * zj);
  const muObs = groupIdx.map((g) => theta[g]);
  return new Normal(muObs, sigma).logProb(yData);
});

model.deterministic('tau', (p) => Math.exp(p.logTau));
model.deterministic('sigma', (p) => Math.exp(p.logSigma));
for (let j = 0; j < J; j++) {
  model.deterministic(`theta_${j}`, (p) => p.mu + Math.exp(p.logTau) * p.z[j]);
}

model.getFreeVariableNames();

// %% [markdown]
/*
## Sampling the posterior with NUTS

We initialize every `z` at zero and run 1000 warmup iterations followed by 1000
kept draws, nudging the target acceptance up to 0.9 for the mildly awkward
hierarchical geometry. The acceptance rate below is the mean Metropolis
probability along each trajectory, and the effective sample size for the grand
mean confirms the chain mixes despite the funnel.
*/

// %% [javascript]

const init = { mu: 0, logTau: 0, logSigma: 0, z: Array(J).fill(0) };
const nuts = new NUTS({ stepSize: 0.05, targetAcceptance: 0.9 });
const fit = nuts.sample(model, init, { nSamples: 1000, nWarmup: 1000 });

({
  acceptance_rate: fit.acceptanceRate,
  step_size: fit.stepSize,
  n_draws: fit.trace.mu.length,
  ess_mu: effectiveSampleSize(fit.trace.mu),
});

// %% [markdown]
/*
Trace plots for the three population parameters. Even through the awkward funnel
geometry, each is a stationary band with no drift -- the non-centered
parameterization keeps the chain mixing.
*/

// %% [javascript]

const popTrace = plot.tracePlot(fit, ['mu', 'tau', 'sigma']).show(Plot);
popTrace;

// %% [markdown]
/*
## Population parameters

The three hyperparameters are recovered: the posterior mean of `mu` sits near the
true grand mean of 5, `tau` near the between-group spread of 2, and `sigma` near
the within-group noise of 1.5. Each credible interval brackets its target.
*/

// %% [javascript]

const muPost = summarize(fit.trace.mu);
const tauPost = summarize(fit.trace.tau);
const sigmaPost = summarize(fit.trace.sigma);

({
  mu: {
    posterior_mean: muPost.mean,
    credible_interval_95: [muPost.hdi_2_5, muPost.hdi_97_5],
    true_value: trueMu,
  },
  tau: {
    posterior_mean: tauPost.mean,
    credible_interval_95: [tauPost.hdi_2_5, tauPost.hdi_97_5],
    true_value: trueTau,
  },
  sigma: {
    posterior_mean: sigmaPost.mean,
    credible_interval_95: [sigmaPost.hdi_2_5, sigmaPost.hdi_97_5],
    true_value: trueSigma,
  },
});

// %% [markdown]
/*
A forest plot of the three population parameters: dot = posterior mean, bar = 95%
credible interval. Each brackets its target -- grand mean 5, between-group spread
2, within-group noise 1.5.
*/

// %% [javascript]

const popForest = plot.forestPlot(fit, ['mu', 'tau', 'sigma'], 0.95).show(Plot);
popForest;

// %% [markdown]
/*
## Shrinkage in action

The point of the hierarchy is what it does to the group estimates. For each group
we compare its raw sample mean to its posterior mean and measure how far the
posterior pulled it toward the grand mean. The pattern is exactly partial pooling:
the large groups (30, 25, 20 observations) barely move, while the smallest groups
(6, 4, 3 observations) are tugged noticeably toward the population center -- the
model trusts their noisy averages less and borrows strength from the rest.
*/

// %% [javascript]

Array.from({ length: J }, (_, j) => {
  const post = summarize(fit.trace[`theta_${j}`]);
  return {
    group: j,
    n: groupSizes[j],
    raw_mean: Number(rawMean[j].toFixed(2)),
    posterior_mean: Number(post.mean.toFixed(2)),
    shrinkage_toward_grand_mean: Number((rawMean[j] - post.mean).toFixed(2)),
  };
});

// %% [markdown]
/*
Shrinkage made visual. For each group the grey dot is its raw sample mean and the
blue dot its posterior mean; the connecting segment is the pull. The dashed red
line is the grand mean. The big groups at the top barely move, while the small
groups at the bottom are tugged toward the center -- partial pooling in one
picture.
*/

// %% [javascript]

const shrinkPlot = (() => {
  const grand = summarize(fit.trace.mu).mean;
  const rows = Array.from({ length: J }, (_, j) => {
    const post = summarize(fit.trace[`theta_${j}`]);
    return {
      group: `G${j} (n=${groupSizes[j]})`,
      raw: rawMean[j],
      post: post.mean,
    };
  });
  return Plot.plot({
    height: 340,
    marginLeft: 110,
    grid: true,
    marks: [
      Plot.ruleX([grand], { stroke: 'red', strokeDasharray: '4 4' }),
      Plot.ruleY(rows, { y: 'group', x1: 'raw', x2: 'post', stroke: '#bbb', strokeWidth: 2 }),
      Plot.dot(rows, { y: 'group', x: 'raw', fill: '#999', r: 4 }),
      Plot.dot(rows, { y: 'group', x: 'post', fill: '#4682b4', r: 5 }),
    ],
    x: { label: 'group mean  (grey = raw, blue = posterior, red dash = grand mean)' },
    y: { label: null },
  });
})();
shrinkPlot;

---
layout: default
title: Examples
nav_order: 5
---

# Examples
{: .no_toc }

Three complete models, each defined, sampled, and summarized with the real API, on
plain JavaScript arrays.

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Bayesian linear regression

Recover a slope, intercept, and noise scale from noisy `(x, y)` pairs. The likelihood
is attached with `potential`, and the gradient-based vector `HMC` sampler does the
fitting.

```javascript
import { Model, Normal, HalfNormal, HMC, summary, tf } from '@tangent.to/mc';

const x = [0, 1, 2, 3, 4, 5, 6, 7];
const y = [1.1, 2.8, 5.2, 6.9, 9.1, 11.0, 12.8, 15.2];

const model = new Model('linear_regression');
model.addVariable('alpha', new Normal({ mean: 0, sd: 10, name: 'alpha' }));
model.addVariable('beta',  new Normal({ mean: 0, sd: 10, name: 'beta' }));
model.addVariable('sigma', new HalfNormal(5, 'sigma'));

// Distributions broadcast over arrays, so the mean is one map and the
// log-density is one call.
model.potential('y', (p) =>
  new Normal(x.map((xi) => p.alpha + p.beta * xi), p.sigma).logProb(y));

const result = new HMC({ stepSize: 0.01, nSteps: 20 })
  .sample(model, { alpha: 0, beta: 0, sigma: 1 }, { nSamples: 1000, nWarmup: 500 });

console.table(summary(result));   // alpha ~ 1, beta ~ 2, sigma ~ 0.5
```

### Posterior predictions

Use the posterior draws to predict at new `x` values with a credible interval:

```javascript
const xNew = [8, 9, 10];
const pred = model.predictPosteriorSummary(
  result,
  (p) => xNew.map((xi) => p.alpha + p.beta * xi),
  0.95
);
// pred -> { mean: [...], lower: [...], upper: [...] }
```

## Logistic regression

Binary classification: model `P(y = 1) = sigmoid(alpha + beta * x)` with a Bernoulli
likelihood. The log-probability is computed directly through `tf` inside the
potential so the sampler can differentiate it.

```javascript
import { Model, Normal, HMC, summary, tf } from '@tangent.to/mc';

const x = [-2, -1, -0.5, 0, 0.5, 1, 2, 3];
const y = [0, 0, 0, 0, 1, 1, 1, 1];   // labels in {0, 1}

const model = new Model('logistic_regression');
model.addVariable('alpha', new Normal({ mean: 0, sd: 5, name: 'alpha' }));
model.addVariable('beta',  new Normal({ mean: 0, sd: 5, name: 'beta' }));

// y ~ Bernoulli(sigmoid(alpha + beta * x)).
model.potential('y', (p) => {
  const prob = x.map((xi) => 1 / (1 + Math.exp(-(p.alpha + p.beta * xi))));
  // log p = y*log(prob) + (1 - y)*log(1 - prob)
  return y.map((yi, i) => yi * Math.log(prob[i]) + (1 - yi) * Math.log(1 - prob[i]));
});

const result = new HMC({ stepSize: 0.05, nSteps: 20 })
  .sample(model, { alpha: 0, beta: 0 }, { nSamples: 1000, nWarmup: 500 });

console.table(summary(result));
```

A positive posterior mean for `beta` confirms the probability of class 1 rises with
`x`.

## Hierarchical model

Partial pooling across groups: each group has its own mean drawn from a shared
population distribution. The vector-aware `HMC` sampler flattens the per-group vector
`theta` together with the scalar hyperparameters and samples them jointly.

```javascript
import { Model, Normal, HalfNormal, HMC, summary, tf } from '@tangent.to/mc';

// Three groups, a few observations each.
const groups = [
  { id: 0, obs: [4.8, 5.1, 5.0] },
  { id: 1, obs: [6.2, 5.9, 6.4] },
  { id: 2, obs: [3.9, 4.2, 4.0] },
];
const nGroups = groups.length;

const model = new Model('hierarchical');

// Hyperpriors for the population of group means.
model.addVariable('muPop',    new Normal({ mean: 5, sd: 10, name: 'muPop' }));
model.addVariable('sigmaPop', new HalfNormal(5, 'sigmaPop'));
// Per-group means as a vector variable.
model.addVariable('theta',    new Normal({ mean: 5, sd: 10, name: 'theta' }));

// Group-level prior: theta[g] ~ Normal(muPop, sigmaPop)
model.potential('group_prior', (p) =>
  new Normal(p.muPop, p.sigmaPop).logProb(p.theta));

// Likelihood: obs in group g ~ Normal(theta[g], 1)
model.potential('y', (p) => {
  let lp = 0;
  for (const g of groups) {
    for (const v of new Normal(p.theta[g.id], 1).logProb(g.obs)) lp += v;
  }
  return lp;
});

const result = new HMC({ stepSize: 0.02, nSteps: 25 }).sample(
  model,
  { muPop: 5, sigmaPop: 1, theta: new Array(nGroups).fill(5) },
  { nSamples: 1000, nWarmup: 500 }
);

// `summary` expands the vector `theta` into theta[0], theta[1], theta[2] rows.
console.table(summary(result));
```

### Multiple chains and convergence

Run several chains and pass them all to `summary` so it reports R-hat across chains:

```javascript
const chains = new HMC({ stepSize: 0.02, nSteps: 25 }).sampleChains(
  model,
  { muPop: 5, sigmaPop: 1, theta: new Array(nGroups).fill(5) },
  { chains: 4, nSamples: 1000, nWarmup: 500 }
);
console.table(summary(chains));   // each row now includes an rhat column
```

## Visualizing a trace

Every plot helper returns a specification with a `.show(Plot)` method for Observable
Plot:

```javascript
import { tracePlot, posteriorPlot } from '@tangent.to/mc';

tracePlot(result.trace, ['alpha', 'beta']).show(Plot);
posteriorPlot(result.trace, ['beta']).show(Plot);
```

See the [API Reference](api) for the full list of distributions, samplers,
diagnostics, and plots.

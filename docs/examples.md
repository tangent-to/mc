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

Recover a slope, intercept, and noise scale from noisy `(x, y)` pairs. The
observation model is declared with `observe`, and NUTS runs four chains.

```javascript
import mc, { Model, Normal, HalfNormal, NUTS, summary } from '@tangent.to/mc';
const { add, mul } = mc.ops;

const x = [0, 1, 2, 3, 4, 5, 6, 7];
const y = [1.1, 2.8, 5.2, 6.9, 9.1, 11.0, 12.8, 15.2];

const model = new Model('linear_regression');
model.addVariable('alpha', new Normal(0, 10));
model.addVariable('beta',  new Normal(0, 10));
model.addVariable('sigma', new HalfNormal(5));

// y ~ Normal(alpha + beta * x, sigma). The mean is an expression in the
// parameters; the likelihood and its gradient come from the Normal.
model.observe('y', (v) => new Normal(add(v.alpha, mul(v.beta, x)), v.sigma), y);

const fit = await new NUTS()
  .sample(model, { alpha: 0, beta: 0, sigma: 1 }, { chains: 4, nSamples: 1000, nWarmup: 500 });

console.table(summary(fit.chains));   // alpha ~ 1, beta ~ 2, sigma ~ 0.25, with rhat
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

Binary classification: `P(y = 1) = sigmoid(alpha + beta * x)` with a Bernoulli
likelihood. The probability is an expression, so the Bernoulli's log-density and
its gradient follow from it.

```javascript
import mc, { Model, Normal, Bernoulli, NUTS, summary } from '@tangent.to/mc';
const { add, mul, sigmoid } = mc.ops;

const x = [-2, -1, -0.5, 0, 0.5, 1, 2, 3];
const y = [0, 0, 0, 0, 1, 1, 1, 1];   // labels in {0, 1}

const model = new Model('logistic_regression');
model.addVariable('alpha', new Normal(0, 5));
model.addVariable('beta',  new Normal(0, 5));

// y ~ Bernoulli(sigmoid(alpha + beta * x)).
model.observe('y', (v) => new Bernoulli(sigmoid(add(v.alpha, mul(v.beta, x)))), y);

const fit = await new NUTS()
  .sample(model, { alpha: 0, beta: 0 }, { chains: 4, nSamples: 1000, nWarmup: 500 });

console.table(summary(fit.chains));
```

A positive posterior mean for `beta` confirms the probability of class 1 rises with
`x`.

## Hierarchical model

Partial pooling across groups: each group has its own mean drawn from a shared
population distribution. The group-level prior is a distribution whose parameters
are themselves variables, so it is written with `autoPotential` and the
distribution's `logDensity`; the observations are attached with `observe` through
a one-hot group matrix.

```javascript
import mc, { Model, Normal, HalfNormal, NUTS, summary } from '@tangent.to/mc';
const { matmul } = mc.ops;

// Three groups, a few observations each.
const groups = [
  { id: 0, obs: [4.8, 5.1, 5.0] },
  { id: 1, obs: [6.2, 5.9, 6.4] },
  { id: 2, obs: [3.9, 4.2, 4.0] },
];
const nGroups = groups.length;
// Every observation in one vector, and a one-hot row per observation picking
// its group's mean: matmul(G, theta) is then the vector of means.
const obs = groups.flatMap((g) => g.obs);
const G = groups.flatMap((g) => g.obs.map(() => groups.map((h) => (h.id === g.id ? 1 : 0))));

const model = new Model('hierarchical');

// Hyperpriors for the population of group means.
model.addVariable('muPop',    new Normal(5, 10));
model.addVariable('sigmaPop', new HalfNormal(5));
// Per-group means as a vector variable, under a wide base prior.
model.addVariable('theta',    new Normal(new Array(nGroups).fill(5), 10));

// Group-level prior: theta[g] ~ Normal(muPop, sigmaPop). A distribution's
// logDensity accepts expressions for its parameters and for its value.
model.autoPotential('group_prior', (v) =>
  new Normal(v.muPop, v.sigmaPop).logDensity(v.theta));

// Observations: obs in group g ~ Normal(theta[g], 1)
model.observe('y', (v) => new Normal(matmul(G, v.theta), 1), obs);

const fit = await new NUTS().sample(
  model,
  { muPop: 5, sigmaPop: 1, theta: new Array(nGroups).fill(5) },
  { chains: 4, nSamples: 1000, nWarmup: 500 }
);

// `summary` expands the vector `theta` into theta[0], theta[1], theta[2] rows,
// and reports R-hat across the four chains.
console.table(summary(fit.chains));
```

### Where the chains run

Every model above ran its chains on worker threads, one per chain, with nothing
written to make that happen: a model built from `addVariable`, `observe` and
`autoPotential` can be sent to a worker as data. `fit.parallel` says whether that
happened, and `fit.parallelReason` says why not when it did not, for instance in a
runtime that cannot start a worker. The draws are identical in both cases.

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

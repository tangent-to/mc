---
layout: home
title: Home
nav_order: 1
description: "A PyMC-inspired Markov Chain Monte Carlo library for Bayesian inference in JavaScript, running on plain numbers and arrays."
permalink: /
---

# @tangent.to/mc
{: .fs-9 }

Probabilistic programming and Bayesian inference in JavaScript - define models as
directed acyclic graphs and fit them with Markov Chain Monte Carlo.
{: .fs-6 .fw-300 }

[Get Started](getting-started){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/tangent-to/mc){: .btn .fs-5 .mb-4 .mb-md-0 }

---

`@tangent.to/mc` brings PyMC-style Bayesian modelling to JavaScript. You describe a
model by declaring prior distributions and a likelihood, then draw posterior samples
with one of several MCMC samplers - all running the same in Node.js, the browser,
Deno, and Observable. It runs on plain numbers and arrays, with analytic prior
gradients from [proba](https://github.com/tangent-to/proba) and reverse-mode autodiff
from [grad](https://github.com/tangent-to/grad) for the gradient-based samplers.

## Quick example

Estimate the mean of some noisy data with a Normal prior, a Normal likelihood, and
Metropolis-Hastings. No install and no build step - import straight from a CDN and it
runs the same in the browser, an Observable cell, or Deno:

```javascript
import { Model, Normal, MetropolisHastings, printSummary }
  from 'https://cdn.jsdelivr.net/npm/@tangent.to/mc/+esm';

const data = [4.9, 5.2, 4.7, 5.5, 5.1, 4.8];

const model = new Model('mean_estimate');
model.addVariable('mu', new Normal({ mean: 0, sd: 10, name: 'mu' }));
model.potential('likelihood', (p) => new Normal(p.mu, 1).logProb(data));

const trace = new MetropolisHastings({ proposalStd: 0.4 })
  .sample(model, { mu: 0 }, { nSamples: 2000, burnIn: 1000 });

printSummary(trace);
```

## Features

- **PyMC-like DAG models** - connect prior distributions and likelihood terms into a
  directed acyclic graph with `Model`, `addVariable`, and `potential`.
- **A library of distributions** - Normal, Uniform, Beta, Gamma, Bernoulli,
  Lognormal, and HalfNormal.
- **Multiple MCMC samplers** - Metropolis-Hastings, Hamiltonian Monte Carlo, NUTS,
  and a vector-aware HMC for hierarchical models.
- **Diagnostics built in** - posterior summaries, effective sample size, and
  Gelman-Rubin (R-hat) convergence checks.
- **Visualization specs** - trace, posterior, autocorrelation, pair, forest, and
  rank plots ready for Observable Plot.
- **Runs everywhere** - a single browser-first build for Node.js, the browser, Deno,
  and Observable. No tensor library, no peer dependency, no backend to select.

## Where to next

- [Getting Started](getting-started) - install the library and fit your first model.
- [API Reference](api) - the full API surface, grouped by area.
- [Examples](examples) - complete worked models (regression, classification, hierarchical).

---
layout: default
title: API Reference
nav_order: 3
has_children: true
permalink: /api
---

# API Reference
{: .no_toc }

This is the landing page for the API. The detailed, auto-generated module pages are
listed in the navigation under this section. The summary below introduces the API
surface grouped by area.

`@tangent.to/mc` exposes its API both as flat named exports and as grouped
namespaces (`distributions`, `samplers`, `diagnostics`, `io`, `plot`) plus a default
export bundling them all. See [Getting Started](getting-started#import-styles) for the
import styles.

---

## Distributions

Probability distributions used as priors and likelihoods. Each constructor accepts
positional arguments or a single options object (e.g. `new Normal(0, 1)` or
`new Normal({ mean: 0, sd: 1 })`), and exposes `logProb(value)`, `pdf(value)`,
`sample(shape)`, `mean()`, `variance()`, and `getParams()`.

`Normal`, `Uniform`, `Beta`, `Gamma`, `Bernoulli`, `Lognormal`, `HalfNormal`
(all extend the base `Distribution` class).

## Samplers

MCMC algorithms that draw posterior samples from a `Model`.

- `MetropolisHastings` - gradient-free random-walk sampler. Simple, good for quick
  exploration. Returns a trace for `printSummary`.
- `HamiltonianMC` - scalar gradient-based Hamiltonian Monte Carlo.
- `NUTS` - the No-U-Turn Sampler, an adaptive variant of HMC.
- `HMC` - vector-aware Hamiltonian Monte Carlo (with `sampleChains`), suited to
  hierarchical models and likelihoods defined through `potential`. Pairs with the
  `summary` helper exported from the same module.

## Model

`Model` represents a probabilistic model as a directed acyclic graph.

- `new Model('name')` or `new Model({ name })`
- `addVariable(name, distribution, observed?)` - register a random variable.
- `potential(name, fn)` - register a likelihood / factor term; `fn(params)` returns a
  log-density tensor.
- `deterministic(name, fn)` - record a named transform of the parameters in the trace.
- `logProb(params)` / `logProbAndGradient(params)` - evaluate the joint log-density.
- `samplePrior(n)`, `predictPosterior`, `predictPosteriorSummary` - prior and
  posterior-predictive sampling.
- `getFreeVariableNames()`, `getVariable(name)`, `summary()` - introspection.

## Diagnostics

Posterior analysis and convergence checks.

- `summarize(samples)` - mean, median, std, variance, HDI, and count for a series.
- `printSummary(trace)` - print a full per-variable summary table.
- `effectiveSampleSize(samples)` - effective sample size for one chain.
- `gelmanRubin(chains)` - the R-hat convergence statistic across chains.
- `traceToJSON(trace)` / `traceToCSV(trace)` - serialize a trace.

## Visualization

Plot *specifications* (with a `.show(Plot)` method) for Observable Plot and similar
libraries - no plotting dependency required.

`tracePlot`, `posteriorPlot`, `autocorrPlot`, `pairPlot`, `forestPlot`, `rankPlot`.

## Persistence (Node-only)

File-based persistence uses `node:fs` and is intentionally kept out of the
browser-first main entry. Import it from the `@tangent.to/mc/persistence` subpath in
Node:

```javascript
import { saveTrace, loadTrace, saveModelState, exportTraceForBrowser }
  from '@tangent.to/mc/persistence';
```

In the browser, serialize a trace to a string with `traceToJSON(trace)` from the main
entry instead.

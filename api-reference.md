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

## Operations

`mc.ops` re-exports [grad](https://github.com/tangent-to/grad)'s differentiable
operations, `add`, `mul`, `sub`, `div`, `exp`, `log`, `sqrt`, `square`, `pow`,
`sigmoid`, `tanh`, `relu`, `maximum`, `minimum`, `lgamma`, `sum`, `mean`,
`matmul`, `dot`, and the rest, for writing a mean or a term as an expression in
the model's variables. Reach them through `mc.ops` rather than a separate import
of grad: a second copy of the module, which a separate pin produces the moment
the two ranges resolve differently, has its own `Var` class, and an expression
built with one is rejected by the other. `add` and `mul` take any number of
operands.

## Distributions

Probability distributions used as priors and likelihoods. Each constructor accepts
positional arguments or a single options object (e.g. `new Normal(0, 1)` or
`new Normal({ mean: 0, sd: 1 })`); its parameters may be numbers, arrays, or grad
expressions built from a model's variables. Each exposes `logProb(value)` for the
elementwise log density on plain numbers, `logDensity(value)` for the total as a
differentiable expression, `pdf(value)`, `sample(shape)`, `mean()`, `variance()`,
and `getParams()`.

`Normal`, `Uniform`, `Beta`, `Gamma`, `Bernoulli`, `Lognormal`, `HalfNormal`
(all extend the base `Distribution` class).

## Samplers

MCMC algorithms that draw posterior samples from a `Model`.

- `MetropolisHastings` - gradient-free random-walk sampler. Simple, good for quick
  exploration.
- `HamiltonianMC` - scalar gradient-based Hamiltonian Monte Carlo with a fixed step.
- `NUTS` - the No-U-Turn Sampler, an adaptive variant of HMC. The recommended default.
- `HMC` - vector-aware Hamiltonian Monte Carlo with step-size adaptation. Pairs with
  the `summary` helper exported from the same module.

Every sampler's `sample(model, init, options)` draws one chain and returns
`{ trace, acceptanceRate, stepSize }`. With `{ chains: n }` in the options it draws
`n` chains and returns a Promise of `{ trace, byChain, chains, acceptanceRates, seeds,
parallel, parallelReason }`: pooled draws, per-chain draws, and each chain's own
result. The chains run on worker threads when the runtime can start one and the
model can be sent across, which holds for any model built from `addVariable`,
`observe` and `autoPotential`; otherwise in series, with `parallelReason` naming
the cause. The draws are identical either way. Options: `chains`, `inits` (one
initial point per chain), `nSamples`, `nWarmup`, `thin`, `seed`, `parallel`, `quiet`.

Every gradient sampler moves through an unconstrained transform for a bounded
variable, a `HalfNormal`, `Gamma`, `Lognormal`, `Beta` or `Uniform`, with the
log-Jacobian applied, and records draws on the natural scale.

`sampleChains(factory, options)` is the older route to workers for a model holding
a term that cannot be serialized, such as a `potential` over plain numbers; the
factory is sent as source and sees only its `(data, mc)` arguments.

## Model

`Model` represents a probabilistic model as a directed acyclic graph.

- `new Model('name')` or `new Model({ name })`
- `addVariable(name, distribution, observed?)` - register a random variable.
- `observe(name, factory, data)` - declare an observed variable. `factory(v)` receives
  the free variables as grad expressions and returns the distribution the data came
  from; its `logDensity` at `data` is the likelihood term, differentiated exactly.
- `autoPotential(name, fn, options?)` - a term written directly as a grad expression,
  for a likelihood or factor no distribution supplies; `fn(v)` returns a scalar
  expression. Compiled by default; `{ compile: false }` opts out.
- `potential(name, fn, gradFn?)` - a term as a plain function of numbers; `fn(params)`
  returns the log density. Without a `gradFn` it is differentiated by central finite
  differences, and it cannot be sent to a worker.
- `serializable()`, `toJSON(at)`, `Model.fromJSON(json)` - the model as data, which
  is how it reaches a worker.
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

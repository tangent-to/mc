---
layout: default
title: Getting Started
nav_order: 2
---

# Getting Started
{: .no_toc }

This guide installs `@tangent.to/mc`, then walks through fitting a complete Bayesian
linear regression from priors to posterior summary.

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Installation

`@tangent.to/mc` ships a single browser-first build, so the fastest way to run it is a
direct ESM import from a CDN - no install and no build step. It is also published to
npm and JSR for bundler and Node projects.

It runs on plain JavaScript numbers and arrays. Its only dependencies are two small
suite leaves — [proba](https://github.com/tangent-to/proba) for distributions and
[grad](https://github.com/tangent-to/grad) for autodiff — and both are resolved for you
on a CDN and by any bundler. There is no peer dependency to install.

### Browser / CDN (no build step)

A single import works in a plain `<script type="module">` - nothing else to load:

```html
<script type="module">
  import { Model, Normal, MetropolisHastings, printSummary }
    from 'https://cdn.jsdelivr.net/npm/@tangent.to/mc/+esm';
  // ... build and sample your model
</script>
```

### Observable

The same single import works in an Observable cell:

```javascript
mc = import("https://cdn.jsdelivr.net/npm/@tangent.to/mc/+esm")
```

### Deno

```typescript
import { Model, Normal, MetropolisHastings } from "jsr:@tangent/mc";
// or, from npm:
import { Model, Normal, MetropolisHastings } from "npm:@tangent.to/mc";
```

### Node.js / npm / bundlers

For a bundler (Vite, webpack, esbuild, …) or a Node project:

```bash
npm install @tangent.to/mc
```

```javascript
import { Model, Normal, MetropolisHastings, printSummary } from '@tangent.to/mc';
```

## Import styles

Three import styles are supported. Flat named imports are used throughout these docs:

```javascript
// 1. Flat named imports
import { Model, Normal, MetropolisHastings } from '@tangent.to/mc';

// 2. Grouped namespaces
import { distributions, samplers, diagnostics, plot } from '@tangent.to/mc';

// 3. Default export bundling every namespace
import mc from '@tangent.to/mc';   // mc.Model, mc.distributions.Normal, ...
```

## Your first model: Bayesian linear regression

We will fit `y = alpha + beta * x + noise` and recover the slope, intercept, and
noise scale from data.

### 1. The data

```javascript
import mc, { Model, Normal, HalfNormal, NUTS, summary } from '@tangent.to/mc';
const { add, mul } = mc.ops;

// Synthetic data generated from alpha = 1, beta = 2, sigma = 0.5
const x = [0, 1, 2, 3, 4, 5, 6, 7];
const y = [1.1, 2.8, 5.2, 6.9, 9.1, 11.0, 12.8, 15.2];
```

### 2. Define the model

Declare a prior for each free variable with `addVariable`, then say what was
observed with `observe`. The factory you give `observe` receives the free variables
and returns the distribution the data came from, with its parameters written as
expressions in those variables; the likelihood and its exact gradient are derived
from that distribution.

```javascript
const model = new Model('linear_regression');

// Priors, each on its natural scale. sigma is positive by its prior, and the
// sampler moves through a transform that keeps it so.
model.addVariable('alpha', new Normal(0, 10));
model.addVariable('beta',  new Normal(0, 10));
model.addVariable('sigma', new HalfNormal(5));

// Observation model: y ~ Normal(alpha + beta * x, sigma).
model.observe('y', (v) => new Normal(add(v.alpha, mul(v.beta, x)), v.sigma), y);
```

The mean is written with `add` and `mul` from `mc.ops` rather than `+` and `*`,
because JavaScript cannot overload operators on an object; both take any number of
operands, so a longer mean is still one flat call. The operations come from `mc.ops`
and not from a separate import of grad, which would load the module twice and break
the expression the moment the two copies disagreed on a version.

For a likelihood no distribution supplies, `autoPotential(name, fn)` takes the
log-density itself as an expression in those operations, and `potential(name, fn)`
takes it as a plain function of numbers, differentiated by finite differences.

### 3. Sample the posterior

`NUTS` uses the gradient and tunes its own step size. Provide a starting point for
every free variable and ask for four chains:

```javascript
const fit = await new NUTS().sample(
  model,
  { alpha: 0, beta: 0, sigma: 1 },
  { chains: 4, nSamples: 1000, nWarmup: 500 }
);
```

The chains run on worker threads when the runtime can start one and the model can
be sent across, which a model built from `addVariable` and `observe` always can;
otherwise they run one after another on the calling thread, and `fit.parallelReason`
says why. The draws are the same either way. `fit.trace` holds the pooled posterior
draws keyed by variable name, `fit.byChain` the same per chain, and `fit.chains` each
chain's own result.

With `chains` absent, `sample` returns a single chain's `{ trace, acceptanceRate,
stepSize }` synchronously.

### 4. Inspect the results

`summary` produces an ArviZ-style table (one row per parameter) with the posterior
mean, standard deviation, HDI, effective sample size, and R-hat:

```javascript
console.table(summary(fit.trace));
// param  mean   sd     hdi_lo  hdi_hi  ess   rhat
// alpha  ~1.0   ...    ...     ...     ...   ...
// beta   ~2.0   ...    ...     ...     ...   ...
// sigma  ~0.5   ...    ...     ...     ...   ...
```

Pass the per-chain results and `summary` computes R-hat across them:

```javascript
console.table(summary(fit.chains));   // each row now includes an rhat column
```

### A simpler alternative: Metropolis-Hastings

For quick exploration, the gradient-free `MetropolisHastings` sampler needs only a
proposal scale and returns a trace you can feed to `printSummary`:

```javascript
import { MetropolisHastings, printSummary } from '@tangent.to/mc';

const mh = new MetropolisHastings({ proposalStd: 0.3 });
const trace = mh.sample(
  model,
  { alpha: 0, beta: 0, sigma: 1 },
  { nSamples: 2000, burnIn: 1000, thin: 1 }
);

printSummary(trace);
```

## Next steps

- [API Reference](api) - every distribution, sampler, diagnostic, and plot.
- [Examples](examples) - logistic regression and hierarchical models worked end to end.

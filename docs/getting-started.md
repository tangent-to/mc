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
import { Model, Normal, HalfNormal, HMC, summary } from '@tangent.to/mc';

// Synthetic data generated from alpha = 1, beta = 2, sigma = 0.5
const x = [0, 1, 2, 3, 4, 5, 6, 7];
const y = [1.1, 2.8, 5.2, 6.9, 9.1, 11.0, 12.8, 15.2];
```

### 2. Define the model

Declare a prior for each free variable with `addVariable`, then attach the
likelihood with `potential`. A potential receives the current parameter values
(as `tf` tensors) and returns a log-density tensor that is summed into the joint
log-probability.

```javascript
const model = new Model('linear_regression');

// Priors
model.addVariable('alpha', new Normal({ mean: 0, sd: 10, name: 'alpha' }));
model.addVariable('beta',  new Normal({ mean: 0, sd: 10, name: 'beta' }));
model.addVariable('sigma', new HalfNormal(5, 'sigma'));

// Likelihood: y ~ Normal(alpha + beta * x, sigma).
// Distributions broadcast over arrays, so this is one vectorized call.
model.potential('y', (p) =>
  new Normal(x.map((xi) => p.alpha + p.beta * xi), p.sigma).logProb(y));
```

`potential` gets its gradient by central finite differences, which is fine for a
three-parameter model. For anything larger, write the same term with `autoPotential`
and [grad](https://github.com/tangent-to/grad) differentiates it exactly:

```javascript
import { add, div, log, mul, square, sub, sum } from '@tangent.to/grad';

model.autoPotential('y', (p) => {
  const z = div(sub(y, add(mul(p.beta, x), p.alpha)), p.sigma);
  return sub(mul(-0.5, sum(square(z))), mul(y.length, log(p.sigma)));
});
```

### 3. Sample the posterior

The vector-aware `HMC` sampler uses gradients (via automatic differentiation) and
works well for this gradient-friendly model. Provide a starting point for every free
variable:

```javascript
const sampler = new HMC({ stepSize: 0.01, nSteps: 20 });

const result = sampler.sample(
  model,
  { alpha: 0, beta: 0, sigma: 1 },
  { nSamples: 1000, nWarmup: 500 }
);
```

`HMC#sample` returns `{ trace, acceptanceRate, stepSize, divergences }`. The `trace`
holds the posterior draws keyed by variable name.

### 4. Inspect the results

`summary` produces an ArviZ-style table (one row per parameter) with the posterior
mean, standard deviation, HDI, effective sample size, and R-hat:

```javascript
console.table(summary(result));
// param  mean   sd     hdi_lo  hdi_hi  ess   rhat
// alpha  ~1.0   ...    ...     ...     ...   ...
// beta   ~2.0   ...    ...     ...     ...   ...
// sigma  ~0.5   ...    ...     ...     ...   ...
```

You can also run several chains and pass them all to `summary` so it can compute
R-hat across chains:

```javascript
const chains = sampler.sampleChains(
  model,
  { alpha: 0, beta: 0, sigma: 1 },
  { chains: 4, nSamples: 1000, nWarmup: 500 }
);
console.table(summary(chains));
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

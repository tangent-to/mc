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

`@tangent.to/mc` ships a single browser-first build and uses
[TensorFlow.js](https://www.tensorflow.org/js) (`@tensorflow/tfjs`) for tensor math
and automatic differentiation. **`tfjs` is a peer dependency** - it is *not* bundled,
so you install or load it once and share it. Mixing two copies of `tfjs` breaks
tensor interop.

### Node.js / npm

```bash
npm install @tangent.to/mc @tensorflow/tfjs
```

```javascript
import { Model, Normal, MetropolisHastings, printSummary } from '@tangent.to/mc';
```

A bundler (Vite, webpack, esbuild, …) resolves the `@tensorflow/tfjs` peer dependency
for you - nothing else to do.

### Deno

```typescript
import { Model, Normal, MetropolisHastings } from "jsr:@tangent/mc";
// or, from npm:
import { Model, Normal, MetropolisHastings } from "npm:@tangent.to/mc";
```

### Browser / CDN (no build step)

Bare specifiers like `@tensorflow/tfjs` don't resolve in the browser, so add an
[import map](https://developer.mozilla.org/docs/Web/HTML/Element/script/type/importmap)
*before* importing `mc`:

```html
<script type="importmap">
{
  "imports": {
    "@tensorflow/tfjs": "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4/+esm"
  }
}
</script>
<script type="module">
  import { Model, Normal, MetropolisHastings, printSummary }
    from 'https://cdn.jsdelivr.net/npm/@tangent.to/mc/+esm';
  // ... build and sample your model
</script>
```

`mc` also re-exports the shared `tf` instance, so you can build tensors with the
exact copy the library uses:

```javascript
import { tf } from '@tangent.to/mc';
```

### Observable

jsDelivr's `+esm` endpoint auto-resolves the `tfjs` dependency, so a single import
works in an Observable notebook:

```javascript
mc = import("https://cdn.jsdelivr.net/npm/@tangent.to/mc/+esm")
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

// Likelihood: y ~ Normal(alpha + beta * x, sigma)
model.potential('y', (p) => {
  const xt = tf.tensor1d(x);
  const mu = tf.add(tf.mul(p.beta, xt), p.alpha);
  return new Normal(mu, p.sigma).logProb(tf.tensor1d(y));
});
```

Here `tf` is the shared TensorFlow.js instance - import it with
`import { tf } from '@tangent.to/mc'`.

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

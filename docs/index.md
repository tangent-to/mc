---
layout: default
title: Home
---

# @tangent.to/mc

**JavaScript Markov Chain Monte Carlo** - A PyMC-inspired probabilistic programming library for Bayesian inference in JavaScript.

[![npm version](https://img.shields.io/npm/v/@tangent.to/mc.svg)](https://www.npmjs.com/package/@tangent.to/mc)
[![JSR](https://jsr.io/badges/@tangent-to/mc)](https://jsr.io/@tangent-to/mc)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Quick Start

### Installation

**Node.js / npm:**
```bash
npm install @tangent.to/mc
```

**Deno:**
```typescript
import { Model, Normal, MetropolisHastings } from "jsr:@tangent-to/mc";
```

**Observable:**
```javascript
mc = import("https://cdn.jsdelivr.net/npm/@tangent.to/mc/src/browser.js")
```

## Features

- **PyMC-like DAG structure** - Define models by connecting distributions in a directed acyclic graph
- **TensorFlow.js integration** - Automatic differentiation for gradient-based samplers
- **Multiple MCMC samplers** - Metropolis-Hastings and Hamiltonian Monte Carlo
- **Rich distribution library** - Normal, Uniform, Beta, Gamma, Bernoulli, and more
- **Gaussian Processes** - Non-parametric regression with multiple kernel functions
- **Posterior predictions** - Generate predictions with uncertainty from MCMC samples
- **Model persistence** - Save and load traces and model configurations to JSON
- **Browser compatible** - Run in Node.js or in the browser (including ObservableHQ)

## Example: Bayesian Linear Regression

```javascript
import { Model, Normal, Uniform, MetropolisHastings, printSummary } from '@tangent.to/mc';

// Create model
const model = new Model('linear_regression');

// Define priors
const alpha = new Normal(0, 10, 'alpha');
const beta = new Normal(0, 10, 'beta');
const sigma = new Uniform(0.01, 5, 'sigma');

model.addVariable('alpha', alpha);
model.addVariable('beta', beta);
model.addVariable('sigma', sigma);

// Define likelihood
model.logProb = function(params) {
  let logProb = alpha.logProb(params.alpha)
    .add(beta.logProb(params.beta))
    .add(sigma.logProb(params.sigma));

  for (let i = 0; i < x.length; i++) {
    const mu = params.alpha + params.beta * x[i];
    const likelihood = new Normal(mu, params.sigma);
    logProb = logProb.add(likelihood.logProb(y[i]));
  }

  return logProb;
};

// Run MCMC sampling
const sampler = new MetropolisHastings(0.5);
const trace = sampler.sample(model, initialValues, 1000, 500, 1);

// Analyze results
printSummary(trace);
```

## Documentation

- [API Reference](api/)
- [Distributions](api/distributions.html)
- [Samplers](api/samplers.html)
- [Gaussian Processes](api/gaussian-processes.html)
- [Platform Guides](PLATFORMS.html)

## Mathematical Foundation

### Bayesian Inference

The goal is to compute the posterior distribution:

$$
p(\theta|y) = \frac{p(y|\theta)p(\theta)}{p(y)} \propto p(y|\theta)p(\theta)
$$

where:
- $\theta$ are the model parameters
- $y$ is the observed data
- $p(\theta)$ is the prior distribution
- $p(y|\theta)$ is the likelihood
- $p(\theta|y)$ is the posterior distribution

### MCMC Sampling

Since the posterior is often intractable, we use Markov Chain Monte Carlo (MCMC) to generate samples from it. The samples $\theta^{(1)}, \theta^{(2)}, \ldots, \theta^{(N)}$ approximate the posterior distribution.

## Contributing

Contributions are welcome! Please see our [GitHub repository](https://github.com/tangent-to/mc).

## License

Apache-2.0

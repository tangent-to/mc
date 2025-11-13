# @tangent/mc - JavaScript Markov Chain Monte Carlo

A PyMC-inspired probabilistic programming library for Bayesian inference in JavaScript. Built on TensorFlow.js with automatic differentiation support for efficient MCMC sampling.

## Overview

MC brings the power of Bayesian statistical modeling to JavaScript, providing an intuitive API similar to PyMC for defining probabilistic models as Directed Acyclic Graphs (DAGs) and performing inference using Markov Chain Monte Carlo methods.

### Key Features

- **PyMC-like DAG structure**: Define models by connecting distributions in a directed acyclic graph
- **TensorFlow.js integration**: Automatic differentiation for gradient-based samplers
- **Multiple MCMC samplers**: Metropolis-Hastings and Hamiltonian Monte Carlo
- **Rich distribution library**: Normal, Uniform, Beta, Gamma, Bernoulli, and more
- **Gaussian Processes**: Non-parametric regression with multiple kernel functions (RBF, Matérn, Periodic)
- **Posterior predictions**: Generate predictions with uncertainty from MCMC samples
- **Model persistence**: Save and load traces and model configurations to JSON
- **Trace analysis utilities**: Summary statistics, effective sample size, convergence diagnostics
- **Hierarchical models**: Support for multilevel Bayesian models
- **Browser compatible**: Run in Node.js or in the browser (including ObservableHQ)

## Installation

### Node.js / npm

```bash
npm install @tangent.to/mc
```

### Deno

```typescript
import { Model, Normal, MetropolisHastings } from "npm:@tangent.to/mc";
```

### Observable

```javascript
mc = import("https://cdn.jsdelivr.net/npm/@tangent.to/mc/src/browser.js")
```

Or add to your `package.json`:

```json
{
  "dependencies": {
    "@tangent.to/mc": "^0.2.0"
  }
}
```

## Quick Start

Here's a simple Bayesian linear regression example:

```javascript
import { Model, Normal, Uniform, MetropolisHastings, printSummary } from '@tangent.to/mc';

// Create model
const model = new Model('linear_regression');

// Define priors (PyMC-like syntax)
const alpha = new Normal(0, 10, 'alpha');
const beta = new Normal(0, 10, 'beta');
const sigma = new Uniform(0.01, 5, 'sigma');

model.addVariable('alpha', alpha);
model.addVariable('beta', beta);
model.addVariable('sigma', sigma);

// Define likelihood (connecting distributions in a DAG)
model.logProb = function(params) {
  let logProb = alpha.logProb(params.alpha)
    .add(beta.logProb(params.beta))
    .add(sigma.logProb(params.sigma));

  // Add likelihood for observations
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

## Core Concepts

### Models as DAGs

Like PyMC, JSMC uses a Directed Acyclic Graph (DAG) structure to represent probabilistic models. Variables can depend on other variables, creating a natural flow from priors through transformations to likelihoods:

```javascript
// Hyperpriors
const mu_global = new Normal(0, 10);
const sigma_global = new Uniform(0, 5);

// Group-level parameters (depend on hyperpriors)
const mu_group = new Normal(mu_global, sigma_global);

// Observations (depend on group parameters)
const y = new Normal(mu_group, sigma_obs);
```

### Distributions

JSMC provides a rich set of probability distributions:

#### Continuous Distributions

- **Normal**: `new Normal(mu, sigma)` - Gaussian distribution
- **Uniform**: `new Uniform(lower, upper)` - Uniform distribution
- **Beta**: `new Beta(alpha, beta)` - Beta distribution (for probabilities)
- **Gamma**: `new Gamma(alpha, beta)` - Gamma distribution (for positive values)

#### Discrete Distributions

- **Bernoulli**: `new Bernoulli(p)` - Binary outcomes

All distributions support:
- `logProb(value)` - Compute log probability density/mass
- `sample(shape)` - Generate random samples
- `mean()` - Get the distribution mean
- `variance()` - Get the distribution variance

### Gaussian Processes

JSMC includes a full implementation of Gaussian Processes for non-parametric regression:

```javascript
import { GaussianProcess, RBF, Matern32 } from '@tangent.to/mc';

// Create GP with RBF kernel
const kernel = new RBF(lengthscale=1.0, variance=1.0);
const gp = new GaussianProcess(meanFunction=0, kernel, noiseVariance=0.01);

// Fit to data
gp.fit(X_train, y_train);

// Make predictions
const predictions = gp.predict(X_test, returnStd=true);
// Returns: { mean: [...], std: [...] }

// Sample functions from posterior
const posteriorSamples = gp.samplePosterior(X_test, nSamples=5);
```

**Available Kernels**:
- **RBF** (Squared Exponential): Smooth, infinitely differentiable functions
- **Matern32**: Less smooth than RBF, once differentiable
- **Matern52**: Middle ground between Matern32 and RBF
- **Periodic**: For periodic/seasonal patterns
- **Linear**: For linear trends

### Model Predictions

Generate posterior predictive samples for new data:

```javascript
// Define prediction function
const predictFn = (params) => {
  return params.alpha + params.beta * x_new;
};

// Get posterior predictions with uncertainty
const predictions = model.predictPosteriorSummary(
  trace,
  predictFn,
  credibleInterval=0.95
);
// Returns: { mean: [...], lower: [...], upper: [...] }
```

### Model Persistence

Save and load model states and traces:

```javascript
import { saveTrace, loadTrace, saveModelState } from '@tangent.to/mc';

// Save trace to JSON
saveTrace(trace, 'trace.json');

// Load trace
const loadedTrace = loadTrace('trace.json');

// Save complete model state
saveModelState(model, trace, 'model_state.json');

// Export for browser (no filesystem)
const jsonString = exportTraceForBrowser(trace);
```

### MCMC Samplers

#### Metropolis-Hastings

A simple but effective random-walk sampler:

```javascript
const sampler = new MetropolisHastings(proposalStd);
const trace = sampler.sample(model, initialValues, nSamples, burnIn, thin);
```

**Parameters**:
- `proposalStd`: Standard deviation of the Gaussian proposal distribution
- `nSamples`: Number of samples to collect
- `burnIn`: Number of initial samples to discard
- `thin`: Keep every nth sample

**Best for**: Simple models, initial exploration

#### Hamiltonian Monte Carlo

A gradient-based sampler that uses automatic differentiation:

```javascript
const sampler = new HamiltonianMC(stepSize, nSteps);
const trace = sampler.sample(model, initialValues, nSamples, burnIn, thin);
```

**Parameters**:
- `stepSize`: Leapfrog integration step size (epsilon)
- `nSteps`: Number of leapfrog steps (L)

**Best for**: Complex models with many parameters, faster convergence

### Trace Analysis

JSMC provides utilities for analyzing MCMC samples:

```javascript
import { summarize, effectiveSampleSize, gelmanRubin, printSummary } from '@tangent.to/mc';

// Print comprehensive summary
printSummary(trace);

// Get statistics for a variable
const stats = summarize(trace.trace.alpha);
// Returns: { mean, median, std, variance, hdi_2_5, hdi_97_5, n }

// Compute effective sample size
const ess = effectiveSampleSize(trace.trace.alpha);

// Check convergence with multiple chains
const rHat = gelmanRubin([chain1.alpha, chain2.alpha, chain3.alpha]);
```

## Examples

The `examples/` directory contains complete working examples:

### Linear Regression
```bash
node examples/linear_regression.js
```

Demonstrates basic Bayesian linear regression with normal priors.

### Logistic Regression
```bash
node examples/logistic_regression.js
```

Binary classification with a logistic link function.

### Hierarchical Model
```bash
node examples/hierarchical_model.js
```

Multilevel model with partial pooling across groups, showcasing complex DAG structures.

### Gaussian Process Regression
```bash
node examples/gaussian_process.js
```

Non-parametric regression using Gaussian Processes with different kernels and uncertainty quantification.

## API Reference

### Model Class

```javascript
const model = new Model(name)
```

**Methods**:
- `addVariable(name, distribution, observed)` - Add a variable to the model
- `getVariable(name)` - Retrieve a variable
- `logProb(params)` - Compute log probability
- `logProbAndGradient(params)` - Compute log prob and gradients
- `samplePrior(nSamples)` - Sample from prior distributions
- `getFreeVariableNames()` - Get unobserved variable names
- `summary()` - Print model structure

### Distribution Classes

All distributions inherit from the base `Distribution` class:

```javascript
class Distribution {
  logProb(value)      // Log probability
  sample(shape)       // Generate samples
  observe(data)       // Set observed data
  mean()             // Distribution mean
  variance()         // Distribution variance
}
```

### Sampler Classes

```javascript
class MetropolisHastings {
  constructor(proposalStd)
  sample(model, initialValues, nSamples, burnIn, thin)
  tuneProposal(acceptanceRate)
}

class HamiltonianMC {
  constructor(stepSize, nSteps)
  sample(model, initialValues, nSamples, burnIn, thin)
}
```

## Browser & ObservableHQ

JSMC works seamlessly in browser environments, including ObservableHQ notebooks:

```javascript
// In Observable, import from npm
jsmc = import("https://cdn.jsdelivr.net/npm/jsmc/src/browser.js")

// Use it!
{
  const { Model, Normal, MetropolisHastings } = jsmc;
  // ... define and run your model
}
```

**Key differences in browser**:
- Uses `@tensorflow/tfjs` instead of `@tensorflow/tfjs-node`
- File I/O functions (`saveTrace`, `loadTrace`) not available
- Use `exportTraceForBrowser()` and download as JSON instead
- Slightly slower than Node.js, but enables interactive visualization

**See [docs/OBSERVABLE.md](docs/OBSERVABLE.md) for detailed Observable examples and best practices.**

## Technical Details

### Built on TensorFlow.js

JSMC leverages TensorFlow.js for:
- **Automatic differentiation**: Essential for gradient-based samplers like HMC
- **Efficient tensor operations**: Fast computation of log probabilities
- **GPU acceleration**: Optional GPU support for large-scale models

### Comparison with PyMC

| Feature | PyMC | JSMC |
|---------|------|------|
| Language | Python | JavaScript |
| Backend | Aesara/JAX | TensorFlow.js |
| DAG Structure | Yes | Yes |
| MCMC Samplers | NUTS, HMC, MH | HMC, MH |
| Variational Inference | Yes | Planned |
| GPU Support | Yes | Yes (via TF.js) |

## Performance Tips

1. **Tune sampler parameters**:
   - MH: Aim for 20-40% acceptance rate by adjusting `proposalStd`
   - HMC: Start with small `stepSize` (~0.01) and moderate `nSteps` (~10)

2. **Use appropriate burn-in**: Discard at least 500-1000 initial samples

3. **Check convergence**:
   - Visual inspection of trace plots
   - R-hat < 1.1 for multiple chains
   - Effective sample size > 100 per chain

4. **Hierarchical models**: Use HMC for faster convergence with many parameters

## Development

```bash
# Clone repository
git clone https://github.com/tangent-to/mc.git
cd jsmc

# Install dependencies
npm install

# Run examples
npm run example

# Run tests
npm test
```

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

Apache-2.0

## Roadmap

**Completed in v0.2.0**:
- [x] Gaussian Processes with multiple kernels
- [x] Posterior predictive sampling
- [x] Model persistence (save/load)
- [x] Browser/Observable support

**Planned**:
- [ ] Additional distributions (Poisson, Student-t, Exponential)
- [ ] NUTS (No-U-Turn Sampler)
- [ ] Variational inference (ADVI)
- [ ] Sparse GPs (inducing points for large datasets)
- [ ] Model comparison utilities (WAIC, LOO)
- [ ] Trace visualization tools
- [ ] PyMC model import/export

## Documentation

- **[Observable Guide](docs/OBSERVABLE.md)** - Using JSMC in ObservableHQ notebooks
- **[Considerations](docs/CONSIDERATIONS.md)** - Best practices, limitations, and design decisions
- **[Examples](examples/)** - Complete working examples

## References

- [PyMC Documentation](https://www.pymc.io/)
- [TensorFlow.js](https://www.tensorflow.org/js)
- [Bayesian Data Analysis (Gelman et al.)](http://www.stat.columbia.edu/~gelman/book/)
- [MCMC sampling for dummies](https://twiecki.io/blog/2015/11/10/mcmc-sampling-for-dummies/)
- [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/)

## Citation

If you use JSMC in your research, please cite:

```bibtex
@software{jsmc,
  title = {JSMC: JavaScript Markov Chain Monte Carlo},
  author = {},
  year = {2025},
  url = {https://github.com/tangent-to/mc}
}
```

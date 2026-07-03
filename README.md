# @tangent/mc - JavaScript Markov Chain Monte Carlo

A PyMC-inspired probabilistic programming library for Bayesian inference in JavaScript. Built on TensorFlow.js with automatic differentiation support for efficient MCMC sampling.

## Overview

MC brings the power of Bayesian statistical modeling to JavaScript, providing an intuitive API similar to PyMC for defining probabilistic models as Directed Acyclic Graphs (DAGs) and performing inference using Markov Chain Monte Carlo methods.

### API conventions

MC follows the same API conventions as its sibling data-science package [`@tangent.to/ds`](https://github.com/tangent-to/ds):

- **Namespaced + flat exports.** Import individual symbols (`import { Normal } from '@tangent.to/mc'`), grouped namespaces (`import { distributions, samplers } from '@tangent.to/mc'`), or the whole library as a default export (`import mc from '@tangent.to/mc'` → `mc.distributions.Normal`). The namespaces are `distributions`, `samplers`, `diagnostics`, and `plot`. (File-based persistence is a Node-only subpath, `@tangent.to/mc/persistence`, not a namespace on the main entry.)
- **Options-object constructors.** Every configurable class accepts a single options object in addition to positional arguments, e.g. `new Normal({ mean: 0, sd: 1 })` or `new MetropolisHastings({ proposalStd: 0.5 })`. Positional forms continue to work.
- **Introspection.** Distributions and samplers expose `getParams()`.

### Key Features

- **PyMC-like DAG structure**: Define models by connecting distributions in a directed acyclic graph
- **TensorFlow.js integration**: Automatic differentiation for gradient-based samplers
- **Multiple MCMC samplers**: Metropolis-Hastings, Hamiltonian Monte Carlo, and the No-U-Turn Sampler (NUTS)
- **Rich distribution library**: Normal, Uniform, Beta, Gamma, Bernoulli, and more
- **Posterior predictions**: Generate predictions with uncertainty from MCMC samples
- **Model persistence**: Save and load traces and model configurations to JSON
- **Trace analysis utilities**: Summary statistics, effective sample size, convergence diagnostics
- **Hierarchical models**: Support for multilevel Bayesian models
- **Browser compatible**: Run in Node.js or in the browser (including ObservableHQ)

## Installation

`@tangent.to/mc` ships a single browser-first build, so the fastest way to run it is a
direct ESM import from a CDN - no install and no build step. It is also on npm and JSR
for bundler and Node projects.

It uses [TensorFlow.js](https://www.tensorflow.org/js) (`@tensorflow/tfjs`) for tensor
math and automatic differentiation. `tfjs` is a **peer dependency**: it is not bundled,
so it is loaded once and shared (mixing two copies breaks tensor interop). On a CDN the
`+esm` endpoint resolves it for you; with a bundler you install it alongside `mc`.

### Browser / CDN (no build step)

jsDelivr's `+esm` endpoint auto-resolves `tfjs`, so a single import works in a plain
`<script type="module">` - nothing else to load:

```html
<script type="module">
  import { Model, Normal, MetropolisHastings }
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

For a bundler (Vite, webpack, esbuild, …) or a Node project, install `mc` with the
`tfjs` peer dependency alongside it:

```bash
npm install @tangent.to/mc @tensorflow/tfjs
```

```javascript
import { Model, Normal, MetropolisHastings } from '@tangent.to/mc';
```

The bundler resolves the peer dependency for you. `mc` also re-exports the shared `tf`
instance, so you can grab it from `mc` itself instead of importing it separately:

```javascript
import { tf } from '@tangent.to/mc';
```

## Quick Start

Here's a simple Bayesian linear regression example:

```javascript
import { Model, Normal, Uniform, MetropolisHastings, printSummary, tf } from '@tangent.to/mc';

// Example data
const x = [1, 2, 3, 4, 5];
const y = [2.1, 3.9, 6.2, 7.8, 10.1];
const xT = tf.tensor1d(x), yT = tf.tensor1d(y);

// Create the model and its priors (options-object form; positional also works).
const model = new Model({ name: 'linear_regression' });
model.addVariable('alpha', new Normal({ mean: 0, sd: 10 }));
model.addVariable('beta',  new Normal({ mean: 0, sd: 10 }));
model.addVariable('sigma', new Uniform({ min: 0.01, max: 5 }));

// Likelihood as a POTENTIAL. The priors above are summed into the joint
// automatically; `potential(name, fn)` adds the data term, with the deterministic
// mean built from the latent parameters. Work in tensors so the gradient-based
// samplers (HMC / NUTS) can differentiate through it, and so the term is vectorized.
model.potential('y', (p) =>
  new Normal(tf.add(tf.mul(p.beta, xT), p.alpha), p.sigma).logProb(yT));

// (Optional) record a post-hoc transform of the draws into the trace:
// model.deterministic('mu_at_x3', (p) => p.alpha + p.beta * 3);

// Run MCMC (options-object form; positional args also work).
const sampler = new MetropolisHastings({ proposalStd: 0.5 });
const trace = sampler.sample(model, { alpha: 0, beta: 0, sigma: 1 }, { nSamples: 1000, burnIn: 500, thin: 1 });

printSummary(trace);
```

### Namespaced / default imports

```javascript
import mc from '@tangent.to/mc';

const model = new mc.Model({ name: 'linear_regression' });
model.addVariable('alpha', new mc.distributions.Normal({ mean: 0, sd: 10, name: 'alpha' }));

const sampler = new mc.samplers.MetropolisHastings({ proposalStd: 0.5 });
// mc.diagnostics, mc.io, mc.plot are also available
```

## Core Concepts

### Models as DAGs

A model's joint log-probability is the sum of (1) the priors of the registered
variables and (2) any `potential` terms - generic log-density factors whose
parameters are arbitrary deterministic functions of the latent variables and data.
Dependencies between variables (hierarchies, transformed parameters) are expressed
**inside a `potential`**, by computing the dependent quantity there - *not* by
passing one distribution object as another distribution's parameter:

```javascript
import { Model, Normal, tf } from '@tangent.to/mc';

const model = new Model('hierarchical');
model.addVariable('mu_global', new Normal(0, 10));
model.addVariable('log_sigma_global', new Normal(0, 1)); // unconstrained - see below
model.addVariable('z', new Normal(0, 1));                // non-centred group offset

// The group mean depends on the hyperparameters - built in the potential:
model.potential('y', (p) => {
  const sigmaGlobal = tf.exp(p.log_sigma_global);
  const muGroup = tf.add(p.mu_global, tf.mul(sigmaGlobal, p.z));
  return new Normal(muGroup, sigmaObs).logProb(yT);
});
```

> **Sampler constraint (important).** The gradient samplers (`HMC`, `NUTS`) treat
> every free variable as an unconstrained **scalar** and apply **no support
> transforms**. A constrained prior (Beta, HalfNormal, Uniform, Lognormal) can wander
> off its support and produce `NaN` gradients. The robust pattern is to declare
> **unconstrained** latents - e.g. a `Normal` prior on `log_sigma` or `logit_p` - 
> and apply `tf.exp` / `tf.sigmoid` **inside** the `potential`, which is equivalent
> to a Lognormal / Beta prior on the natural parameter. (Metropolis-Hastings has no
> such restriction - it rejects out-of-support proposals via the `-Infinity` logProb.)

### Distributions

mc provides a rich set of probability distributions:

Each constructor accepts positional arguments or an options object (shown second).

#### Continuous Distributions

- **Normal**: `new Normal(mu, sigma)` / `new Normal({ mean, sd })` - Gaussian distribution
- **Uniform**: `new Uniform(lower, upper)` / `new Uniform({ min, max })` - Uniform distribution
- **Beta**: `new Beta(alpha, beta)` / `new Beta({ alpha, beta })` - Beta distribution (for probabilities)
- **Gamma**: `new Gamma(alpha, beta)` / `new Gamma({ shape, rate })` - Gamma distribution (for positive values)
- **Lognormal**: `new Lognormal(mu, sigma)` / `new Lognormal({ mu, sigma })` - positive values (log-scale normal); a good prior for rates, scales, and plateaus
- **HalfNormal**: `new HalfNormal(sigma)` / `new HalfNormal({ sigma })` - positive values concentrated near zero; a good prior for scale / standard-deviation parameters

#### Discrete Distributions

- **Bernoulli**: `new Bernoulli(p)` / `new Bernoulli({ p })` - Binary outcomes

All distributions support:
- `logProb(value)` - Compute log probability density/mass
- `pdf(value)` - Compute probability density/mass (`exp(logProb)`)
- `sample(shape)` - Generate random samples
- `mean()` - Get the distribution mean
- `variance()` - Get the distribution variance
- `getParams()` - Get the distribution's parameters as a plain object

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

File-based persistence is **Node-only** (`node:fs`) and is intentionally kept out of
the browser-first main entry. Import it directly from its module in Node:

```javascript
import {
  saveTrace,
  loadTrace,
  saveModelState,
  exportTraceForBrowser
} from '@tangent.to/mc/persistence';

// Save trace to JSON
saveTrace(trace, 'trace.json');

// Load trace
const loadedTrace = loadTrace('trace.json');

// Save complete model state
saveModelState(model, trace, 'model_state.json');

// In the browser, serialize to a JSON string instead (no filesystem)
const jsonString = exportTraceForBrowser(trace);
```

### MCMC Samplers

#### Metropolis-Hastings

A simple but effective random-walk sampler:

```javascript
const sampler = new MetropolisHastings({ proposalStd });
const trace = sampler.sample(model, initialValues, { nSamples, burnIn, thin });
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
const sampler = new HamiltonianMC({ stepSize, nSteps });
const trace = sampler.sample(model, initialValues, { nSamples, burnIn, thin });
```

**Parameters**:
- `stepSize`: Leapfrog integration step size (epsilon)
- `nSteps`: Number of leapfrog steps (L)

**Best for**: Complex models with many parameters, faster convergence

### Trace Analysis

mc provides utilities for analyzing MCMC samples:

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

## API Reference

### Model Class

```javascript
const model = new Model(name)
```

**Methods**:
- `addVariable(name, distribution, observed)` - Add a variable to the model
- `potential(name, fn)` - Add a generic log-density term (likelihood / factor); `fn(params)` returns a log-density tensor
- `deterministic(name, fn)` - Register a post-hoc transform of the draws; recorded into the trace by the samplers
- `getVariable(name)` - Retrieve a variable
- `logProb(params)` - Compute log probability
- `logProbAndGradient(params)` - Compute log prob and gradients
- `samplePrior(nSamples)` - Sample from prior distributions
- `getFreeVariableNames()` - Get unobserved variable names
- `computeDeterministics(trace)` - Append deterministic columns to a trace (called automatically by samplers)
- `summary()` - Return a string summary of the model structure

### Distribution Classes

All distributions inherit from the base `Distribution` class:

```javascript
class Distribution {
  logProb(value)      // Log probability
  pdf(value)          // Probability density/mass (exp of logProb)
  sample(shape)       // Generate samples
  observe(data)       // Set observed data
  mean()             // Distribution mean
  variance()         // Distribution variance
  getParams()        // Parameters as a plain object
}
```

### Sampler Classes

Constructors and `sample()` accept either positional arguments or a single
options object.

```javascript
class MetropolisHastings {
  constructor(proposalStd)                 // or ({ proposalStd })
  sample(model, initialValues, nSamples, burnIn, thin)  // or (model, init, { nSamples, burnIn, thin })
  tuneProposal(acceptanceRate)
  getParams()
}

class HamiltonianMC {
  constructor(stepSize, nSteps)            // or ({ stepSize, nSteps })
  sample(model, initialValues, nSamples, burnIn, thin)
  getParams()
}

class NUTS {
  constructor(stepSize, maxTreeDepth, targetAcceptance)  // or ({ stepSize, maxTreeDepth, targetAcceptance })
  sample(model, initialValues, nSamples, nWarmup, thin)
  getParams()
}
```

## Browser & ObservableHQ

`mc` is a single browser-first build that runs the same in the browser, Node, and
ObservableHQ - see [Installation](#installation) for loading `tfjs`. In Observable:

```javascript
mc = import("https://cdn.jsdelivr.net/npm/@tangent.to/mc/+esm")

{
  const { Model, Normal, MetropolisHastings } = mc;
  // ... define and run your model
}
```

**Notes for browser/Observable use**:
- `tfjs` runs on its CPU/WebGL backend (there is no `@tensorflow/tfjs-node` - the
  single build uses `@tensorflow/tfjs` everywhere), which enables interactive
  visualization.
- File-based persistence (`saveTrace`, `loadTrace`) is Node-only (`node:fs`) and is
  not part of the browser entry. Import it from the `@tangent.to/mc/persistence`
  subpath in Node if needed; in the browser, serialize with `traceToJSON(trace)`.

## Technical Details

### Built on TensorFlow.js

mc leverages TensorFlow.js for:
- **Automatic differentiation**: Essential for gradient-based samplers like HMC/NUTS
- **Efficient tensor operations**: Fast computation of log probabilities
- **WebGL acceleration**: GPU-backed tensor math in the browser via the WebGL backend

### Comparison with PyMC

| Feature | PyMC | mc |
|---------|------|------|
| Language | Python | JavaScript |
| Backend | Aesara/JAX | TensorFlow.js |
| DAG Structure | Yes | Yes |
| MCMC Samplers | NUTS, HMC, MH | NUTS, HMC, MH |
| Variational Inference | Yes | Planned |
| GPU Support | Yes | Browser only (TF.js WebGL) |

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
cd mc

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

**Done**:
- [x] Posterior predictive sampling
- [x] Model persistence (save/load)
- [x] Browser/Observable support
- [x] No-U-Turn Sampler (NUTS) with dual-averaging step-size adaptation
- [x] Convergence diagnostics (Gelman-Rubin R-hat, effective sample size)
- [x] Namespaced + flat + default exports and options-object constructors (aligned with `@tangent.to/ds`)
- [x] Lognormal and HalfNormal distributions
- [x] Post-hoc deterministics recorded into the trace

**Planned**:
- [ ] Additional distributions (Poisson, Student-t, Exponential)
- [ ] Variational inference (ADVI)
- [ ] Model comparison utilities (WAIC, LOO)
- [ ] Trace visualization tools
- [ ] PyMC model import/export

## Documentation

- **[Observable Guide](docs/OBSERVABLE.md)** - Using mc in ObservableHQ notebooks
- **[Considerations](docs/CONSIDERATIONS.md)** - Best practices, limitations, and design decisions
- **[Examples](examples/)** - Complete working examples

## References

- [PyMC Documentation](https://www.pymc.io/)
- [TensorFlow.js](https://www.tensorflow.org/js)
- [Bayesian Data Analysis (Gelman et al.)](http://www.stat.columbia.edu/~gelman/book/)
- [MCMC sampling for dummies](https://twiecki.io/blog/2015/11/10/mcmc-sampling-for-dummies/)

## Citation

If you use mc in your research, please cite:

```bibtex
@software{mc,
  title = {mc: JavaScript Markov Chain Monte Carlo},
  author = {tangent-to},
  year = {2025},
  url = {https://github.com/tangent-to/mc}
}
```

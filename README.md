# @tangent/mc - JavaScript Markov Chain Monte Carlo

A PyMC-inspired probabilistic programming library for Bayesian inference in JavaScript. Runs on plain numbers and arrays, with exact gradients for the samplers that need them.

## Overview

MC brings the power of Bayesian statistical modeling to JavaScript, providing an intuitive API similar to PyMC for defining probabilistic models as Directed Acyclic Graphs (DAGs) and performing inference using Markov Chain Monte Carlo methods.

### API conventions

MC follows the same API conventions as its sibling data-science package [`@tangent.to/ds`](https://github.com/tangent-to/ds):

- **Namespaced + flat exports.** Import individual symbols (`import { Normal } from '@tangent.to/mc'`), grouped namespaces (`import { distributions, samplers } from '@tangent.to/mc'`), or the whole library as a default export (`import mc from '@tangent.to/mc'` → `mc.distributions.Normal`). The namespaces are `distributions`, `samplers`, `diagnostics`, and `plot`. (File-based persistence is a Node-only subpath, `@tangent.to/mc/persistence`, not a namespace on the main entry.)
- **Options-object constructors.** Every configurable class accepts a single options object in addition to positional arguments, e.g. `new Normal({ mean: 0, sd: 1 })` or `new MetropolisHastings({ proposalStd: 0.5 })`. Positional forms continue to work.
- **Introspection.** Distributions and samplers expose `getParams()`.

### Key Features

- **PyMC-like DAG structure**: Define models by connecting distributions in a directed acyclic graph
- **Observed variables**: `model.observe(name, (v) => new Normal(mean(v), v.sigma), data)` derives the likelihood, and its exact gradient, from the distribution
- **Chains where they fit**: `sample(model, init, { chains: 4 })` runs on worker threads when the runtime and the model allow it, on the calling thread otherwise, with the same draws either way
- **Exact gradients**: analytic log-density derivatives from [`@tangent.to/proba`](https://github.com/tangent-to/proba), and reverse-mode autodiff from [`@tangent.to/grad`](https://github.com/tangent-to/grad) for likelihoods you write yourself
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

It runs on plain JavaScript numbers and arrays. There is no tensor library to load and
no peer dependency to install: its only dependencies are two small suite leaves,
[`@tangent.to/proba`](https://github.com/tangent-to/proba) for distributions and
[`@tangent.to/grad`](https://github.com/tangent-to/grad) for autodiff, both resolved
for you on a CDN and by any bundler.

### Browser / CDN (no build step)

A single import works in a plain `<script type="module">` - nothing else to load:

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

For a bundler (Vite, webpack, esbuild, …) or a Node project:

```bash
npm install @tangent.to/mc
```

```javascript
import { Model, Normal, MetropolisHastings } from '@tangent.to/mc';
```

## Quick Start

Here's a simple Bayesian linear regression example:

```javascript
import mc, { Model, Normal, HalfNormal, NUTS, printSummary } from '@tangent.to/mc';
const { add, mul } = mc.ops;

// Example data - plain arrays, no tensors
const x = [1, 2, 3, 4, 5];
const y = [2.1, 3.9, 6.2, 7.8, 10.1];

// Priors, on their natural scale: sigma is a HalfNormal and the sampler
// handles the transform.
const model = new Model('linear_regression');
model.addVariable('alpha', new Normal(0, 10));
model.addVariable('beta',  new Normal(0, 10));
model.addVariable('sigma', new HalfNormal(5));

// The observation model. The mean is an expression in the parameters; the
// likelihood, and its exact gradient, are derived from the Normal.
model.observe('y', (v) => new Normal(add(v.alpha, mul(v.beta, x)), v.sigma), y);

// Four chains. mc runs them on worker threads where it can, in series where it
// cannot, with the same draws either way.
const fit = await new NUTS().sample(model, { alpha: 0, beta: 0, sigma: 1 },
  { chains: 4, nSamples: 1000, nWarmup: 500 });

printSummary(fit.trace);
```

That is the whole model. There is no log-density written out, no Jacobian for
`sigma`, and no loop over chains. The one trace of JavaScript in it is that
`alpha + beta * x` has to be written as a call, because the language cannot
overload `+` on an object; `add` and `mul` take any number of operands so that
nothing nests.

The lower-level forms remain. `model.potential(name, fn)` adds a term as a
plain function of the parameters, differentiated by finite differences;
`model.autoPotential(name, fn)` adds one written in grad's ops, differentiated
exactly, for a likelihood no distribution supplies. See "Gradients" below.

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
import { Model, Normal, HalfNormal } from '@tangent.to/mc';

const model = new Model('hierarchical');
model.addVariable('mu_global',    new Normal(0, 10));
model.addVariable('sigma_global', new HalfNormal(1));  // a scale, sampled transformed
model.addVariable('z',            new Normal(0, 1));   // non-centred group offset

// The group mean depends on the hyperparameters - built in the potential:
model.potential('y', (p) => {
  const muGroup = p.z.map((zi) => p.mu_global + p.sigma_global * zi);
  return new Normal(muGroup, sigmaObs).logProb(y);
});
```

> **Constrained parameters.** Since 0.9.0, `NUTS` samples a bounded variable in an
> unconstrained parameterization and applies the log-Jacobian correction, as Stan and
> PyMC do — so a `HalfNormal` scale or a `Beta` probability is stepped through `ℝ` and
> can never be proposed outside its support. Declare the natural parameter and let the
> sampler transform it.
>
> Before 0.9.0 this was the user's job: the advice was to declare an unconstrained
> latent (`Normal` on `log_sigma`) and exponentiate inside the potential. That still
> works and remains a good idea for a hierarchical scale, where the non-centred form
> helps the geometry regardless. `HMC` and the vector `HMC` do not yet go through the
> transform; `MetropolisHastings` never needed it, rejecting out-of-support proposals
> through their `-Infinity` log-density.

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

#### Parallel chains: `sampleChains`

MCMC chains are independent, so they can run simultaneously — one worker per
chain (browser/Deno `Worker` or Node `worker_threads`). Four chains cost
roughly one chain of wall-clock time:

```javascript
import { sampleChains, gelmanRubin } from '@tangent.to/mc';

const fit = await sampleChains(
  // The factory must be SELF-CONTAINED: it is serialized to each worker, so
  // it may only use its two arguments (data, mc) and JavaScript built-ins.
  (data, mc) => {
    const model = new mc.Model('lin');
    model.addVariable('a', new mc.distributions.Normal(0, 5));
    model.addVariable('b', new mc.distributions.Normal(0, 5));
    model.addVariable('logSig', new mc.distributions.Normal(0, 1));
    model.potential('lik', (p) => {
      const sig = Math.exp(p.logSig);
      const mu = data.xs.map((x) => p.a + p.b * x);
      return new mc.distributions.Normal(mu, sig).logProb(data.ys);
    });
    return model;
  },
  {
    data: { xs, ys },            // structured-clonable data for the factory
    chains: 4,
    inits: [init1, init2, init3, init4], // over-dispersed starts, one per chain
    sampler: 'nuts',             // 'nuts' | 'hmc' | 'metropolis'
    samplerOptions: { stepSize: 0.02, maxTreeDepth: 8, targetAcceptance: 0.85 },
    nSamples: 400, nWarmup: 400,
    seed: 20240115,              // per-chain seeds are derived from this
  },
);

fit.byChain.a      // [[chain-0 draws], [chain-1 draws], …] → gelmanRubin(fit.byChain.a)
fit.trace.a        // pooled draws across chains
fit.chains[0]      // {trace, acceptanceRate, stepSize, seed} per chain
fit.parallel       // false if the runtime had no workers (sequential fallback)
```

Notes:
- **Reproducible, but differently seeded**: chain *c* is seeded from
  `options.seed`, so a `sampleChains` run reproduces exactly, but its draws
  differ from a single-stream sequential run of the same chains (independent
  per-chain streams are what R-hat assumes).
- **Fallback**: with `parallel: false`, or in a runtime without workers, the
  same chains run in-process with the same seeds — identical results, serial
  wall-clock.
- Everything the model needs must travel through `options.data`; referencing
  an outer variable from the factory throws with guidance.

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
ObservableHQ. In Observable:

```javascript
mc = import("https://cdn.jsdelivr.net/npm/@tangent.to/mc/+esm")

{
  const { Model, Normal, MetropolisHastings } = mc;
  // ... define and run your model
}
```

**Notes for browser/Observable use**:
- Everything runs on plain numbers, so there is no backend to select and no
  environment-specific build - the same bundle serves Node, the browser and Deno.
- File-based persistence (`saveTrace`, `loadTrace`) is Node-only (`node:fs`) and is
  not part of the browser entry. Import it from the `@tangent.to/mc/persistence`
  subpath in Node if needed; in the browser, serialize with `traceToJSON(trace)`.

## Technical Details

### Gradients

HMC and NUTS need the gradient of the joint log-probability. mc gets it from four
places, in descending order of preference:

- **Priors** are differentiated analytically, from `@tangent.to/proba`'s `dlogpdf`.
- **Observed variables, `model.observe(name, factory, data)`,** get their term from
  the distribution: `factory(v)` returns a distribution whose parameters are
  expressions in the free variables, and its `logDensity` at the data is added
  as a compiled term. Every one of mc's seven distributions can be observed, a
  Gamma or Beta with a differentiated shape parameter included, through grad's
  `lgamma`. This is the form to reach for first; it is what the Quick Start uses.
- **Potentials written with `autoPotential`** are differentiated exactly by
  [`@tangent.to/grad`](https://github.com/tangent-to/grad), reverse-mode, with no
  derivation by hand:

  ```javascript
  const { add, div, log, mul, square, sub, sum } = mc.ops;

  model.autoPotential('y', (v) => {
    const z = div(sub(yData, add(mul(v.beta, xData), v.alpha)), v.sigma);
    return sub(mul(-0.5, sum(square(z))), mul(yData.length, log(v.sigma)));
  });
  ```

  The tape is compiled: built once and replayed at each new set of parameters,
  rather than reconstructed on every call. On a 300-observation regression with
  10 parameters that is 2.8x on a full NUTS run, drawing the same chain sample
  for sample. It is safe by default because `autoPotential`'s contract already
  requires the term to be an expression built from grad's ops, which fixes its
  graph, and a sampler holds every parameter's shape constant for the length of
  a run. Pass `{ compile: false }` if you step outside that, by branching on a
  parameter's numeric value or closing over data that mutates mid-run.

  The operations come from `mc.ops` rather than a separate import of
  `@tangent.to/grad`. That second import is a correctness hazard, not a matter
  of taste: it loads a second copy of the module as soon as mc's own dependency
  range resolves to a different version than the one pinned beside it, and the
  two copies have different `Var` classes, so `autoPotential` rejects an
  expression built with the other one.

  Models written this way run in parallel as they are, through
  `sampler.sample(model, init, { chains: 4 })`. The model is serialized, its
  variables as their distribution's kind and parameters and its terms as
  compiled grad plans, and one worker per chain rebuilds it from that data; no
  factory is written and nothing is threaded through by hand. When the model
  holds a term that cannot travel, a `potential` over plain numbers, or the
  runtime cannot start a worker, the chains run in series with the reason named
  once, and `fit.parallel` and `fit.parallelReason` say which happened. The
  draws are identical either way, since both paths derive the same per-chain
  seeds.

  `sampleChains` remains for the case that cannot be serialized and still wants
  workers: it sends a factory's source to each worker, where it can reference
  nothing but its two arguments, so grad's ops arrive as `mc.ops`:

  ```javascript
  await sampleChains((data, mc) => {
    const { add, div, log, mul, square, sub, sum } = mc.ops;
    const model = new mc.Model('lin');
    // ... addVariable, then autoPotential written in those ops
    return model;
  }, { data, chains: 4, inits, nSamples: 400, nWarmup: 400, seed: 20240115 });
  ```

  On a 340-observation model with 15 parameters, four chains of 400 draws take
  19.5 s in series and 5.7 s on four workers, drawing the same samples.

- **Potentials written with `potential`** fall back to central finite differences,
  which cost 2·(#params) extra evaluations of the whole term per gradient and are
  accurate only to ~1e-7 - enough error to cost the leapfrog integrator its
  symplectic property. Pass an explicit `gradFn` if you have one, or prefer
  `autoPotential`.

Constrained parameters (a scale in `(0, ∞)`, a probability in `(0, 1)`) are sampled
in an unconstrained parameterization with the log-Jacobian correction applied, as Stan
and PyMC do, so the sampler never proposes a value outside the support.

### Not built on TensorFlow.js

Until 0.5.0 mc ran on `@tensorflow/tfjs` and took it as a peer dependency. It no
longer does: the numerics are plain numbers and arrays. If you are following older
material, there is no `tf` export, no peer dependency to install, and no backend to
select.

### Comparison with PyMC

| Feature | PyMC | mc |
|---------|------|------|
| Language | Python | JavaScript |
| Backend | PyTensor | plain arrays + @tangent.to/grad |
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

GPL-3.0 (application layer of the tangent suite; the numeric leaves it
builds on are MIT).

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
- [Stan Reference Manual](https://mc-stan.org/docs/reference-manual/) - the
  constrained-parameter transforms and NUTS follow it
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

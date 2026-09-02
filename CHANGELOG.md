# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **NUTS counts divergent transitions.** `sample()` returns `divergences`
  (after warmup, the count to act on) and `divergencesWarmup`, and warns once
  when the former is positive; a multi-chain run aggregates and warns once
  over all chains, with the per-chain counts in `fit.divergences`. The
  diagnostic R-hat and ESS cannot give: a chain that diverges is exploring a
  region its step size cannot resolve, typically the neck of a funnel where a
  scale parameter approaches zero.

### Fixed

- `chainToolkit.samplers` is a getter, for the same import-cycle reason the
  sampler registry is resolved at call time: loading a sampler module first
  read the bindings before they existed.

## [0.7.1] - 2026-07-09

### Fixed
- **`sampleChains` silently ran sequentially in browsers when served from a CDN.** Runtime detection checked `process.versions.node` first, but CDN builds (e.g. esm.sh) shim `process` in the browser with `versions.node` defined, routing chains to the (unavailable) `worker_threads` path and triggering the sequential fallback. The web `Worker` path is now tried first — Node has no global `Worker`, so Node still reaches `worker_threads`.

## [0.7.0] - 2026-07-09

### Added
- **Parallel chains: `sampleChains(modelFactory, options)`.** Runs each MCMC chain in its own worker (browser/Deno `Worker`, Node `worker_threads`), so four chains cost roughly one chain of wall-clock time (~2.8x measured on 4 chains including worker startup). The model is described by a self-contained factory `(data, mc) => Model` — serialized to each worker — plus structured-clonable `data`. Returns `{chains, byChain, trace, acceptanceRates, seeds, parallel}`; `byChain` feeds `gelmanRubin` directly. Per-chain seeds derive from `options.seed` (reproducible, but draws differ from a single-stream sequential run). Falls back to an in-process sequential run with identical results when the runtime has no workers or with `parallel: false`.

## [0.6.2] - 2026-07-09

### Fixed
- Release workflow: pin npm 11 (npm 12.0.0 broke trusted publishing the day it became `latest`) and skip a version already on JSR so a half-succeeded release can be re-run. 0.6.1 reached JSR but not npm; 0.6.2 is the release both registries share.

## [0.6.1] - 2026-07-09

### Changed
- **Faster leapfrog: `Model.gradientsOnly()`.** NUTS (and the legacy `hmc.js`) called `logProbAndGradient` in every leapfrog step and discarded the log-probability, paying a full potential-value pass over the data on top of the gradient pass. `Model.gradientsOnly(params)` returns identical gradients without the value pass, and the samplers use it; the Hamiltonians used for acceptance still come from the same `logProb` calls as before. Seeded traces are bit-identical; NUTS with analytic potential gradients runs ~2x faster (~17% with the finite-difference fallback). `hmc-vector` (the exported `HMC`) already consumed both value and gradient from one call and is unchanged.

## [0.6.0] - 2026-07-08

### Fixed
- **NUTS slice-membership criterion.** In the tree builder a state was tested for slice membership with `slice <= exp(H0 - H)` instead of `slice <= exp(-H)` (the slice variable is drawn on the `exp(-H0)` scale). Because `H0` scales with the data log-likelihood magnitude, the test was effectively always true, disabling the energy weighting so NUTS sampled trajectory states almost uniformly. This left posterior *means* correct but inflated posterior *standard deviations* (~40% on a conjugate-Gaussian check: NUTS sd ≈ 0.31 vs analytic 0.224; now ≈ 0.224). Added `tests/nuts-posterior.test.js` pinning the recovered spread. The divergence check sign was corrected to `(H - H0) > deltaMax`.
- **NUTS reported acceptance rate.** `acceptanceRate` now returns the mean Metropolis acceptance probability over the run (the statistic dual averaging targets via `targetAcceptance`), instead of the fraction of iterations whose average tree acceptance exceeded an arbitrary 0.5.

### Added

**API alignment with `@tangent.to/ds`**
- Namespaced exports (`distributions`, `kernels`, `samplers`, `diagnostics`, `io`, `plot`) and a default export bundling every namespace, alongside the existing flat named exports.
- Options-object constructors for every configurable class (distributions, kernels, Gaussian processes, samplers, and `Model`), e.g. `new Normal({ mean, sd })` and `new MetropolisHastings({ proposalStd })`. Positional forms remain supported.
- Options-object form for sampler `sample()` run controls, e.g. `sample(model, init, { nSamples, burnIn, thin })`.
- `getParams()` on distributions, kernels, and samplers; `setParams()` on kernels.
- `Kernel` base class (now exported) with a `call()` alias for `compute()`; all kernels extend it and expose a camelCase `lengthScale` accessor (alias of `lengthscale`).
- `pdf()` on distributions and `isFitted()` on `GaussianProcess`.

### Changed
- `package.json` now declares a `module` entry point and `import`/`default` export conditions, matching the `@tangent.to/ds` packaging convention.

### Removed
- **Gaussian Processes.** The `GaussianProcess` distribution and its GP kernels (`Kernel`, `RBF`, `Matern32`, `Matern52`, `Periodic`, `Linear`) have been removed, along with the `kernels` namespace. GP regression is an ML concern better served by `@tangent.to/ds`'s `GaussianProcessRegressor`; the `mc` implementation overlapped it, was the buggiest part of the package, and was the only consumer of the `ml-matrix` dependency (now dropped).
- `energyPlot`, a non-functional placeholder, has been removed from the public API.

### Internal
- Removed dead code: an unused `tf` import in the Metropolis-Hastings sampler, an unused `accept` variable (and a thrice-computed `Math.exp`) in the NUTS tree builder, an unused intermediate array in `pairPlot`, and a vestigial `shape` getter on the `Distribution` base class.
- De-duplicated samplers and kernels: the `hamiltonian()` calculation and trace bookkeeping now live in a shared `samplers/_shared.js` helper, and the pairwise squared-distance computation shared by the RBF and Matérn kernels was extracted into a single helper. No change to sampling results.
- Stopped tracking generated API docs (`docs/api/`) in git and removed the unreferenced `IMPROVEMENTS.md`.

## [0.2.0] - 2025-11-06

### Added

**Gaussian Processes**
- Full GP implementation with Cholesky-based inference
- Five kernel functions: RBF, Matern32, Matern52, Periodic, Linear
- `fit()`, `predict()`, and `samplePosterior()` methods
- Log marginal likelihood for hyperparameter optimization
- Comprehensive GP example in `examples/gaussian_process.js`

**Model Predictions**
- `predictPosterior()`: Generate predictions from MCMC samples
- `predictPosteriorSummary()`: Compute mean and credible intervals
- Full uncertainty quantification for predictions

**Model Persistence**
- `saveTrace()` and `loadTrace()`: JSON serialization
- `saveModelState()` and `loadModelState()`: Complete model persistence
- `saveTraceCSV()`: Export for external analysis tools
- `exportTraceForBrowser()` and `importTraceFromJSON()`: Browser-compatible persistence

**Browser Support**
- `src/browser.js`: Browser-specific build using @tensorflow/tfjs
- Dual export system in package.json (Node.js and browser)
- Full ObservableHQ compatibility
- CDN-ready for use via jsdelivr

**Documentation**
- Auto-generated API documentation from JSDoc comments
- `scripts/generate-docs.js`: Documentation generator
- `scripts/serve-docs.js`: Local documentation server
- `docs/OBSERVABLE.md`: Complete Observable integration guide
- `docs/PLATFORMS.md`: Guide for Observable, Deno, and Jupyter Lab
- `docs/CONSIDERATIONS.md`: Best practices and architecture decisions
- Professional documentation without emojis

**Testing**
- PyMC comparison test suite with 12 tests
- Tests for linear regression, GP, predictions, and persistence
- All tests validate against PyMC-equivalent results
- Automated testing in CI/CD pipeline

**CI/CD**
- GitHub Actions workflow for npm publishing
- Automated testing on Node.js 18, 20, and 22
- Pre-publish test and documentation generation
- Release workflow triggered by GitHub releases

### Changed
- Package name: `jsmc` → `@tangent.to/mc`
- Repository: `essicolo/jsmc` → `tangent-to/mc`
- Organization: Individual → tangent-to organization
- Improved error handling in GP implementation
- Better numerical stability with jitter handling

### Fixed
- GP implementation now uses ml-matrix for Cholesky decomposition
- TensorFlow.js linalg limitations worked around
- Proper tensor disposal to prevent memory leaks
- Forward/backward substitution for efficient linear solves

### Dependencies
- Added: `ml-matrix@6.12.1` for linear algebra operations

## [0.1.0] - 2025-11-06

### Added

**Core Distributions**
- Normal (Gaussian) distribution
- Uniform distribution
- Beta distribution
- Gamma distribution
- Bernoulli distribution
- Base distribution class with common interface

**MCMC Samplers**
- Metropolis-Hastings algorithm
- Hamiltonian Monte Carlo with automatic differentiation
- Configurable burn-in, thinning, and sample counts
- Acceptance rate monitoring and diagnostics

**Model System**
- PyMC-like DAG structure for defining Bayesian models
- `addVariable()` for composing probabilistic models
- Automatic log probability computation
- Gradient computation via TensorFlow.js

**Trace Analysis**
- Summary statistics (mean, std, HDI)
- Effective Sample Size (ESS) calculation
- Gelman-Rubin convergence diagnostic (R-hat)
- `printSummary()` for comprehensive output
- JSON and CSV export utilities

**Examples**
- Linear regression with normal priors
- Logistic regression for binary classification
- Hierarchical model with partial pooling

**Documentation**
- Comprehensive README with API reference
- Quick start guide
- Performance tips
- Comparison with PyMC

### Technical
- Built on TensorFlow.js for automatic differentiation
- ES6 modules for modern JavaScript
- Dual export for Node.js and browser
- Apache-2.0 license

## Package Information

- **npm**: https://www.npmjs.com/package/@tangent.to/mc
- **GitHub**: https://github.com/tangent-to/mc
- **Documentation**: https://github.com/tangent-to/mc#readme
- **Issues**: https://github.com/tangent-to/mc/issues

[Unreleased]: https://github.com/tangent-to/mc/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/tangent-to/mc/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/tangent-to/mc/releases/tag/v0.1.0

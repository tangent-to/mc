# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

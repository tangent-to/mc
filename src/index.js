/**
 * Main entry point for `@tangent.to/mc` — a browser-first Bayesian inference
 * library (PyMC-style models, MCMC samplers, and trace diagnostics) running on
 * plain numbers/arrays via `@tangent.to/proba`.
 *
 * Two complementary import styles are supported, matching the convention used
 * by the sibling `@tangent.to/ds` package:
 *
 *   1. Flat named imports:
 *        import { Model, Normal, MetropolisHastings } from '@tangent.to/mc';
 *
 *   2. Namespaced imports (and a default export bundling every namespace):
 *        import { distributions, samplers, diagnostics } from '@tangent.to/mc';
 *        import mc from '@tangent.to/mc';  // mc.distributions.Normal, ...
 *
 * @module
 */

// Since 0.5.0 mc runs on plain numbers/arrays via @tangent.to/proba —
// no TensorFlow.js. Use setRandomSeed(seed) for reproducible runs.
import { getRng, setRandomSeed } from './rng.js';
export { getRng, setRandomSeed };

import { Model } from './model.js';

// Parallel chains (one worker per chain; sequential in-process fallback).
// __runChain is internal but must be exported: the chain workers import this
// module by URL (also when bundled, where parallel.js is inlined here) and
// call it.
import { sampleChains, __runChain } from './parallel.js';
export { sampleChains, __runChain };

// grad's operations, so a model written with autoPotential can reach them
// without importing @tangent.to/grad separately. That second import is not a
// convenience question: it loads a SECOND copy of the module as soon as mc's
// own dependency range resolves to a different version than the one pinned
// alongside it, and the two copies have different Var classes, so
// autoPotential's `instanceof` check rejects an expression built with the
// other one. Reaching them through mc guarantees one copy.
//
// This is the same namespace the model factory receives as `mc.ops` inside a
// worker, so a model reads identically whether its chains run in workers or on
// the calling thread.
import * as ops from '@tangent.to/grad';
export { ops };

import {
  Distribution,
  Normal,
  Uniform,
  Bernoulli,
  Beta,
  Gamma,
  Lognormal,
  HalfNormal
} from './distributions/index.js';

import {
  MetropolisHastings,
  HamiltonianMC,
  NUTS,
  HMC,
  summary
} from './samplers/index.js';

import {
  summarize,
  effectiveSampleSize,
  gelmanRubin,
  printSummary,
  traceToJSON,
  traceToCSV
} from './utils/trace.js';

// Note: file-based trace persistence lives in ./utils/persistence.js (Node-only,
// uses node:fs) and is intentionally NOT re-exported here so that this entry —
// and the single browser build produced from it — stays browser-first. Import it
// from the '@tangent.to/mc/persistence' subpath in a Node context if needed.

import {
  tracePlot,
  posteriorPlot,
  autocorrPlot,
  pairPlot,
  forestPlot,
  rankPlot
} from './utils/visualize.js';

// ---------------------------------------------------------------------------
// Flat named exports (backwards compatible)
// ---------------------------------------------------------------------------
export { Model };
export {
  Distribution,
  Normal,
  Uniform,
  Bernoulli,
  Beta,
  Gamma,
  Lognormal,
  HalfNormal
};
export { MetropolisHastings, HamiltonianMC, NUTS, HMC, summary };
export {
  summarize,
  effectiveSampleSize,
  gelmanRubin,
  printSummary,
  traceToJSON,
  traceToCSV
};
export {
  tracePlot,
  posteriorPlot,
  autocorrPlot,
  pairPlot,
  forestPlot,
  rankPlot
};

// ---------------------------------------------------------------------------
// Namespaced exports (mirrors the @tangent.to/ds module convention)
// ---------------------------------------------------------------------------
/**
 * Namespace bundling every probability distribution class.
 * @type {Object}
 */
export const distributions = {
  Distribution,
  Normal,
  Uniform,
  Bernoulli,
  Beta,
  Gamma,
  Lognormal,
  HalfNormal
};

/**
 * Namespace bundling every MCMC sampler and the `summary` helper.
 * @type {Object}
 */
export const samplers = {
  MetropolisHastings,
  HamiltonianMC,
  NUTS,
  HMC,
  summary
};

/**
 * Namespace bundling the trace summary, convergence, and export helpers.
 * @type {Object}
 */
export const diagnostics = {
  summarize,
  effectiveSampleSize,
  gelmanRubin,
  printSummary,
  traceToJSON,
  traceToCSV
};

/**
 * Namespace bundling the ASCII/text trace-visualization helpers.
 * @type {Object}
 */
export const plot = {
  tracePlot,
  posteriorPlot,
  autocorrPlot,
  pairPlot,
  forestPlot,
  rankPlot
};

/**
 * Default export: the whole library grouped by namespace
 * ({@link Model}, `setRandomSeed`, `getRng`, and the `distributions`,
 * `samplers`, `diagnostics`, and `plot` namespaces).
 * @type {Object}
 */
export default {
  Model,
  setRandomSeed,
  getRng,
  sampleChains,
  ops,
  distributions,
  samplers,
  diagnostics,
  plot
};

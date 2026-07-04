// Main entry point for @tangent.to/mc
//
// Two complementary import styles are supported, matching the convention used
// by the sibling @tangent.to/ds package:
//
//   1. Flat named imports:
//        import { Model, Normal, MetropolisHastings } from '@tangent.to/mc';
//
//   2. Namespaced imports (and a default export bundling every namespace):
//        import { distributions, samplers, diagnostics } from '@tangent.to/mc';
//        import mc from '@tangent.to/mc';  // mc.distributions.Normal, ...

// Since 0.5.0 mc runs on plain numbers/arrays via @tangent.to/proba —
// no TensorFlow.js. Use setRandomSeed(seed) for reproducible runs.
import { getRng, setRandomSeed } from './rng.js';
export { getRng, setRandomSeed };

import { Model } from './model.js';

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

export const samplers = {
  MetropolisHastings,
  HamiltonianMC,
  NUTS,
  HMC,
  summary
};

export const diagnostics = {
  summarize,
  effectiveSampleSize,
  gelmanRubin,
  printSummary,
  traceToJSON,
  traceToCSV
};

export const plot = {
  tracePlot,
  posteriorPlot,
  autocorrPlot,
  pairPlot,
  forestPlot,
  rankPlot
};

// Default export: the whole library grouped by namespace
export default {
  Model,
  setRandomSeed,
  getRng,
  distributions,
  samplers,
  diagnostics,
  plot
};

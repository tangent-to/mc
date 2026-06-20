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

import { Model } from './model.js';

import {
  Distribution,
  Normal,
  Uniform,
  Bernoulli,
  Beta,
  Gamma
} from './distributions/index.js';

import {
  MetropolisHastings,
  HamiltonianMC,
  NUTS
} from './samplers/index.js';

import {
  summarize,
  effectiveSampleSize,
  gelmanRubin,
  printSummary,
  traceToJSON,
  traceToCSV
} from './utils/trace.js';

import {
  saveTrace,
  loadTrace,
  saveModelConfig,
  saveModelState,
  loadModelState,
  saveTraceCSV,
  exportTraceForBrowser,
  importTraceFromJSON
} from './utils/persistence.js';

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
  Gamma
};
export { MetropolisHastings, HamiltonianMC, NUTS };
export {
  summarize,
  effectiveSampleSize,
  gelmanRubin,
  printSummary,
  traceToJSON,
  traceToCSV
};
export {
  saveTrace,
  loadTrace,
  saveModelConfig,
  saveModelState,
  loadModelState,
  saveTraceCSV,
  exportTraceForBrowser,
  importTraceFromJSON
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
  Gamma
};

export const samplers = {
  MetropolisHastings,
  HamiltonianMC,
  NUTS
};

export const diagnostics = {
  summarize,
  effectiveSampleSize,
  gelmanRubin,
  printSummary,
  traceToJSON,
  traceToCSV
};

export const io = {
  saveTrace,
  loadTrace,
  saveModelConfig,
  saveModelState,
  loadModelState,
  saveTraceCSV,
  exportTraceForBrowser,
  importTraceFromJSON
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
  distributions,
  samplers,
  diagnostics,
  io,
  plot
};

/**
 * Browser-compatible build of @tangent.to/mc
 * Uses TensorFlow.js (@tensorflow/tfjs) instead of tfjs-node and excludes
 * Node.js-specific features (filesystem persistence).
 *
 * Exposes the same dual import shape as the Node entry point:
 *   - flat named exports
 *   - namespaced exports + a default export bundling every namespace
 */

import * as tf from '@tensorflow/tfjs';

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
  exportTraceForBrowser,
  importTraceFromJSON
} from './utils/persistence.js';

// ---------------------------------------------------------------------------
// Flat named exports (backwards compatible)
// ---------------------------------------------------------------------------
export { tf };
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
export { exportTraceForBrowser, importTraceFromJSON };

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
  exportTraceForBrowser,
  importTraceFromJSON
};

export default {
  tf,
  Model,
  distributions,
  samplers,
  diagnostics,
  io
};

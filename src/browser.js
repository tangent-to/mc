/**
 * Browser-compatible build of JSMC
 * Uses TensorFlow.js instead of tfjs-node
 * Excludes Node.js-specific features (fs operations)
 */

import * as tf from '@tensorflow/tfjs';

// Re-export TensorFlow for use in browser
export { tf };

// Export main classes (need to be browser-compatible versions)
export { Model } from './model.js';

// Distributions
export {
  Distribution,
  Normal,
  Uniform,
  Bernoulli,
  Beta,
  Gamma,
  GaussianProcess
} from './distributions/index.js';

// Kernels
export {
  RBF,
  Matern32,
  Matern52,
  Periodic,
  Linear
} from './distributions/index.js';

// Samplers
export {
  MetropolisHastings,
  HamiltonianMC
} from './samplers/index.js';

// Utilities (trace analysis only, no file I/O)
export {
  summarize,
  effectiveSampleSize,
  gelmanRubin,
  printSummary,
  traceToJSON,
  traceToCSV
} from './utils/trace.js';

// Browser-compatible persistence
export {
  exportTraceForBrowser,
  importTraceFromJSON
} from './utils/persistence.js';

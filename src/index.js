// Main exports for JSMC library
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

// Kernels for Gaussian Processes
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
  HamiltonianMC,
  NUTS
} from './samplers/index.js';

// Utilities
export {
  summarize,
  effectiveSampleSize,
  gelmanRubin,
  printSummary,
  traceToJSON,
  traceToCSV
} from './utils/trace.js';

// Persistence utilities
export {
  saveTrace,
  loadTrace,
  saveModelConfig,
  saveModelState,
  loadModelState,
  saveTraceCSV,
  exportTraceForBrowser,
  importTraceFromJSON
} from './utils/persistence.js';

// Visualization utilities
export {
  tracePlot,
  posteriorPlot,
  autocorrPlot,
  pairPlot,
  forestPlot,
  rankPlot,
  energyPlot
} from './utils/visualize.js';

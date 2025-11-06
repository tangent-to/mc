// Main exports for JSMC library
export { Model } from './model.js';

// Distributions
export {
  Distribution,
  Normal,
  Uniform,
  Bernoulli,
  Beta,
  Gamma
} from './distributions/index.js';

// Samplers
export {
  MetropolisHastings,
  HamiltonianMC
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

import fs from 'fs';

/**
 * Save trace data to JSON file
 * @param {Object} trace - Trace object from MCMC sampling
 * @param {string} filepath - Path to save the file
 */
export function saveTrace(trace, filepath) {
  const data = {
    trace: trace.trace || trace,
    metadata: {
      acceptanceRate: trace.acceptanceRate,
      nSamples: trace.nSamples,
      timestamp: new Date().toISOString()
    }
  };

  fs.writeFileSync(filepath, JSON.stringify(data, null, 2));
  console.log(`Trace saved to ${filepath}`);
}

/**
 * Load trace data from JSON file
 * @param {string} filepath - Path to the file
 * @returns {Object} Trace object
 */
export function loadTrace(filepath) {
  const data = JSON.parse(fs.readFileSync(filepath, 'utf8'));
  console.log(`Trace loaded from ${filepath}`);
  return data;
}

/**
 * Save model configuration to JSON
 * Note: This saves the model structure, not the trained parameters
 * @param {Model} model - The model to save
 * @param {string} filepath - Path to save the file
 */
export function saveModelConfig(model, filepath) {
  const config = {
    name: model.name,
    variables: [],
    timestamp: new Date().toISOString()
  };

  // Save variable names and distribution types
  for (const [name, distribution] of model.variables.entries()) {
    config.variables.push({
      name: name,
      distributionType: distribution.name,
      isObserved: distribution.observed !== null
    });
  }

  fs.writeFileSync(filepath, JSON.stringify(config, null, 2));
  console.log(`Model configuration saved to ${filepath}`);
}

/**
 * Save complete model state (config + trace)
 * @param {Model} model - The model
 * @param {Object} trace - The MCMC trace
 * @param {string} filepath - Path to save the file
 */
export function saveModelState(model, trace, filepath) {
  const state = {
    modelConfig: {
      name: model.name,
      variables: []
    },
    trace: trace.trace || trace,
    metadata: {
      acceptanceRate: trace.acceptanceRate,
      nSamples: trace.nSamples,
      timestamp: new Date().toISOString()
    }
  };

  // Save variable names and types
  for (const [name, distribution] of model.variables.entries()) {
    state.modelConfig.variables.push({
      name: name,
      distributionType: distribution.name,
      isObserved: distribution.observed !== null
    });
  }

  fs.writeFileSync(filepath, JSON.stringify(state, null, 2));
  console.log(`Model state saved to ${filepath}`);
}

/**
 * Load model state from file
 * @param {string} filepath - Path to the file
 * @returns {Object} {modelConfig, trace, metadata}
 */
export function loadModelState(filepath) {
  const state = JSON.parse(fs.readFileSync(filepath, 'utf8'));
  console.log(`Model state loaded from ${filepath}`);
  return state;
}

/**
 * Save trace to CSV format (for external analysis tools)
 * @param {Object} trace - Trace object
 * @param {string} filepath - Path to save the file
 */
export function saveTraceCSV(trace, filepath) {
  const traceData = trace.trace || trace;
  const variables = Object.keys(traceData);
  const nSamples = traceData[variables[0]].length;

  // Create CSV header
  let csv = 'iteration,' + variables.join(',') + '\n';

  // Add data rows
  for (let i = 0; i < nSamples; i++) {
    const row = [i];
    for (const variable of variables) {
      row.push(traceData[variable][i]);
    }
    csv += row.join(',') + '\n';
  }

  fs.writeFileSync(filepath, csv);
  console.log(`Trace saved to CSV: ${filepath}`);
}

/**
 * Export trace for browser environments (no fs dependency)
 * Returns a downloadable blob URL
 * @param {Object} trace - Trace object
 * @returns {string} JSON string for download
 */
export function exportTraceForBrowser(trace) {
  const data = {
    trace: trace.trace || trace,
    metadata: {
      acceptanceRate: trace.acceptanceRate,
      nSamples: trace.nSamples,
      timestamp: new Date().toISOString()
    }
  };

  return JSON.stringify(data, null, 2);
}

/**
 * Import trace from JSON string (browser-compatible)
 * @param {string} jsonString - JSON string
 * @returns {Object} Trace object
 */
export function importTraceFromJSON(jsonString) {
  return JSON.parse(jsonString);
}

/**
 * Visualization utilities for MCMC traces and posterior analysis
 *
 * Inspired by ArviZ (arviz-devs.github.io/arviz/)
 *
 * Design pattern: Generate plot specifications that can be passed to Observable Plot
 * or other plotting libraries. This avoids having plotting libraries as dependencies.
 *
 * Usage:
 *   const plotSpec = tracePlot(trace, ['alpha', 'beta']);
 *   plotSpec.show(Plot); // Pass Observable Plot
 */

/**
 * Generate trace plot specification
 * Shows the sampled values over iterations to assess convergence
 *
 * @param {Object} trace - MCMC trace object
 * @param {Array<string>} variables - Variable names to plot (null = all)
 * @param {Object} options - Plot options
 * @returns {Object} Plot specification with .show() method
 */
export function tracePlot(trace, variables = null, options = {}) {
  const traceData = trace.trace || trace;
  const varNames = variables || Object.keys(traceData);

  // Prepare data for plotting
  const plotData = [];
  for (const varName of varNames) {
    const samples = traceData[varName];
    samples.forEach((value, iteration) => {
      plotData.push({
        variable: varName,
        iteration: iteration,
        value: value
      });
    });
  }

  // Plot specification
  const spec = {
    type: 'trace',
    data: plotData,
    variables: varNames,
    title: options.title || 'Trace Plot',
    width: options.width || 800,
    height: options.height || 400,

    // Observable Plot specification
    observablePlot: (Plot) => {
      return Plot.plot({
        title: spec.title,
        width: spec.width,
        height: spec.height,
        facet: {
          data: plotData,
          y: 'variable',
          marginLeft: 100
        },
        marks: [
          Plot.line(plotData, {
            x: 'iteration',
            y: 'value',
            stroke: '#4682b4',
            strokeWidth: 1
          }),
          Plot.ruleY([0], { stroke: '#ccc', strokeDasharray: '4 4' })
        ],
        y: {
          label: null,
          grid: true
        },
        x: {
          label: 'Iteration',
          grid: true
        }
      });
    },

    // Generic show method
    show(plotLib) {
      if (plotLib && plotLib.plot) {
        return this.observablePlot(plotLib);
      }
      // Return data for manual plotting
      return {
        data: this.data,
        variables: this.variables,
        type: this.type
      };
    }
  };

  return spec;
}

/**
 * Generate posterior distribution plot specification
 * Shows histograms and KDE of posterior samples
 *
 * @param {Object} trace - MCMC trace object
 * @param {Array<string>} variables - Variable names to plot
 * @param {Object} options - Plot options
 * @returns {Object} Plot specification with .show() method
 */
export function posteriorPlot(trace, variables = null, options = {}) {
  const traceData = trace.trace || trace;
  const varNames = variables || Object.keys(traceData);

  // Prepare data
  const plotData = [];
  for (const varName of varNames) {
    const samples = traceData[varName];
    samples.forEach((value) => {
      plotData.push({
        variable: varName,
        value: value
      });
    });
  }

  // Compute summary statistics
  const stats = {};
  for (const varName of varNames) {
    const samples = traceData[varName];
    const sorted = [...samples].sort((a, b) => a - b);
    const n = sorted.length;
    stats[varName] = {
      mean: samples.reduce((a, b) => a + b, 0) / n,
      median: sorted[Math.floor(n / 2)],
      hdi_2_5: sorted[Math.floor(n * 0.025)],
      hdi_97_5: sorted[Math.floor(n * 0.975)]
    };
  }

  const spec = {
    type: 'posterior',
    data: plotData,
    stats: stats,
    variables: varNames,
    title: options.title || 'Posterior Distributions',
    width: options.width || 800,
    height: options.height || 400,

    observablePlot: (Plot) => {
      return Plot.plot({
        title: spec.title,
        width: spec.width,
        height: spec.height,
        marginLeft: 100,
        facet: {
          data: plotData,
          y: 'variable',
          marginLeft: 100
        },
        marks: [
          // Histogram
          Plot.rectY(plotData, Plot.binX(
            { y: 'count' },
            { x: 'value', fill: '#4682b4', fillOpacity: 0.6 }
          )),
          // Mean line
          Plot.ruleX(
            varNames.map(v => ({ variable: v, value: stats[v].mean })),
            { x: 'value', stroke: 'red', strokeWidth: 2 }
          ),
          // HDI interval
          Plot.ruleX(
            varNames.map(v => ({
              variable: v,
              x1: stats[v].hdi_2_5,
              x2: stats[v].hdi_97_5
            })),
            { x1: 'x1', x2: 'x2', stroke: 'black', strokeWidth: 3 }
          )
        ],
        x: {
          label: 'Value',
          grid: true
        },
        y: {
          label: null
        }
      });
    },

    show(plotLib) {
      if (plotLib && plotLib.plot) {
        return this.observablePlot(plotLib);
      }
      return {
        data: this.data,
        stats: this.stats,
        variables: this.variables,
        type: this.type
      };
    }
  };

  return spec;
}

/**
 * Generate autocorrelation plot specification
 * Shows autocorrelation to assess mixing
 *
 * @param {Object} trace - MCMC trace object
 * @param {Array<string>} variables - Variable names to plot
 * @param {number} maxLag - Maximum lag to compute
 * @param {Object} options - Plot options
 * @returns {Object} Plot specification with .show() method
 */
export function autocorrPlot(trace, variables = null, maxLag = 50, options = {}) {
  const traceData = trace.trace || trace;
  const varNames = variables || Object.keys(traceData);

  // Compute autocorrelation
  function autocorr(series, lag) {
    const n = series.length;
    const mean = series.reduce((a, b) => a + b, 0) / n;

    let numerator = 0;
    let denominator = 0;

    for (let i = 0; i < n - lag; i++) {
      numerator += (series[i] - mean) * (series[i + lag] - mean);
    }

    for (let i = 0; i < n; i++) {
      denominator += (series[i] - mean) ** 2;
    }

    return numerator / denominator;
  }

  const plotData = [];
  for (const varName of varNames) {
    const samples = traceData[varName];
    for (let lag = 0; lag <= Math.min(maxLag, samples.length - 1); lag++) {
      plotData.push({
        variable: varName,
        lag: lag,
        autocorrelation: autocorr(samples, lag)
      });
    }
  }

  const spec = {
    type: 'autocorr',
    data: plotData,
    variables: varNames,
    maxLag: maxLag,
    title: options.title || 'Autocorrelation Plot',
    width: options.width || 800,
    height: options.height || 400,

    observablePlot: (Plot) => {
      return Plot.plot({
        title: spec.title,
        width: spec.width,
        height: spec.height,
        facet: {
          data: plotData,
          y: 'variable',
          marginLeft: 100
        },
        marks: [
          Plot.ruleY([0], { stroke: '#ccc' }),
          Plot.ruleY([0.05], { stroke: 'red', strokeDasharray: '4 4' }),
          Plot.ruleY([-0.05], { stroke: 'red', strokeDasharray: '4 4' }),
          Plot.line(plotData, {
            x: 'lag',
            y: 'autocorrelation',
            stroke: '#4682b4',
            strokeWidth: 2
          })
        ],
        x: {
          label: 'Lag',
          grid: true
        },
        y: {
          label: 'Autocorrelation',
          domain: [-1, 1],
          grid: true
        }
      });
    },

    show(plotLib) {
      if (plotLib && plotLib.plot) {
        return this.observablePlot(plotLib);
      }
      return {
        data: this.data,
        variables: this.variables,
        maxLag: this.maxLag,
        type: this.type
      };
    }
  };

  return spec;
}

/**
 * Generate pair plot specification (scatter plot matrix)
 * Shows relationships between parameters
 *
 * @param {Object} trace - MCMC trace object
 * @param {Array<string>} variables - Variable names to plot
 * @param {Object} options - Plot options
 * @returns {Object} Plot specification with .show() method
 */
export function pairPlot(trace, variables = null, options = {}) {
  const traceData = trace.trace || trace;
  const varNames = variables || Object.keys(traceData);

  // Prepare data: create all pairwise combinations
  const plotData = [];
  const n = traceData[varNames[0]].length;

  for (let i = 0; i < n; i++) {
    const point = {};
    for (const varName of varNames) {
      point[varName] = traceData[varName][i];
    }
    plotData.push(point);
  }

  const spec = {
    type: 'pair',
    data: plotData,
    variables: varNames,
    title: options.title || 'Pair Plot',
    width: options.width || 800,
    height: options.height || 800,

    observablePlot: (Plot) => {
      const plots = [];

      // Create grid of scatter plots
      for (let i = 0; i < varNames.length; i++) {
        for (let j = 0; j < varNames.length; j++) {
          const xVar = varNames[j];
          const yVar = varNames[i];

          if (i === j) {
            // Diagonal: histogram
            plots.push({
              x: i,
              y: j,
              type: 'histogram',
              variable: xVar
            });
          } else {
            // Off-diagonal: scatter
            plots.push({
              x: i,
              y: j,
              type: 'scatter',
              xVar: xVar,
              yVar: yVar
            });
          }
        }
      }

      // Note: Full pair plot implementation requires more complex layout
      // This is a simplified version - full implementation would use facets
      return Plot.plot({
        title: spec.title,
        width: spec.width,
        height: spec.height,
        grid: true,
        marks: varNames.flatMap((yVar, i) =>
          varNames.flatMap((xVar, j) => {
            if (i === j) {
              // Diagonal: density
              return Plot.rectY(plotData, Plot.binX(
                { y: 'count' },
                { x: xVar, fill: '#4682b4', fillOpacity: 0.6 }
              ));
            } else {
              // Off-diagonal: scatter
              return Plot.dot(plotData, {
                x: xVar,
                y: yVar,
                fill: '#4682b4',
                fillOpacity: 0.3,
                r: 2
              });
            }
          })
        )
      });
    },

    show(plotLib) {
      if (plotLib && plotLib.plot) {
        return this.observablePlot(plotLib);
      }
      return {
        data: this.data,
        variables: this.variables,
        type: this.type
      };
    }
  };

  return spec;
}

/**
 * Generate forest plot specification
 * Shows posterior summaries with credible intervals
 *
 * @param {Object} trace - MCMC trace object
 * @param {Array<string>} variables - Variable names to plot
 * @param {number} hdi - Highest Density Interval (default 0.95)
 * @param {Object} options - Plot options
 * @returns {Object} Plot specification with .show() method
 */
export function forestPlot(trace, variables = null, hdi = 0.95, options = {}) {
  const traceData = trace.trace || trace;
  const varNames = variables || Object.keys(traceData);

  // Compute statistics
  const plotData = [];
  for (const varName of varNames) {
    const samples = traceData[varName];
    const sorted = [...samples].sort((a, b) => a - b);
    const n = sorted.length;

    const lowerIdx = Math.floor(n * (1 - hdi) / 2);
    const upperIdx = Math.ceil(n * (1 + hdi) / 2) - 1;

    plotData.push({
      variable: varName,
      mean: samples.reduce((a, b) => a + b, 0) / n,
      median: sorted[Math.floor(n / 2)],
      lower: sorted[lowerIdx],
      upper: sorted[upperIdx]
    });
  }

  const spec = {
    type: 'forest',
    data: plotData,
    variables: varNames,
    hdi: hdi,
    title: options.title || `Forest Plot (${(hdi * 100).toFixed(0)}% HDI)`,
    width: options.width || 600,
    height: options.height || 300,

    observablePlot: (Plot) => {
      return Plot.plot({
        title: spec.title,
        width: spec.width,
        height: spec.height,
        marginLeft: 100,
        marks: [
          // HDI intervals
          Plot.ruleX(plotData, {
            x1: 'lower',
            x2: 'upper',
            y: 'variable',
            stroke: '#4682b4',
            strokeWidth: 3
          }),
          // Mean points
          Plot.dot(plotData, {
            x: 'mean',
            y: 'variable',
            fill: 'white',
            stroke: '#4682b4',
            strokeWidth: 2,
            r: 5
          }),
          // Zero reference line
          Plot.ruleX([0], { stroke: 'red', strokeDasharray: '4 4' })
        ],
        x: {
          label: 'Value',
          grid: true
        },
        y: {
          label: null
        }
      });
    },

    show(plotLib) {
      if (plotLib && plotLib.plot) {
        return this.observablePlot(plotLib);
      }
      return {
        data: this.data,
        variables: this.variables,
        hdi: this.hdi,
        type: this.type
      };
    }
  };

  return spec;
}

/**
 * Generate rank plot specification (for convergence diagnostics)
 * Useful for detecting non-stationarity and comparing chains
 *
 * @param {Object} trace - MCMC trace object
 * @param {Array<string>} variables - Variable names to plot
 * @param {Object} options - Plot options
 * @returns {Object} Plot specification with .show() method
 */
export function rankPlot(trace, variables = null, options = {}) {
  const traceData = trace.trace || trace;
  const varNames = variables || Object.keys(traceData);

  // Compute ranks
  const plotData = [];
  for (const varName of varNames) {
    const samples = traceData[varName];

    // Create array of (value, originalIndex) pairs
    const indexed = samples.map((value, idx) => ({ value, idx }));

    // Sort by value and assign ranks
    indexed.sort((a, b) => a.value - b.value);
    const ranks = new Array(samples.length);
    indexed.forEach((item, rank) => {
      ranks[item.idx] = rank;
    });

    // Add to plot data
    ranks.forEach((rank, iteration) => {
      plotData.push({
        variable: varName,
        iteration: iteration,
        rank: rank
      });
    });
  }

  const spec = {
    type: 'rank',
    data: plotData,
    variables: varNames,
    title: options.title || 'Rank Plot',
    width: options.width || 800,
    height: options.height || 400,

    observablePlot: (Plot) => {
      return Plot.plot({
        title: spec.title,
        width: spec.width,
        height: spec.height,
        facet: {
          data: plotData,
          y: 'variable',
          marginLeft: 100
        },
        marks: [
          Plot.dot(plotData, {
            x: 'iteration',
            y: 'rank',
            fill: '#4682b4',
            fillOpacity: 0.5,
            r: 2
          })
        ],
        x: {
          label: 'Iteration',
          grid: true
        },
        y: {
          label: 'Rank',
          grid: true
        }
      });
    },

    show(plotLib) {
      if (plotLib && plotLib.plot) {
        return this.observablePlot(plotLib);
      }
      return {
        data: this.data,
        variables: this.variables,
        type: this.type
      };
    }
  };

  return spec;
}

/**
 * Generate energy plot specification (for HMC/NUTS diagnostics)
 * Compares the distribution of energy transitions
 *
 * @param {Object} trace - MCMC trace object with energy information
 * @param {Object} options - Plot options
 * @returns {Object} Plot specification with .show() method
 */
export function energyPlot(trace, options = {}) {
  // Energy plot requires energy information from HMC/NUTS
  // This is a placeholder - full implementation would need energy tracking

  const spec = {
    type: 'energy',
    data: [],
    title: options.title || 'Energy Plot',
    width: options.width || 600,
    height: options.height || 400,

    observablePlot: (Plot) => {
      return Plot.plot({
        title: 'Energy plot requires HMC/NUTS energy information',
        marks: []
      });
    },

    show(plotLib) {
      console.warn('Energy plot requires energy transition information from HMC/NUTS sampler');
      return {
        data: this.data,
        type: this.type,
        note: 'Not yet implemented - requires energy tracking in samplers'
      };
    }
  };

  return spec;
}

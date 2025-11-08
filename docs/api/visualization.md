---
layout: default
title: Visualization
---

# Visualization Utilities

MCMC visualization tools inspired by [ArviZ](https://arviz-devs.github.io/arviz/).

## Design Philosophy

The visualization system is **library-agnostic**:
1. Generate plot specifications with data and configuration
2. Pass the spec to `.show(Plot)` for Observable Plot, or
3. Call `.show()` without arguments to get raw data for custom plotting

This avoids hard dependencies on plotting libraries while supporting multiple environments.

## Usage Pattern

```javascript
import { tracePlot } from '@tangent.to/mc';

// Create plot specification
const spec = tracePlot(trace, ['alpha', 'beta']);

// Option 1: Use with Observable Plot
spec.show(Plot);  // Returns Observable Plot object

// Option 2: Get data for custom plotting
const data = spec.show();  // Returns { data, variables, type }
```

## Available Plots

### Trace Plot

Shows parameter values over iterations to assess convergence.

```javascript
import { tracePlot } from '@tangent.to/mc';

const spec = tracePlot(trace, variables=['alpha', 'beta'], {
  title: 'MCMC Trace',
  width: 800,
  height: 400
});

// In Observable
spec.show(Plot);
```

**What to look for:**
- Should look like a "fuzzy caterpillar"
- No trends or patterns
- Stable around mean value
- Quick convergence from initial values

### Posterior Plot

Shows posterior distributions with summary statistics.

```javascript
import { posteriorPlot } from '@tangent.to/mc';

const spec = posteriorPlot(trace, ['alpha', 'beta', 'sigma'], {
  title: 'Posterior Distributions',
  width: 800,
  height: 400
});

// Access computed statistics
console.log(spec.stats.alpha.mean);
console.log(spec.stats.alpha.hdi_2_5);  // 2.5th percentile
console.log(spec.stats.alpha.hdi_97_5); // 97.5th percentile

spec.show(Plot);
```

**Features:**
- Histogram of samples
- Mean (red line)
- 95% HDI (black interval)

### Autocorrelation Plot

Shows autocorrelation to assess mixing and effective sample size.

```javascript
import { autocorrPlot } from '@tangent.to/mc';

const spec = autocorrPlot(trace, ['alpha', 'beta'], maxLag=50, {
  title: 'Autocorrelation',
  width: 800,
  height: 400
});

spec.show(Plot);
```

**What to look for:**
- Quick decay to zero indicates good mixing
- High autocorrelation means you need more samples
- Red dashed lines show ±0.05 threshold

### Forest Plot

Shows posterior summaries with credible intervals.

```javascript
import { forestPlot } from '@tangent.to/mc';

const spec = forestPlot(trace, ['alpha', 'beta'], hdi=0.95, {
  title: 'Parameter Estimates',
  width: 600,
  height: 300
});

spec.show(Plot);
```

**Features:**
- Mean (white dot)
- 95% HDI (blue interval)
- Zero reference line (red dashed)

**Use cases:**
- Compare multiple parameters
- Check if intervals include zero
- Publication-ready summaries

### Pair Plot

Shows pairwise relationships between parameters.

```javascript
import { pairPlot } from '@tangent.to/mc';

const spec = pairPlot(trace, ['alpha', 'beta', 'sigma'], {
  title: 'Parameter Relationships',
  width: 800,
  height: 800
});

spec.show(Plot);
```

**What to look for:**
- Diagonal: Marginal distributions
- Off-diagonal: Correlations between parameters
- Identifies multicollinearity

### Rank Plot

Shows rank distribution for convergence diagnostics.

```javascript
import { rankPlot } from '@tangent.to/mc';

const spec = rankPlot(trace, ['alpha', 'beta'], {
  title: 'Rank Plot',
  width: 800,
  height: 400
});

spec.show(Plot);
```

**Use cases:**
- Detect non-stationarity
- Compare multiple chains
- Visual convergence diagnostic

## Complete Example: Observable

```javascript
// In an Observable notebook

// 1. Import library
mc = await import("https://cdn.jsdelivr.net/npm/@tangent.to/mc/src/browser.js")

// 2. Run your model
{
  const { Model, Normal, MetropolisHastings, tracePlot, posteriorPlot } = mc;

  // Build and sample model
  const model = new Model();
  const x = new Normal(0, 1, 'x');
  model.addVariable('x', x);

  const sampler = new MetropolisHastings(0.5);
  const trace = sampler.sample(model, {x: 0}, 1000, 500, 1);

  return { trace };
}

// 3. Create visualizations
trace_plot = mc.tracePlot(trace, ['x'])
posterior_plot = mc.posteriorPlot(trace, ['x'])

// 4. Show plots
trace_plot.show(Plot)
posterior_plot.show(Plot)
```

## Complete Example: Node.js

```javascript
import {
  Model,
  Normal,
  MetropolisHastings,
  tracePlot,
  posteriorPlot,
  printSummary
} from '@tangent.to/mc';

// Run model
const model = new Model();
// ... define model ...
const trace = sampler.sample(model, init, 1000, 500, 1);

// Print summary
printSummary(trace);

// Create plots
const trace_spec = tracePlot(trace);
const posterior_spec = posteriorPlot(trace);

// Export data for custom plotting
const trace_data = trace_spec.show();
const posterior_data = posterior_spec.show();

// Use with any plotting library
console.log(trace_data);      // { data, variables, type }
console.log(posterior_data);  // { data, stats, variables, type }
```

## Complete Example: Deno

```javascript
import {
  tracePlot,
  posteriorPlot,
  forestPlot
} from "npm:@tangent.to/mc@0.2.0";

// After sampling...
const trace_spec = tracePlot(trace, ['alpha', 'beta']);
const posterior_spec = posteriorPlot(trace);
const forest_spec = forestPlot(trace);

// Get data
const data = trace_spec.show();
console.log(data);

// Or if you have Observable Plot in Deno
trace_spec.show(Plot);
```

## Data Export Structure

All plot specs return consistent data structures when called without arguments:

```javascript
const spec = tracePlot(trace);
const data = spec.show();

// Returns:
{
  type: 'trace',           // Plot type
  data: [...],            // Array of data points
  variables: ['x', 'y'],  // Variable names
  // ... additional metadata
}
```

### Trace Plot Data

```javascript
[
  { variable: 'alpha', iteration: 0, value: 1.2 },
  { variable: 'alpha', iteration: 1, value: 1.3 },
  { variable: 'beta', iteration: 0, value: 2.1 },
  // ...
]
```

### Posterior Plot Data

```javascript
{
  data: [
    { variable: 'alpha', value: 1.2 },
    { variable: 'alpha', value: 1.3 },
    // ...
  ],
  stats: {
    alpha: {
      mean: 1.25,
      median: 1.24,
      hdi_2_5: 1.1,
      hdi_97_5: 1.4
    }
  }
}
```

### Forest Plot Data

```javascript
[
  {
    variable: 'alpha',
    mean: 1.25,
    median: 1.24,
    lower: 1.1,
    upper: 1.4
  },
  // ...
]
```

## Integration with Other Libraries

### D3.js

```javascript
const spec = tracePlot(trace);
const { data } = spec.show();

// Use D3 to plot
const svg = d3.select("svg");
svg.selectAll("path")
  .data(d3.group(data, d => d.variable))
  .enter()
  .append("path")
  .attr("d", d3.line()
    .x(d => xScale(d.iteration))
    .y(d => yScale(d.value))
  );
```

### Plotly

```javascript
const spec = posteriorPlot(trace);
const { data } = spec.show();

// Group by variable
const traces = Array.from(
  d3.group(data, d => d.variable),
  ([name, values]) => ({
    x: values.map(d => d.value),
    type: 'histogram',
    name: name
  })
);

Plotly.newPlot('plot', traces);
```

### Vega-Lite

```javascript
const spec = tracePlot(trace);
const { data, variables } = spec.show();

const vegaSpec = {
  $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
  data: { values: data },
  mark: 'line',
  encoding: {
    x: { field: 'iteration', type: 'quantitative' },
    y: { field: 'value', type: 'quantitative' },
    color: { field: 'variable', type: 'nominal' }
  },
  facet: {
    row: { field: 'variable' }
  }
};
```

## Best Practices

1. **Always check trace plots first** - Ensure convergence before interpreting results
2. **Use autocorrelation to set thinning** - Thin until autocorrelation drops below 0.05
3. **Compare forest plots across models** - For model selection
4. **Check pair plots for correlations** - Identify parameter dependencies
5. **Export data for publication** - Use `.show()` to get data for final plots

## Tips

- **Observable**: Plots render immediately with `spec.show(Plot)`
- **Node.js**: Export data and use your preferred plotting library
- **Deno**: Same as Node.js, works with any plotting system
- **R/Python**: Export to JSON and load in your analysis pipeline

## Coming Soon

- **Energy plot** - HMC/NUTS diagnostic (requires energy tracking)
- **Comparison plots** - Compare multiple model runs
- **Predictive plots** - Posterior predictive checks
- **LOO plots** - Leave-one-out cross-validation

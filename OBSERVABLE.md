# Using mc in ObservableHQ

mc is fully compatible with Observable notebooks, allowing you to perform Bayesian inference and visualize results interactively in the browser.

## Quick Start

### 1. Import mc

In an Observable cell, you can import mc directly from npm. A single import works — mc has no peer dependency to resolve:

```javascript
mc = import("https://cdn.jsdelivr.net/npm/@tangent.to/mc/+esm")
```

Or using dynamic import:

```javascript
{
  const module = await import("https://cdn.jsdelivr.net/npm/@tangent.to/mc/+esm");
  return module;
}
```

### 2. Basic Example: Bayesian Linear Regression

```javascript
{
  const { Model, Normal, Uniform, MetropolisHastings } = await import("@tangent.to/mc");

  // Generate synthetic data
  const n = 50;
  const x = Array.from({length: n}, () => Math.random() * 10);
  const y = x.map(xi => 2 + 3 * xi + (Math.random() - 0.5) * 2);

  // Define model
  const model = new Model('linear_regression');

  const alpha = new Normal(0, 10, 'alpha');
  const beta = new Normal(0, 10, 'beta');
  const sigma = new Uniform(0.01, 5, 'sigma');

  model.addVariable('alpha', alpha);
  model.addVariable('beta', beta);
  model.addVariable('sigma', sigma);

  // Custom likelihood
  model.logProb = function(params) {
    // Implementation here...
  };

  // Run sampling
  const sampler = new MetropolisHastings(0.5);
  const trace = sampler.sample(model, {alpha: 0, beta: 0, sigma: 1}, 1000, 500, 1);

  return trace;
}
```

### 3. Visualizing Results with Plot

Observable's Plot library works great with mc traces:

```javascript
Plot.plot({
  marks: [
    Plot.line(trace.trace.alpha.map((val, i) => ({iteration: i, value: val})), {
      x: "iteration",
      y: "value",
      stroke: "steelblue"
    }),
    Plot.ruleY([0])
  ],
  y: {label: "Alpha"},
  x: {label: "Iteration"}
})
```

## Browser-Specific Considerations

### Memory

Nothing to manage. mc computes on plain numbers and arrays, so there are no tensors to
dispose and no `tf.tidy()` to wrap anything in — ordinary garbage collection applies.

A long run's memory is dominated by the trace itself: one number per draw per scalar
parameter, plus the vector parameters. Thin the chain (`{ thin: 5 }`) if a browser tab
needs to hold a very long one.

There is a single browser-first build, identical in Observable, the browser, Node and
Deno.

### Performance

Browser-based MCMC is slower than Node.js:
- Use fewer samples (500-1000 instead of 5000)
- Reduce burn-in period
- Consider using HMC instead of Metropolis-Hastings for efficiency
- Enable WebGL acceleration if available

### Saving Results

Since there's no filesystem in the browser, use the browser-compatible serialization. File-based persistence (`saveTrace`, `loadTrace`, `saveModelState`, `exportTraceForBrowser`, `importTraceFromJSON`) lives in a Node-only module that imports `node:fs` and is not importable in the browser. Instead use `traceToJSON(trace)` (returns a JSON string) and `traceToCSV(...)`, which are exported from the main entry `@tangent.to/mc`:

```javascript
{
  const { traceToJSON } = await import("@tangent.to/mc");

  const jsonString = traceToJSON(trace);

  // Create download link
  const blob = new Blob([jsonString], { type: 'application/json' });
  const url = URL.createObjectURL(blob);

  return html`<a href="${url}" download="trace.json">Download Trace</a>`;
}
```

### Loading Saved Traces

```javascript
{
  // User uploads JSON file
  const file = await Inputs.file({accept: ".json"});
  const text = await file.text();
  const trace = JSON.parse(text);

  return trace;
}
```

## Interactive Posterior Predictive Checks

One of the best features of Observable is interactive exploration:

```javascript
viewof nSamples = Inputs.range([10, 1000], {
  value: 100,
  step: 10,
  label: "Number of posterior samples"
})
```

```javascript
{
  // Use the slider value to control predictions
  const predictions = model.predictPosterior(
    trace,
    params => params.alpha + params.beta * testX,
    nSamples
  );

  return predictions;
}
```

## Full Observable Notebook Template

Here's a complete template for an Observable notebook:

```javascript
// Cell 1: Import library
mc = import("https://cdn.jsdelivr.net/npm/@tangent.to/mc/+esm")
```

```javascript
// Cell 2: Generate data
data = {
  const n = 50;
  const x = Array.from({length: n}, () => Math.random() * 10);
  const y = x.map(xi => 2 + 3 * xi + (Math.random() - 0.5) * 2);
  return {x, y};
}
```

```javascript
// Cell 3: Define and fit model
trace = {
  const { Model, Normal, Uniform, MetropolisHastings } = mc;

  const model = new Model('linear_regression');

  // Define priors
  model.addVariable('alpha', new Normal(0, 10));
  model.addVariable('beta', new Normal(0, 10));
  model.addVariable('sigma', new Uniform(0.01, 5));

  // Likelihood
  model.logProb = function(params) {
    // ... implementation
  };

  // Sample
  const sampler = new MetropolisHastings(0.5);
  return sampler.sample(model, {alpha: 0, beta: 0, sigma: 1}, 500, 250, 1);
}
```

```javascript
// Cell 4: Trace plots
Plot.plot({
  facet: {
    data: [
      ...trace.trace.alpha.map((v, i) => ({variable: 'alpha', iteration: i, value: v})),
      ...trace.trace.beta.map((v, i) => ({variable: 'beta', iteration: i, value: v}))
    ],
    y: "variable"
  },
  marks: [
    Plot.line({x: "iteration", y: "value"})
  ]
})
```

```javascript
// Cell 5: Posterior distributions
Plot.plot({
  marks: [
    Plot.rectY(trace.trace.alpha, Plot.binX({y: "count"}, {x: d => d, fill: "steelblue"}))
  ],
  x: {label: "Alpha"},
  y: {label: "Count"}
})
```

## Tips for Observable

1. **Break into cells**: Each step (import, data, model, sampling, visualization) should be a separate cell
2. **Use viewof for interactivity**: Create sliders and inputs to explore posteriors
3. **Cache expensive operations**: Observable's reactivity will re-run cells, so cache MCMC traces
4. **Show progress**: Use Observable's yield to show progress during sampling
5. **Visualize uncertainty**: Always plot credible intervals, not just point estimates

## Resources

- [Observable Plot Documentation](https://observablehq.com/plot/)
- [Observable Inputs](https://observablehq.com/@observablehq/inputs)
- [mc Examples](https://github.com/tangent-to/mc/tree/main/examples)

## Example Notebooks

Coming soon:
- Bayesian Linear Regression
- Hierarchical Models with Partial Pooling
- A/B Testing with Bayesian Statistics

# Using JSMC in ObservableHQ

JSMC is fully compatible with Observable notebooks, allowing you to perform Bayesian inference and visualize results interactively in the browser.

## Quick Start

### 1. Import JSMC

In an Observable cell, you can import JSMC directly from npm:

```javascript
jsmc = import("https://cdn.jsdelivr.net/npm/jsmc@0.2.0/src/browser.js")
```

Or using dynamic import:

```javascript
{
  const module = await import("https://cdn.jsdelivr.net/npm/jsmc/src/browser.js");
  return module;
}
```

### 2. Basic Example: Bayesian Linear Regression

```javascript
{
  const { Model, Normal, Uniform, MetropolisHastings } = await import("jsmc/browser");

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

Observable's Plot library works great with JSMC traces:

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

### 4. Gaussian Process Visualization

```javascript
{
  const { GaussianProcess, RBF } = await import("jsmc/browser");

  // Training data
  const X_train = [[-2], [-1], [0], [1], [2]];
  const y_train = X_train.map(([x]) => Math.sin(x) + Math.random() * 0.1);

  // Fit GP
  const kernel = new RBF(1.0, 1.0);
  const gp = new GaussianProcess(0, kernel, 0.05);
  gp.fit(X_train, y_train);

  // Predict
  const X_test = Array.from({length: 100}, (_, i) => [(i - 50) / 10]);
  const predictions = gp.predict(X_test, true);

  // Prepare data for plotting
  const data = X_test.map(([x], i) => ({
    x: x,
    mean: predictions.mean[i],
    lower: predictions.mean[i] - 2 * predictions.std[i],
    upper: predictions.mean[i] + 2 * predictions.std[i]
  }));

  return data;
}
```

Then visualize with confidence bands:

```javascript
Plot.plot({
  marks: [
    // Confidence band
    Plot.areaY(gpData, {
      x: "x",
      y1: "lower",
      y2: "upper",
      fill: "steelblue",
      fillOpacity: 0.2
    }),
    // Mean prediction
    Plot.line(gpData, {
      x: "x",
      y: "mean",
      stroke: "steelblue"
    }),
    // Training points
    Plot.dot(trainingData, {
      x: "x",
      y: "y",
      fill: "red"
    })
  ],
  y: {label: "y"},
  x: {label: "x"}
})
```

## Browser-Specific Considerations

### Memory Management

TensorFlow.js in the browser requires manual memory management. Use `tf.tidy()` or dispose tensors:

```javascript
// The GP and samplers already use tf.tidy() internally,
// but be mindful when creating custom operations
```

### Performance

Browser-based MCMC is slower than Node.js:
- Use fewer samples (500-1000 instead of 5000)
- Reduce burn-in period
- Consider using HMC instead of Metropolis-Hastings for efficiency
- Enable WebGL acceleration if available

### Saving Results

Since there's no filesystem in the browser, use the browser-compatible persistence:

```javascript
{
  const { exportTraceForBrowser } = await import("jsmc/browser");

  const jsonString = exportTraceForBrowser(trace);

  // Create download link
  const blob = new Blob([jsonString], { type: 'application/json' });
  const url = URL.createObjectURL(blob);

  return html`<a href="${url}" download="trace.json">Download Trace</a>`;
}
```

### Loading Saved Traces

```javascript
{
  const { importTraceFromJSON } = await import("jsmc/browser");

  // User uploads JSON file
  const file = await Inputs.file({accept: ".json"});
  const text = await file.text();
  const trace = importTraceFromJSON(text);

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
jsmc = import("https://cdn.jsdelivr.net/npm/jsmc/src/browser.js")
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
  const { Model, Normal, Uniform, MetropolisHastings } = jsmc;

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
- [JSMC Examples](https://github.com/essicolo/jsmc/tree/main/examples)

## Example Notebooks

Coming soon:
- Bayesian Linear Regression
- Gaussian Process Regression with Interactive Kernel Selection
- Hierarchical Models with Partial Pooling
- A/B Testing with Bayesian Statistics

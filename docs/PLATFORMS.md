# Platform Support: Observable, Deno, and Jupyter

mc is designed to work across multiple JavaScript/TypeScript platforms. This guide covers the three primary use cases.

## Priority 1: ObservableHQ

Observable is the primary target platform for mc, enabling interactive Bayesian data analysis in the browser.

### Quick Start

```javascript
// Import mc
mc = import("https://cdn.jsdelivr.net/npm/@tangent.to/mc/+esm")

// Use it
{
  const { Model, Normal, MetropolisHastings } = mc;
  // Define your model...
}
```

### Complete Example

```javascript
// Cell 1: Import
mc = import("https://cdn.jsdelivr.net/npm/@tangent.to/mc/+esm")

// Cell 2: Generate data
data = {
  const n = 100;
  const x = Array.from({length: n}, () => Math.random() * 10);
  const y = x.map(xi => 2 + 3 * xi + (Math.random() - 0.5));
  return {x, y};
}

// Cell 3: Run MCMC
trace = {
  const { Model, Normal, Uniform, MetropolisHastings } = mc;

  const model = new Model('regression');
  model.addVariable('alpha', new Normal(0, 10));
  model.addVariable('beta', new Normal(0, 10));
  model.addVariable('sigma', new Uniform(0.01, 5));

  // Define likelihood
  model.logProb = function(params) {
    // Custom implementation
  };

  const sampler = new MetropolisHastings(0.5);
  return sampler.sample(model, {alpha: 0, beta: 0, sigma: 1}, 500, 250, 1);
}

// Cell 4: Visualize with Plot
Plot.plot({
  marks: [
    Plot.line(trace.trace.alpha.map((v, i) => ({i, v})), {x: "i", y: "v"}),
    Plot.ruleY([2])
  ]
})
```

### Observable Tips

1. **Memory**: nothing to dispose — computation is on plain arrays — but a long trace still holds one number per draw per parameter, so thin it if a tab must keep one
2. **Sample Counts**: Use fewer samples than in Node.js (500-1000 vs 5000)
3. **Reactivity**: Cache expensive MCMC runs in separate cells
4. **Visualization**: Use Observable Plot for trace plots and posteriors
5. **Download Results**: Use `traceToJSON()` to serialize, then download

### Performance

- Linear regression: ~1000 samples/minute
- Browser can handle models with up to ~10 parameters efficiently

See **[docs/OBSERVABLE.md](OBSERVABLE.md)** for comprehensive examples.

## Priority 2: Deno REPL (Zed Editor)

Deno provides a modern TypeScript/JavaScript runtime with excellent developer experience.

### Setup

mc works with Deno using npm specifiers:

```typescript
// Import from npm
import { Model, Normal, MetropolisHastings } from "npm:@tangent.to/mc";

// Or use JSR
// import { Model, Normal, MetropolisHastings } from "jsr:@tangent/mc";
```

### Example in Deno REPL

```typescript
// Start Deno REPL
// deno

import { Model, Normal, Uniform, MetropolisHastings } from "npm:@tangent.to/mc";

// Generate data
const x = Array.from({length: 50}, () => Math.random() * 10);
const y = x.map(xi => 2 + 3 * xi + (Math.random() - 0.5));

// Define model
const model = new Model('linear_regression');
model.addVariable('alpha', new Normal(0, 10));
model.addVariable('beta', new Normal(0, 10));
model.addVariable('sigma', new Uniform(0.01, 5));

// Custom likelihood
model.logProb = function(params) {
  // Implementation
};

// Run sampling
const sampler = new MetropolisHastings(0.5);
const trace = sampler.sample(model, {alpha: 0, beta: 0, sigma: 1}, 1000, 500, 1);

// View results
console.log('Posterior means:');
console.log('alpha:', trace.trace.alpha.reduce((a,b) => a+b) / trace.trace.alpha.length);
console.log('beta:', trace.trace.beta.reduce((a,b) => a+b) / trace.trace.beta.length);
```

### Deno Compatibility Notes

**Works out of the box**:
- All mc functionality
- npm imports via `npm:` specifier
- JSR imports via `jsr:@tangent/mc`
- TypeScript types (if provided)

**Considerations**:
- Use `npm:@tangent.to/mc` not `./src/index.js` (no local imports without deno.json)
- No peer dependency to resolve: mc's only dependencies are `@tangent.to/proba` and `@tangent.to/grad`, both pulled in automatically
- Performance: Similar to Node.js for most operations
- File I/O: Use Deno's built-in APIs, not Node.js `fs`

### Zed Editor Integration

In Zed with Deno REPL:

1. Open JavaScript/TypeScript file
2. Start Deno REPL: `Cmd+Shift+P` -> "Deno: Start REPL"
3. Run selections with `Cmd+Enter`
4. Results appear inline

```typescript
// example.ts in Zed

import { Model, Normal } from "npm:@tangent.to/mc";

const model = new Model('test');
model.addVariable('mu', new Normal(0, 1));

// Select and run this line to see output
const samples = model.samplePrior(10);
console.log(samples); // Shows inline in Zed
```

### Deno Script

Create a standalone Deno script:

```typescript
// analysis.ts
import { Model, Normal, MetropolisHastings, printSummary } from "npm:@tangent.to/mc";

async function main() {
  const model = new Model('analysis');
  // ... define model

  const sampler = new MetropolisHastings(0.5);
  const trace = sampler.sample(model, initialValues, 2000, 1000, 1);

  printSummary(trace);

  // Save results
  await Deno.writeTextFile(
    "results.json",
    JSON.stringify(trace, null, 2)
  );
}

main();
```

Run with:
```bash
deno run --allow-write analysis.ts
```

## Priority 3: Jupyter Lab (Node.js Kernel)

Use mc in Jupyter notebooks with the IJavascript kernel for literate programming and reproducible research.

### Setup

1. Install IJavascript kernel:
```bash
npm install -g ijavascript
ijsinstall
```

2. Install mc:
```bash
npm install @tangent.to/mc
```

3. Start Jupyter:
```bash
jupyter lab
```

4. Create new notebook with JavaScript (Node.js) kernel

### Jupyter Notebook Example

```javascript
// Cell 1: Import
const mc = await import('@tangent.to/mc');
const { Model, Normal, Uniform, MetropolisHastings, printSummary } = mc;

// Cell 2: Generate synthetic data
const n = 100;
const x = Array.from({length: n}, () => Math.random() * 10);
const y = x.map(xi => 2 + 3 * xi + (Math.random() - 0.5) * 2);

console.log(`Generated ${n} data points`);

// Cell 3: Define Bayesian model
const model = new Model('linear_regression');
model.addVariable('alpha', new Normal(0, 10));
model.addVariable('beta', new Normal(0, 10));
model.addVariable('sigma', new Uniform(0.01, 5));

// Custom log probability (likelihood)
const originalLogProb = model.logProb.bind(model);
model.logProb = function(params) {
  let logProb = originalLogProb(params);

  // Add likelihood
  for (let i = 0; i < n; i++) {
    const mu = params.alpha + params.beta * x[i];
    const likelihood = new Normal(mu, params.sigma);
    logProb = logProb.add(likelihood.logProb(y[i]));
  }

  return logProb;
};

console.log('Model defined');
console.log(model.summary());

// Cell 4: Run MCMC sampling
const sampler = new MetropolisHastings(0.5);
console.log('Starting MCMC sampling...');

const trace = sampler.sample(
  model,
  { alpha: 0, beta: 0, sigma: 1 },
  2000,  // samples
  1000,  // burn-in
  2      // thin
);

// Cell 5: Analyze results
printSummary(trace);

// Cell 6: Posterior predictive checks
const x_new = 5;
const predictions = model.predictPosteriorSummary(
  trace,
  (params) => params.alpha + params.beta * x_new,
  0.95
);

console.log(`Prediction at x=${x_new}:`);
console.log(`  Mean: ${predictions.mean.toFixed(2)}`);
console.log(`  95% CI: [${predictions.lower.toFixed(2)}, ${predictions.upper.toFixed(2)}]`);

// Cell 7: Save results (Node-only persistence subpath)
const { saveTrace } = await import('@tangent.to/mc/persistence');
saveTrace(trace, 'trace.json');
console.log('Results saved to trace.json');
```

### Visualization in Jupyter

For plots, you can:

1. **Use $$.svg()** for inline SVG:
```javascript
const createTracePlot = (samples) => {
  const width = 600, height = 300;
  const max_val = Math.max(...samples);
  const min_val = Math.min(...samples);

  const points = samples.map((val, i) => {
    const x = (i / samples.length) * width;
    const y = height - ((val - min_val) / (max_val - min_val)) * height;
    return `${x},${y}`;
  }).join(' ');

  return `<svg width="${width}" height="${height}">
    <polyline points="${points}"
      fill="none" stroke="steelblue" stroke-width="1"/>
  </svg>`;
};

$$.svg(createTracePlot(trace.trace.alpha));
```

2. **Use plotly.js** via CDN:
```javascript
$$.async();

const plotly = await import('plotly.js-dist-min');

const data = [{
  y: trace.trace.alpha,
  type: 'scatter',
  mode: 'lines',
  name: 'Alpha trace'
}];

const layout = { title: 'MCMC Trace Plot' };

plotly.newPlot('trace-plot', data, layout);
$$.sendResult();
```

3. **Export to Python**: Save JSON and load in Python cell:
```python
# Python cell
import json
import matplotlib.pyplot as plt

with open('trace.json') as f:
    trace = json.load(f)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(trace['trace']['alpha'])
plt.title('Alpha trace')
plt.subplot(1, 2, 2)
plt.hist(trace['trace']['alpha'], bins=30)
plt.title('Alpha posterior')
plt.show()
```

### Jupyter Tips

1. **Async/Await**: Use `await import()` for ES modules
2. **Display**: Use `$$.text()`, `$$.html()`, `$$.svg()` for rich output
3. **Errors**: Errors show inline with stack traces
4. **Memory**: IJavascript has same V8 limits as Node.js
5. **Performance**: Full Node.js performance (better than browser)
6. **Integration**: Easy to mix JavaScript and Python cells

## Cross-Platform Comparison

| Feature | Observable | Deno REPL | Jupyter Lab |
|---------|-----------|-----------|-------------|
| **Use Case** | Interactive viz | Quick experiments | Reproducible research |
| **Speed** | Slower (browser) | Fast (native) | Fast (Node.js) |
| **Visualization** | Excellent (Plot) | Terminal only | Good (with setup) |
| **Sharing** | Easy (notebooks) | Scripts | Notebooks |
| **Dependencies** | CDN only | npm specifiers | npm packages |
| **Best For** | Exploration, teaching | Development, REPL | Research, reports |

## Installation Methods

### Observable
```javascript
import("https://cdn.jsdelivr.net/npm/@tangent.to/mc/+esm")
```

### Deno
```typescript
import { ... } from "npm:@tangent.to/mc";
```

### Node.js/Jupyter
```bash
npm install @tangent.to/mc
```
```javascript
import { ... } from '@tangent.to/mc';
```

## Performance Guidelines

### Sample Recommendations by Platform

| Platform | Simple Model | Hierarchical |
|----------|--------------|--------------|
| Observable | 500-1000 | 1000-2000 |
| Deno | 2000-5000 | 2000-5000 |
| Jupyter | 2000-5000 | 2000-5000 |

### Memory Limits

- **Observable**: ~2GB browser limit
- **Deno**: System memory
- **Jupyter**: System memory

## Troubleshooting

### Observable Issues

**Problem**: "Module not found"
- **Solution**: Use full CDN URL with version

**Problem**: Out of memory
- **Solution**: Reduce sample count, use thinning

### Deno Issues

**Problem**: "Cannot find module"
- **Solution**: Use `npm:@tangent.to/mc` not relative path

**Problem**: Permission denied
- **Solution**: Add `--allow-read --allow-write` flags

### Jupyter Issues

**Problem**: "Cannot use import outside module"
- **Solution**: Use `await import()` syntax

**Problem**: Kernel crashes
- **Solution**: Restart kernel, reduce sample count

## Best Practices

### Observable
1. Cache expensive computations
2. Use viewof for interactive parameters
3. Break into small cells
4. Use Plot for visualization
5. Export results as JSON

### Deno
1. Use TypeScript for type safety
2. Leverage Deno's built-in formatter
3. Use Deno.test() for unit tests
4. Pin npm versions
5. Use deno.json for configuration

### Jupyter
1. One concept per cell
2. Add markdown explanations
3. Use printSummary() liberally
4. Save intermediates to disk
5. Mix with Python for viz

## Additional Resources

- [Deno Manual](https://deno.land/manual)
- [IJavascript Docs](https://github.com/n-riesco/ijavascript)
- [mc API Docs](api/index.html)

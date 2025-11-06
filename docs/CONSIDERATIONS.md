# Additional Considerations for JSMC

This document covers important considerations, best practices, limitations, and future directions for JSMC.

## Architecture & Design Decisions

### Why TensorFlow.js?

JSMC uses TensorFlow.js as its computational backend for several key reasons:

1. **Automatic Differentiation**: Essential for gradient-based samplers like HMC
2. **Cross-Platform**: Works in Node.js and browsers
3. **GPU Acceleration**: Optional WebGL backend for faster computation
4. **Mature Ecosystem**: Well-tested, widely adopted library

### Why Not Pure JavaScript?

While pure JavaScript would have fewer dependencies, it would require:
- Manual gradient computation (error-prone and slow)
- Custom linear algebra implementations
- No GPU acceleration
- Significantly more development time

The trade-off is worth it for the performance and reliability TensorFlow.js provides.

### PyMC Comparison

| Feature | PyMC | JSMC | Notes |
|---------|------|------|-------|
| Language | Python | JavaScript | JSMC brings Bayesian inference to JS ecosystem |
| Backend | Aesara/JAX | TensorFlow.js | Both support autodiff |
| DAG Structure | ✓ | ✓ | Core feature for both |
| MCMC Samplers | NUTS, HMC, MH, etc. | HMC, MH | JSMC has fewer samplers currently |
| Variational Inference | ✓ | ⚠️ Planned | Major feature gap |
| Model Comparison | WAIC, LOO | ⚠️ Planned | Important for model selection |
| Visualization | ArviZ | External tools | Observable, D3.js recommended |
| Performance | High (JAX/C++) | Medium (JS/WASM) | ~2-5x slower typically |
| Browser Support | ✗ | ✓ | JSMC's key advantage |
| Gaussian Processes | ✓ | ✓ | Both support GPs |

## Performance Considerations

### MCMC Sampling Speed

Typical performance on a modern CPU:

- **Metropolis-Hastings**: ~1000-5000 samples/second (simple models)
- **HMC**: ~100-500 samples/second (depends on gradient complexity)
- **Gaussian Processes**: ~10-100 predictions/second (depends on training set size)

**Optimization Tips**:

1. **Use HMC for high-dimensional problems**: More efficient than MH
2. **Batch operations**: Process multiple chains in parallel if needed
3. **Reduce model complexity**: Simplify likelihood functions
4. **Use GPU**: Enable WebGL backend in browser
5. **Tune sampler parameters**: Proper step size and proposal std make a huge difference

### Memory Management

TensorFlow.js requires explicit memory management:

```javascript
// Good: Use tf.tidy() for automatic cleanup
const result = tf.tidy(() => {
  const x = tf.tensor([1, 2, 3]);
  const y = tf.square(x);
  return y.arraySync(); // Extract value before cleanup
});

// Bad: Memory leak
const x = tf.tensor([1, 2, 3]);
const y = tf.square(x);
// x and y are never disposed!
```

JSMC handles this internally for most operations, but be careful with custom likelihood functions.

### Browser vs Node.js

| Aspect | Browser | Node.js |
|--------|---------|---------|
| Speed | Slower | Faster |
| Memory | Limited | More available |
| GPU | WebGL | CUDA (via tfjs-node-gpu) |
| Use Case | Interactive exploration | Production inference |
| Best For | Visualization, teaching | Large-scale modeling |

## Limitations & Known Issues

### Current Limitations

1. **No NUTS sampler**: The most efficient MCMC algorithm is not yet implemented
2. **Limited distributions**: Fewer than PyMC (no Poisson, Student-t, etc.)
3. **No variational inference**: ADVI and other VI methods not available
4. **Single-chain diagnostics**: Multi-chain R-hat requires manual implementation
5. **No model serialization**: Can't save/load model structure (only traces)
6. **No sparse matrix support**: GPs become slow with >1000 data points

### Performance Bottlenecks

1. **GP Cholesky decomposition**: O(n³) - slow for large datasets
   - **Workaround**: Use inducing points (sparse GPs) - planned feature
2. **HMC gradient computation**: Can be slow for complex models
   - **Workaround**: Use simpler models or MH
3. **JavaScript overhead**: ~2-5x slower than compiled languages
   - **Workaround**: Use WebAssembly backend (experimental)

### Numerical Stability

JSMC includes safeguards for numerical stability:

- **Cholesky jitter**: Small diagonal term (1e-6) added to prevent singular matrices
- **Log-space computations**: Probabilities computed in log space to avoid underflow
- **Gradient clipping**: Optional for HMC to prevent explosions

However, you may still encounter issues with:
- Very small/large parameter values
- Poorly conditioned covariance matrices
- Extreme likelihood ratios

**Solutions**:
- Scale your data (standardize inputs)
- Use informative priors to constrain parameters
- Increase jitter if encountering Cholesky errors

## Best Practices

### Model Design

1. **Start simple**: Begin with simple models, add complexity gradually
2. **Use informative priors**: Even weak priors help with convergence
3. **Standardize data**: Center and scale inputs for better sampling
4. **Check prior predictive**: Sample from prior before running MCMC
5. **Visualize DAG**: Draw out your model structure on paper

### MCMC Sampling

1. **Tune samplers**: Aim for 20-40% acceptance rate (MH), 60-80% (HMC)
2. **Run multiple chains**: Check convergence with R-hat < 1.1
3. **Use adequate burn-in**: At least 500-1000 samples
4. **Check trace plots**: Visual inspection is crucial
5. **Compute ESS**: Effective sample size should be >100 per chain

### Code Organization

```javascript
// Good: Modular, reusable
function createModel(data) {
  const model = new Model('my_model');
  // ... define model
  return model;
}

function runInference(model, options) {
  const sampler = new MetropolisHastings(options.stepSize);
  return sampler.sample(model, options.initial, options.nSamples, options.burnIn);
}

// Bad: Everything in one function
function doEverything() {
  // 200 lines of code...
}
```

## Testing & Validation

### How to Validate Your Model

1. **Prior predictive checks**: Do prior samples look reasonable?
2. **Posterior predictive checks**: Do posterior predictions match data?
3. **Convergence diagnostics**: R-hat, ESS, trace plots
4. **Sensitivity analysis**: How do results change with different priors?
5. **Cross-validation**: Hold out data and check predictions

### Debugging MCMC Issues

**Problem**: Low acceptance rate (<5%)
- **Solution**: Decrease proposal std (MH) or step size (HMC)

**Problem**: High acceptance rate (>95%)
- **Solution**: Increase proposal std or step size (exploring too slowly)

**Problem**: Trace plots look like random walk
- **Solution**: Increase thinning, run longer, check model specification

**Problem**: Parameters hitting boundaries
- **Solution**: Use unbounded parameterization or stronger priors

**Problem**: Multimodal posterior
- **Solution**: Run multiple chains with different initializations

## Future Roadmap

### Planned Features

**High Priority**:
- [ ] NUTS (No-U-Turn Sampler)
- [ ] Additional distributions (Poisson, Student-t, Exponential, Categorical)
- [ ] Variational inference (ADVI)
- [ ] Sparse Gaussian Processes (inducing points)
- [ ] Model comparison (WAIC, LOO)

**Medium Priority**:
- [ ] Parallel chains with web workers
- [ ] Automatic sampler tuning
- [ ] Built-in visualization utilities
- [ ] TypeScript definitions
- [ ] More kernel functions for GPs

**Low Priority**:
- [ ] Custom distributions via class extension
- [ ] PyMC model import/export
- [ ] Hamiltonian Monte Carlo with mass matrix adaptation
- [ ] Sequential Monte Carlo (SMC)

### Community Contributions

We welcome contributions! Priority areas:

1. **More distributions**: Implement standard distributions from R/PyMC
2. **Tests**: Unit tests for all distributions and samplers
3. **Documentation**: More examples, tutorials, blog posts
4. **Benchmarks**: Compare performance with other libraries
5. **Observable notebooks**: Interactive examples

## When NOT to Use JSMC

JSMC may not be the best choice if you need:

1. **Production-scale inference**: Use PyMC, Stan, or JAX
2. **Real-time inference**: MCMC is too slow
3. **Deep learning integration**: Use PyTorch/TensorFlow directly
4. **Complex time series**: Specialized libraries (Prophet, statsmodels) are better
5. **Massive datasets**: JSMC doesn't scale beyond ~10k observations

**Alternatives**:
- **PyMC**: Most feature-complete Bayesian library (Python)
- **Stan**: Fast, robust, production-ready (C++/R/Python)
- **TensorFlow Probability**: Deep learning + Bayesian (Python/JS)
- **Turing.jl**: Fast, flexible (Julia)

## Security Considerations

### User-Provided Data

If accepting user-uploaded data:
- Validate JSON structure
- Limit file sizes
- Sanitize inputs
- Don't execute user-provided code

### Model Serialization

When saving/loading models:
- Only save data, not code
- Validate JSON schema
- Be careful with `eval()` or `Function()` constructor
- Consider signing serialized models

### Browser Security

In Observable or other browser environments:
- JSMC runs client-side (no server execution)
- Data stays in the browser (privacy-friendly)
- Be mindful of CORS when loading external data
- Large computations may freeze the browser

## Resources & Learning

### Books

- **Bayesian Data Analysis** (Gelman et al.) - The definitive textbook
- **Statistical Rethinking** (McElreath) - Accessible introduction
- **Doing Bayesian Data Analysis** (Kruschke) - Great for beginners

### Online Resources

- [PyMC Documentation](https://www.pymc.io/) - Many concepts transfer directly
- [Michael Betancourt's Blog](https://betanalpha.github.io/) - Deep dives into MCMC
- [ArviZ Documentation](https://arviz-devs.github.io/) - Visualization best practices

### Papers

- **NUTS**: Hoffman & Gelman (2014) - The No-U-Turn Sampler
- **HMC**: Neal (2011) - MCMC using Hamiltonian dynamics
- **ADVI**: Kucukelbir et al. (2017) - Automatic variational inference

## Support & Community

- **GitHub Issues**: [github.com/essicolo/jsmc/issues](https://github.com/essicolo/jsmc/issues)
- **Discussions**: Use GitHub Discussions for questions
- **Examples**: Check the `examples/` directory
- **Observable**: Share your notebooks!

## License

JSMC is licensed under Apache-2.0, same as TensorFlow.

## Citation

If you use JSMC in research, please cite:

```bibtex
@software{jsmc2025,
  title = {JSMC: JavaScript Markov Chain Monte Carlo},
  author = {},
  year = {2025},
  url = {https://github.com/essicolo/jsmc},
  note = {A PyMC-inspired probabilistic programming library for JavaScript}
}
```

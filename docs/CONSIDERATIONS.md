# Additional Considerations for mc

This document covers important considerations, best practices, limitations, and future directions for mc.

## Architecture & Design Decisions

### Why plain numbers and arrays?

mc ran on TensorFlow.js until 0.5.0, for the reasons you would expect: it supplied
automatic differentiation, vectorized tensor math, and a mature ecosystem. That
decision was reversed, and the reversal is worth recording because the original
reasoning looked sound.

What it cost:

- **A peer dependency users had to install and keep singular.** Two copies of tfjs in
  one page break tensor interop, which is a failure mode nobody enjoys diagnosing.
- **A backend to select**, and browser-only GPU acceleration that mattered far less
  than expected: MCMC is dominated by long sequential chains, not by wide tensor
  kernels. A chain is a Markov chain — it cannot be parallelized in time.
- **Manual memory management.** Every model author had to think about `tf.tidy()`.
- **Weight.** A probabilistic programming library that pulls in a deep-learning
  runtime is hard to justify in a notebook.

What replaced each piece:

- **Vectorized math** → distributions broadcast over plain arrays, via
  [proba](https://github.com/tangent-to/proba). At MCMC sizes the tensor overhead
  exceeded the gain.
- **Automatic differentiation** → priors carry analytic `dlogpdf` gradients, and
  likelihoods you write yourself are differentiated by
  [grad](https://github.com/tangent-to/grad), a reverse-mode tape built for this
  suite (0.9.0). Exact, and one likelihood evaluation per gradient.

The parallelism that does help is across CHAINS, which `sampleChains` runs one per
worker — no tensor backend required.

### PyMC Comparison

| Feature | PyMC | mc | Notes |
|---------|------|------|-------|
| Language | Python | JavaScript | mc brings Bayesian inference to JS ecosystem |
| Backend | PyTensor | plain arrays + @tangent.to/grad | Both support autodiff |
| DAG Structure | Yes | Yes | Core feature for both |
| MCMC Samplers | NUTS, HMC, MH, etc. | HMC, MH | mc has fewer samplers currently |
| Variational Inference | Yes | Planned | Major feature gap |
| Model Comparison | WAIC, LOO | Planned | Important for model selection |
| Visualization | ArviZ | External tools | Observable, D3.js recommended |
| Performance | High (JAX/C++) | Medium (JS/WASM) | ~2-5x slower typically |
| Browser Support | No | Yes | mc's key advantage |

## Performance Considerations

### MCMC Sampling Speed

Typical performance on a modern CPU:

- **Metropolis-Hastings**: ~1000-5000 samples/second (simple models)
- **HMC**: ~100-500 samples/second (depends on gradient complexity)

**Optimization Tips**:

1. **Use HMC for high-dimensional problems**: More efficient than MH
2. **Batch operations**: Process multiple chains in parallel if needed
3. **Reduce model complexity**: Simplify likelihood functions
4. **Give the sampler exact gradients**: an `autoPotential` costs one likelihood
   evaluation per gradient where `potential` costs 2·(#params)
5. **Tune sampler parameters**: Proper step size and proposal std make a huge difference

### Memory

Nothing to manage explicitly. Computation is on plain numbers and arrays, so ordinary
garbage collection applies — there are no tensors to dispose.

A long run's memory is dominated by the trace: one number per draw per scalar
parameter, plus the vector parameters. Thin the chain if you need a very long one:

```javascript
sampler.sample(model, init, { nSamples: 100000, thin: 10 });
```

Running several chains multiplies that, so `sampleChains` is where a long run's memory
actually goes.

## Limitations & Known Issues

### Current Limitations

1. **No NUTS sampler**: The most efficient MCMC algorithm is not yet implemented
2. **Limited distributions**: Fewer than PyMC (no Poisson, Student-t, etc.)
3. **No variational inference**: ADVI and other VI methods not available
4. **Single-chain diagnostics**: Multi-chain R-hat requires manual implementation
5. **No model serialization**: Can't save/load model structure (only traces)

### Performance Bottlenecks

1. **HMC gradient computation**: Can be slow for complex models
   - **Workaround**: Use simpler models or MH
2. **JavaScript overhead**: ~2-5x slower than compiled languages
   - **Workaround**: Use WebAssembly backend (experimental)

### Numerical Stability

mc includes safeguards for numerical stability:

- **Log-space computations**: Probabilities computed in log space to avoid underflow
- **Gradient clipping**: Optional for HMC to prevent explosions

However, you may still encounter issues with:
- Very small/large parameter values
- Extreme likelihood ratios

**Solutions**:
- Scale your data (standardize inputs)
- Use informative priors to constrain parameters

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
- [ ] Model comparison (WAIC, LOO)

**Medium Priority**:
- [ ] Parallel chains with web workers
- [ ] Automatic sampler tuning
- [ ] Built-in visualization utilities
- [ ] TypeScript definitions

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

## When NOT to Use mc

mc may not be the best choice if you need:

1. **Production-scale inference**: Use PyMC, Stan, or JAX
2. **Real-time inference**: MCMC is too slow
3. **Deep learning integration**: Use PyTorch or JAX directly
4. **Complex time series**: Specialized libraries (Prophet, statsmodels) are better
5. **Massive datasets**: mc doesn't scale beyond ~10k observations

**Alternatives**:
- **PyMC**: Most feature-complete Bayesian library (Python)
- **Stan**: Fast, robust, production-ready (C++/R/Python)
- **TensorFlow Probability**: Deep learning + Bayesian (Python/JS)
- **NumPyro**: Fast, JAX-backed (Python)
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
- mc runs client-side (no server execution)
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

- **GitHub Issues**: [github.com/tangent-to/mc/issues](https://github.com/tangent-to/mc/issues)
- **Discussions**: Use GitHub Discussions for questions
- **Examples**: Check the `examples/` directory
- **Observable**: Share your notebooks!

## License

mc is licensed under GPL-3.0, as the application layer of the tangent suite. The
numeric leaves it builds on (proba, grad) are MIT.

## Citation

If you use mc in research, please cite:

```bibtex
@software{mc2025,
  title = {mc: JavaScript Markov Chain Monte Carlo},
  author = {},
  year = {2025},
  url = {https://github.com/tangent-to/mc},
  note = {A PyMC-inspired probabilistic programming library for JavaScript}
}
```

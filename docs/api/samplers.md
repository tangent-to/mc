---
layout: default
title: MCMC Samplers
---

# MCMC Samplers

MCMC (Markov Chain Monte Carlo) samplers generate samples from the posterior distribution $p(\theta|y)$ by constructing a Markov chain that has the posterior as its stationary distribution.

## Metropolis-Hastings

A simple but effective random-walk MCMC sampler.

### Algorithm

At each iteration $t$:

1. Propose a new state: $\theta' \sim q(\theta'|\theta^{(t)}) = \mathcal{N}(\theta^{(t)}, \sigma^2I)$

2. Compute acceptance probability:

$$
\alpha = \min\left(1, \frac{p(\theta'|y)}{p(\theta^{(t)}|y)}\right)
$$

3. Accept with probability $\alpha$:

$$
\theta^{(t+1)} = \begin{cases}
\theta' & \text{with probability } \alpha \\
\theta^{(t)} & \text{with probability } 1-\alpha
\end{cases}
$$

### Usage

```javascript
import { MetropolisHastings } from '@tangent.to/mc';

const sampler = new MetropolisHastings(proposalStd=0.5);

const trace = sampler.sample(
  model,
  initialValues,
  nSamples=1000,
  burnIn=500,
  thin=1
);
```

**Parameters:**
- `proposalStd` - Standard deviation of the Gaussian proposal distribution

**Sampling parameters:**
- `nSamples` - Number of samples to collect (after burn-in)
- `burnIn` - Number of initial samples to discard
- `thin` - Keep every nth sample (reduces autocorrelation)

### Tuning

**Target acceptance rates:**
- **1D problems:** ~44%
- **High-dimensional problems:** ~23.4%

**Tuning the proposal:**
- If acceptance rate too high → increase `proposalStd` (take larger steps)
- If acceptance rate too low → decrease `proposalStd` (take smaller steps)

```javascript
// Adaptive tuning
const currentRate = trace.acceptanceRate;
sampler.tuneProposal(currentRate);
```

### When to use

- **Simple models** with few parameters
- **Initial exploration** of a model
- When gradients are **not available** or expensive
- Models with **discrete** parameters

## Hamiltonian Monte Carlo (HMC)

A gradient-based sampler that uses Hamiltonian dynamics for efficient exploration.

### Algorithm

HMC simulates a physical system where parameters are positions and we introduce auxiliary momentum variables.

**Hamiltonian:**

$$
H(\theta, p) = -\log p(\theta|y) + \frac{1}{2}p^Tp
$$

where $\theta$ is position (parameters) and $p$ is momentum.

**Leapfrog integrator** (preserves volume and is reversible):

1. Half-step for momentum:

$$
p_{i+1/2} = p_i + \frac{\epsilon}{2}\nabla_\theta \log p(\theta_i|y)
$$

2. Full-step for position:

$$
\theta_{i+1} = \theta_i + \epsilon p_{i+1/2}
$$

3. Half-step for momentum:

$$
p_{i+1} = p_{i+1/2} + \frac{\epsilon}{2}\nabla_\theta \log p(\theta_{i+1}|y)
$$

Repeat steps 2-3 for $L$ leapfrog steps.

**Metropolis acceptance:**

$$
\alpha = \min(1, \exp(H(\theta^{(t)}, p^{(t)}) - H(\theta', p')))
$$

### Usage

```javascript
import { HamiltonianMC } from '@tangent.to/mc';

const sampler = new HamiltonianMC(stepSize=0.01, nSteps=10);

const trace = sampler.sample(
  model,
  initialValues,
  nSamples=1000,
  burnIn=500,
  thin=1
);
```

**Parameters:**
- `stepSize` ($\epsilon$) - Leapfrog integration step size
- `nSteps` ($L$) - Number of leapfrog steps per iteration

### Tuning

**Step size ($\epsilon$):**
- Too large → numerical instability, low acceptance
- Too small → inefficient exploration
- Typical range: 0.001 - 0.1

**Number of steps ($L$):**
- Controls how far the proposal moves
- Trade-off: more steps = farther proposals but more computation
- Typical range: 5 - 50

**Target acceptance rate:** ~65%

### When to use

- **Complex models** with many parameters
- When **gradients are available** (requires TensorFlow.js)
- Need **faster convergence** than Metropolis-Hastings
- **Hierarchical models**

## Comparison

| Feature | Metropolis-Hastings | Hamiltonian MC |
|---------|-------------------|----------------|
| Requires gradients | No | Yes |
| Acceptance rate | 20-40% | ~65% |
| Step size | Random walk | Gradient-guided |
| Convergence | Slower | Faster |
| Best for | Simple models | Complex models |
| Computation per step | Low | Higher |

## Trace Analysis

After sampling, analyze the trace to assess convergence and quality:

```javascript
import { summarize, effectiveSampleSize, printSummary } from '@tangent.to/mc';

// Print comprehensive summary
printSummary(trace);

// Get statistics for a variable
const stats = summarize(trace.trace.alpha);
// Returns: { mean, median, std, variance, hdi_2_5, hdi_97_5, n }

// Compute effective sample size
const ess = effectiveSampleSize(trace.trace.alpha);
```

### Convergence Diagnostics

**Visual inspection:**
- Trace plots should look like "fuzzy caterpillar"
- No trends or patterns
- Stable around a mean value

**Effective Sample Size (ESS):**
- Accounts for autocorrelation
- Want ESS > 100 per chain
- If ESS < 100, increase samples or adjust sampler

**R-hat (Gelman-Rubin):**
- Requires multiple chains
- R-hat < 1.1 indicates convergence
- R-hat > 1.1 suggests chains haven't converged

```javascript
import { gelmanRubin } from '@tangent.to/mc';

// Run multiple chains
const trace1 = sampler.sample(model, init1, 1000, 500, 1);
const trace2 = sampler.sample(model, init2, 1000, 500, 1);
const trace3 = sampler.sample(model, init3, 1000, 500, 1);

// Check convergence
const rHat = gelmanRubin([
  trace1.trace.alpha,
  trace2.trace.alpha,
  trace3.trace.alpha
]);

console.log(`R-hat: ${rHat}`); // Should be < 1.1
```

## Best Practices

1. **Run multiple chains** from different initializations
2. **Use adequate burn-in** (500-1000 samples minimum)
3. **Check convergence** with R-hat and trace plots
4. **Monitor acceptance rate** and tune accordingly
5. **Use thinning** if autocorrelation is high
6. **Compute ESS** to assess effective sample size
7. **Start with Metropolis-Hastings** for exploration, then switch to HMC for final inference

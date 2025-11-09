---
layout: default
title: Gaussian Processes
---

# Gaussian Processes

A Gaussian Process (GP) is a distribution over functions. It is a powerful non-parametric method for regression and uncertainty quantification.

## Mathematical Foundation

### Prior

A GP is fully specified by a mean function $m(x)$ and a covariance (kernel) function $k(x, x')$:

$$
f(x) \sim \mathcal{GP}(m(x), k(x, x'))
$$

For any finite set of points $X = \{x_1, \ldots, x_n\}$, the function values follow a multivariate normal:

$$
f(X) \sim \mathcal{N}(m(X), K)
$$

where $K_{ij} = k(x_i, x_j)$

### Posterior

After observing data $(X, y)$ with noise $y = f(X) + \epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma^2I)$, the posterior at new points $X_*$ is:

$$
f_* | X, y, X_* \sim \mathcal{N}(\mu_*, \Sigma_*)
$$

**Posterior mean:**

$$
\mu_* = m(X_*) + K(X_*, X)[K(X, X) + \sigma^2I]^{-1}(y - m(X))
$$

**Posterior covariance:**

$$
\Sigma_* = K(X_*, X_*) - K(X_*, X)[K(X, X) + \sigma^2I]^{-1}K(X, X_*)
$$

### Log Marginal Likelihood

For hyperparameter optimization:

$$
\log p(y|X) = -\frac{1}{2}y^T K_y^{-1} y - \frac{1}{2}\log|K_y| - \frac{n}{2}\log(2\pi)
$$

where $K_y = K(X, X) + \sigma^2I$

## Usage

### Basic Example

```javascript
import { GaussianProcess, RBF } from '@tangent.to/mc';

// Define kernel
const kernel = new RBF(lengthscale=1.0, variance=1.0);

// Create GP
const gp = new GaussianProcess(
  mean=0,
  kernel,
  noiseVariance=0.01
);

// Fit to training data
gp.fit(X_train, y_train);

// Make predictions
const { mean, std } = gp.predict(X_test, returnStd=true);

// Sample from posterior
const posteriorSamples = gp.samplePosterior(X_test, nSamples=5);
```

### Prior Sampling

Sample functions from the GP prior (before seeing data):

```javascript
const X = Array.from({length: 100}, (_, i) => [i / 10]);
const samples = gp.sample(X, nSamples=3);

// samples[0], samples[1], samples[2] are different function realizations
```

## Kernel Functions

The choice of kernel encodes assumptions about the function being modeled.

### RBF (Radial Basis Function)

Also called Squared Exponential kernel. Produces infinitely smooth functions.

$$
k(x, x') = \sigma^2 \exp\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)
$$

```javascript
import { RBF } from '@tangent.to/mc';

const kernel = new RBF(
  lengthscale=1.0,  // Controls smoothness
  variance=1.0      // Controls amplitude
);
```

**Properties:**
- Infinitely differentiable (very smooth)
- Most commonly used kernel
- Good default choice

**Use when:**
- Function is expected to be smooth
- No prior knowledge suggests otherwise

### Matérn 3/2

Produces once-differentiable functions. Less smooth than RBF.

$$
k(x, x') = \sigma^2 \left(1 + \frac{\sqrt{3}r}{\ell}\right) \exp\left(-\frac{\sqrt{3}r}{\ell}\right)
$$

```javascript
import { Matern32 } from '@tangent.to/mc';

const kernel = new Matern32(lengthscale=1.0, variance=1.0);
```

**Properties:**
- Once differentiable
- More flexible than RBF for rough functions

**Use when:**
- Function may have kinks or non-smooth features
- RBF is too restrictive

### Matérn 5/2

Produces twice-differentiable functions. Middle ground between Matérn 3/2 and RBF.

$$
k(x, x') = \sigma^2 \left(1 + \frac{\sqrt{5}r}{\ell} + \frac{5r^2}{3\ell^2}\right) \exp\left(-\frac{\sqrt{5}r}{\ell}\right)
$$

```javascript
import { Matern52 } from '@tangent.to/mc';

const kernel = new Matern52(lengthscale=1.0, variance=1.0);
```

**Properties:**
- Twice differentiable
- Good balance between smoothness and flexibility

### Periodic

For modeling periodic or seasonal patterns.

$$
k(x, x') = \sigma^2 \exp\left(-\frac{2\sin^2(\pi|x - x'|/p)}{\ell^2}\right)
$$

```javascript
import { Periodic } from '@tangent.to/mc';

const kernel = new Periodic(
  period=1.0,        // Repeat period
  lengthscale=1.0,   // Smoothness within period
  variance=1.0
);
```

**Use when:**
- Data has clear periodic structure
- Modeling seasonal effects
- Time series with known periodicity

### Linear

For functions with a global linear trend.

$$
k(x, x') = \sigma^2 (x - c)^T(x' - c)
$$

```javascript
import { Linear } from '@tangent.to/mc';

const kernel = new Linear(variance=1.0, offset=0.0);
```

**Use when:**
- Function has linear behavior
- As a component in kernel combinations

## Complete Example: Sin Function Regression

```javascript
import { GaussianProcess, RBF } from '@tangent.to/mc';

// Generate training data from sin function with noise
const X_train = Array.from({length: 20}, (_, i) => [i * 0.3]);
const y_train = X_train.map(([x]) => Math.sin(x) + 0.1 * (Math.random() - 0.5));

// Create GP with RBF kernel
const kernel = new RBF(lengthscale=1.0, variance=1.0);
const gp = new GaussianProcess(0, kernel, noiseVariance=0.01);

// Fit to data
gp.fit(X_train, y_train);

// Make predictions on test set
const X_test = Array.from({length: 100}, (_, i) => [i * 0.06]);
const { mean, std } = gp.predict(X_test, returnStd=true);

// Log marginal likelihood (for model comparison)
const logML = gp.logMarginalLikelihood();
console.log(`Log marginal likelihood: ${logML}`);

// Posterior predictive samples
const samples = gp.samplePosterior(X_test, nSamples=5);
```

## Hyperparameter Selection

### Manual Tuning

**Lengthscale** ($\ell$):
- Large → smooth, slowly varying functions
- Small → rapidly varying functions
- Typical range: 0.1 - 10

**Variance** ($\sigma^2$):
- Controls the amplitude/scale of the function
- Should match the scale of your data

**Noise variance** ($\sigma_n^2$):
- Observation noise level
- Typical range: 0.001 - 0.1

### Grid Search

```javascript
const lengthscales = [0.1, 0.5, 1.0, 2.0];
const noises = [0.001, 0.01, 0.1];

let bestLogML = -Infinity;
let bestParams = null;

for (const ls of lengthscales) {
  for (const noise of noises) {
    const kernel = new RBF(ls, 1.0);
    const gp = new GaussianProcess(0, kernel, noise);
    gp.fit(X_train, y_train);

    const logML = gp.logMarginalLikelihood();
    if (logML > bestLogML) {
      bestLogML = logML;
      bestParams = { lengthscale: ls, noise };
    }
  }
}

console.log('Best parameters:', bestParams);
```

## Uncertainty Quantification

GPs naturally provide uncertainty estimates:

```javascript
const { mean, std } = gp.predict(X_test, returnStd=true);

// 95% confidence interval
const lower = mean.map((m, i) => m - 1.96 * std[i]);
const upper = mean.map((m, i) => m + 1.96 * std[i]);
```

**Interpreting uncertainty:**
- **Near training data:** Low uncertainty (std is small)
- **Far from training data:** High uncertainty (std is large)
- **More training data:** Reduces uncertainty everywhere

## Use Cases

1. **Regression with uncertainty** - When you need confidence intervals
2. **Small datasets** - GPs work well with limited data
3. **Black-box optimization** - Bayesian optimization uses GPs
4. **Time series forecasting** - With appropriate kernels
5. **Spatial modeling** - Kriging in geostatistics
6. **Calibration** - Model calibration and emulation

## Limitations

1. **Computational cost:** $O(n^3)$ for training, $O(n^2)$ for prediction
   - Becomes slow for $n > 1000$
   - Consider sparse GPs (inducing points) for large datasets

2. **Memory:** Storing kernel matrix requires $O(n^2)$ memory

3. **Kernel choice:** Results sensitive to kernel and hyperparameters

4. **High dimensions:** Kernel methods struggle in very high dimensions (curse of dimensionality)

## Tips

- Start with RBF kernel as default
- Use Matérn kernels if function is not smooth
- Normalize input features to similar scales
- Check log marginal likelihood for model comparison
- Visualize posterior samples to understand uncertainty
- Use multiple random restarts for hyperparameter optimization

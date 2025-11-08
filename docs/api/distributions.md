---
layout: default
title: Distributions
---

# Probability Distributions

All distributions in `@tangent.to/mc` inherit from the base `Distribution` class and support:
- `logProb(value)` - Compute log probability density/mass
- `sample(shape)` - Generate random samples
- `mean()` - Get the distribution mean
- `variance()` - Get the distribution variance

## Continuous Distributions

### Normal Distribution

**PDF:**

$$
p(x | \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

**Usage:**
```javascript
import { Normal } from '@tangent.to/mc';

const dist = new Normal(mu=0, sigma=1, name='x');
const logProb = dist.logProb(0.5);
const samples = dist.sample([100]);
```

**Parameters:**
- `mu` ($\mu$) - Mean parameter
- `sigma` ($\sigma > 0$) - Standard deviation parameter

### Uniform Distribution

**PDF:**

$$
p(x | a, b) = \begin{cases}
\frac{1}{b-a} & \text{if } a \leq x \leq b \\
0 & \text{otherwise}
\end{cases}
$$

**Usage:**
```javascript
import { Uniform } from '@tangent.to/mc';

const dist = new Uniform(lower=0, upper=1, name='x');
```

**Parameters:**
- `lower` ($a$) - Lower bound
- `upper` ($b > a$) - Upper bound

### Beta Distribution

**PDF:**

$$
p(x | \alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}
$$

where $B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$

**Usage:**
```javascript
import { Beta } from '@tangent.to/mc';

const dist = new Beta(alpha=2, beta=5, name='p');
```

**Parameters:**
- `alpha` ($\alpha > 0$) - Shape parameter
- `beta` ($\beta > 0$) - Shape parameter

**Common uses:** Modeling probabilities and proportions

### Gamma Distribution

**PDF:**

$$
p(x | \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}
$$

**Usage:**
```javascript
import { Gamma } from '@tangent.to/mc';

const dist = new Gamma(alpha=2, beta=1, name='x');
```

**Parameters:**
- `alpha` ($\alpha > 0$) - Shape parameter
- `beta` ($\beta > 0$) - Rate parameter

**Common uses:** Modeling waiting times, positive continuous values

## Discrete Distributions

### Bernoulli Distribution

**PMF:**

$$
p(x | p) = \begin{cases}
p & \text{if } x = 1 \\
1-p & \text{if } x = 0
\end{cases}
$$

**Usage:**
```javascript
import { Bernoulli } from '@tangent.to/mc';

const dist = new Bernoulli(p=0.7, name='success');
```

**Parameters:**
- `p` ($0 \leq p \leq 1$) - Success probability

**Common uses:** Binary outcomes, classification

## Gaussian Processes

### GaussianProcess

A distribution over functions, defined by a mean function and covariance (kernel) function.

**Prior:**

$$
f(x) \sim \mathcal{GP}(m(x), k(x, x'))
$$

**Posterior predictive:**

$$
f_* | X, y, X_* \sim \mathcal{N}(\mu_*, \Sigma_*)
$$

where:

$$
\mu_* = m(X_*) + K(X_*, X)[K(X, X) + \sigma^2I]^{-1}(y - m(X))
$$

$$
\Sigma_* = K(X_*, X_*) - K(X_*, X)[K(X, X) + \sigma^2I]^{-1}K(X, X_*)
$$

**Usage:**
```javascript
import { GaussianProcess, RBF } from '@tangent.to/mc';

const kernel = new RBF(lengthscale=1.0, variance=1.0);
const gp = new GaussianProcess(mean=0, kernel, noiseVariance=0.01);

// Fit to data
gp.fit(X_train, y_train);

// Predict
const { mean, std } = gp.predict(X_test, returnStd=true);

// Sample from posterior
const samples = gp.samplePosterior(X_test, nSamples=5);
```

**Parameters:**
- `mean` - Mean function or constant mean value
- `kernel` - Kernel function (e.g., RBF, Matern32)
- `noiseVariance` ($\sigma^2$) - Observation noise variance

See [Gaussian Processes](gaussian-processes.html) for more details on kernels.

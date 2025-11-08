# Deno/Zed REPL Examples

This directory contains examples for using `@tangent.to/mc` in Deno and Zed editor's REPL mode.

## Running Examples

### In Zed Editor

1. Open any `.ts` file in this directory
2. Use Zed's REPL mode (Cmd+Shift+R or Ctrl+Shift+R)
3. The code will execute in Deno runtime

### From Command Line

```bash
# Linear regression
deno run --allow-read --allow-env linear_regression.ts

# Gaussian Process
deno run --allow-read --allow-env gaussian_process.ts

# Hierarchical model
deno run --allow-read --allow-env hierarchical_model.ts
```

## Examples

### 1. Linear Regression (`linear_regression.ts`)

Demonstrates basic Bayesian linear regression:
- Defining priors
- Creating a likelihood function
- MCMC sampling with Metropolis-Hastings
- Posterior analysis

**Mathematical model:**

$$y_i = \alpha + \beta x_i + \epsilon_i$$

where $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$

**Priors:**
- $\alpha \sim \mathcal{N}(0, 10)$
- $\beta \sim \mathcal{N}(0, 10)$
- $\sigma \sim \text{Uniform}(0.01, 5)$

### 2. Gaussian Process (`gaussian_process.ts`)

Shows GP regression with different kernels:
- RBF kernel for smooth functions
- Matérn 3/2 kernel for rough functions
- Posterior predictive uncertainty
- Model comparison via log marginal likelihood
- Posterior sampling

**Key concepts:**
- Non-parametric regression
- Uncertainty quantification
- Kernel selection

### 3. Hierarchical Model (`hierarchical_model.ts`)

Demonstrates hierarchical (multilevel) modeling:
- Global hyperpriors
- Group-level parameters
- Partial pooling across groups
- Borrowing strength from related groups

**Model structure:**

```
mu_global ~ N(0, 20)
sigma_global ~ Uniform(0.1, 10)

mu_a ~ N(mu_global, sigma_global)  # Group A mean
mu_b ~ N(mu_global, sigma_global)  # Group B mean
mu_c ~ N(mu_global, sigma_global)  # Group C mean

y_i ~ N(mu_group[i], sigma)        # Observations
```

**Benefits:**
- Better estimates with limited data
- Automatic regularization
- Quantifies between-group variation

## Requirements

- Deno 1.x or higher
- Internet connection (for downloading npm packages)

## Package Import

All examples import from npm registry:

```typescript
import { Model, Normal, GaussianProcess } from "npm:@tangent.to/mc@0.2.0";
```

After JSR publication, you can also use:

```typescript
import { Model, Normal, GaussianProcess } from "jsr:@tangent-to/mc@0.2.0";
```

## Tips for Zed REPL

1. **Run entire file:** Use Cmd+Shift+R (Mac) or Ctrl+Shift+R (Linux/Windows)
2. **Run selection:** Select code and use the same shortcut
3. **Clear output:** Use the REPL panel controls
4. **Inline results:** Some values appear inline in the editor

## Notes

- First run may be slow due to package downloads
- Subsequent runs use cached packages
- MCMC sampling can take time (especially hierarchical models)
- TensorFlow.js warnings are normal and can be ignored

## Learn More

- [Main README](../../README.md)
- [API Documentation](../../docs/)
- [Platform Guides](../../docs/PLATFORMS.md)

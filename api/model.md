---
layout: default
title: model
parent: API Reference
nav_order: 2
permalink: /api/model
---
# model

## Classes

### Model

Defined in: [model.js:22](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/model.js#L22)

Model class for defining Bayesian probabilistic models

Similar to PyMC's Model context manager, this class represents a probabilistic model
as a Directed Acyclic Graph (DAG) of random variables.

**Joint probability**:
$$
p(\theta, y) = p(y|\theta)p(\theta)
$$
where $\theta$ are parameters (latent variables) and $y$ is observed data.

**Posterior** (via Bayes' theorem):
$$
p(\theta|y) = \frac{p(y|\theta)p(\theta)}{p(y)} \propto p(y|\theta)p(\theta)
$$

#### See

[Documentation](https://www.pymc.io/|PyMC)

#### Constructors

##### Constructor

```ts
new Model(name?): Model;
```

Defined in: [model.js:33](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/model.js#L33)

Accepts either a positional name or a single options object `{ name }`.

###### Parameters

###### name?

`string` \| `Object`

Model name, or an options object `{ name }`

###### Returns

[`Model`](#model)

###### Examples

```ts
new Model('linear_regression')
```

```ts
new Model({ name: 'linear_regression' })
```

#### Properties

##### name

```ts
name: string | Object;
```

Defined in: [model.js:37](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/model.js#L37)

##### variables

```ts
variables: Map<any, any>;
```

Defined in: [model.js:38](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/model.js#L38)

##### observedVars

```ts
observedVars: Map<any, any>;
```

Defined in: [model.js:39](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/model.js#L39)

##### potentials

```ts
potentials: Map<any, any>;
```

Defined in: [model.js:40](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/model.js#L40)

##### deterministics

```ts
deterministics: Map<any, any>;
```

Defined in: [model.js:41](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/model.js#L41)

##### logProbFn

```ts
logProbFn: any;
```

Defined in: [model.js:42](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/model.js#L42)

#### Methods

##### potential()

```ts
potential(name, fn): Model;
```

Defined in: [model.js:64](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/model.js#L64)

Register a generic log-density term (a "potential" / factor) contributing to
the joint log-probability. `fn(params)` receives the current free-variable
values as tf tensors keyed by name and must return a tf.Tensor of
log-density values (which are summed into the total).

This is the general mechanism for likelihoods whose parameters are arbitrary
deterministic functions of the latent variables and data - the deterministic
expression is computed inside `fn`, so it is not specific to any one model:

```js
model.potential('y', (v) =>
  new Normal(tf.add(tf.mul(v.slope, xData), v.intercept), v.sigma).logProb(yData));
```

###### Parameters

###### name

`string`

Identifier for the term

###### fn

(`params`) => `Tensor`

Returns a log-density tensor

###### Returns

[`Model`](#model)

this

##### deterministic()

```ts
deterministic(name, fn): Model;
```

Defined in: [model.js:79](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/model.js#L79)

Register a named deterministic transform of the parameters for recording in
the trace (computed post-hoc from posterior draws). Deterministics do NOT
affect the log-probability - use [Model#potential](#potential) for likelihood or
factor terms.

###### Parameters

###### name

`string`

Identifier for the transform

###### fn

(`params`) => `number` \| `any`[] \| `Tensor`\<`Rank`\>

The transform

###### Returns

[`Model`](#model)

this

##### addVariable()

```ts
addVariable(
   name, 
   distribution, 
   observed?): Distribution;
```

Defined in: [model.js:91](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/model.js#L91)

Add a random variable to the model

###### Parameters

###### name

`string`

Name of the variable

###### distribution

`Distribution`

Distribution of the variable

###### observed?

`any` = `null`

Observed data (optional)

###### Returns

`Distribution`

The distribution

##### getVariable()

```ts
getVariable(name): Distribution;
```

Defined in: [model.js:107](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/model.js#L107)

Get a variable from the model

###### Parameters

###### name

`string`

Name of the variable

###### Returns

`Distribution`

The distribution

##### logProb()

```ts
logProb(params): Tensor<Rank>;
```

Defined in: [model.js:116](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/model.js#L116)

Compute the log probability of the model given parameter values

###### Parameters

###### params

`Object`

Parameter values as {name: value} pairs

###### Returns

`Tensor`\<`Rank`\>

Log probability (scalar)

##### logProbAndGradient()

```ts
logProbAndGradient(params): object;
```

Defined in: [model.js:149](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/model.js#L149)

Compute the log probability and its gradient with respect to parameters

###### Parameters

###### params

`Object`

Parameter values as {name: tf.Tensor} pairs

###### Returns

`object`

The scalar log probability
  and a `{name: tf.Tensor}` map of gradients, one per parameter

###### logProb

```ts
logProb: number;
```

###### gradients

```ts
gradients: Object;
```

##### samplePrior()

```ts
samplePrior(nSamples?): Object;
```

Defined in: [model.js:186](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/model.js#L186)

Sample from the prior distributions

###### Parameters

###### nSamples?

`number` = `1`

Number of samples to generate

###### Returns

`Object`

Samples as {name: Array} pairs

##### getFreeVariableNames()

```ts
getFreeVariableNames(): string[];
```

Defined in: [model.js:204](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/model.js#L204)

Get list of unobserved variable names

###### Returns

`string`[]

Variable names

##### computeDeterministics()

```ts
computeDeterministics(trace): Object;
```

Defined in: [model.js:225](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/model.js#L225)

Evaluate registered [Model#deterministic](#deterministic) transforms on each posterior
draw and append them to the trace as extra columns. Computed post-hoc - they
do not affect sampling - and the MCMC samplers call this automatically before
returning their trace. Each `fn(params)` receives a `{name: number}` map of
the free-variable values for one draw and may return a number, an array, or a
tf.Tensor (tensors are read out and disposed).

###### Parameters

###### trace

`Object`

Trace map `{ name: [...] }` or a `{ trace }` wrapper.

###### Returns

`Object`

The same trace, with one column per deterministic.

##### predictPosterior()

```ts
predictPosterior(
   trace, 
   predictFn, 
   nSamples?): any[];
```

Defined in: [model.js:257](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/model.js#L257)

Posterior predictive sampling
Generate predictions by sampling from the posterior

###### Parameters

###### trace

`Object`

Trace object from MCMC sampling

###### predictFn

`Function`

Function that takes params and returns predictions

###### nSamples?

`number` = `null`

Number of posterior samples to use (null = use all)

###### Returns

`any`[]

Array of predictions from each posterior sample

##### predictPosteriorSummary()

```ts
predictPosteriorSummary(
   trace, 
   predictFn, 
   credibleInterval?): Object;
```

Defined in: [model.js:286](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/model.js#L286)

Compute posterior predictive mean and credible intervals

###### Parameters

###### trace

`Object`

Trace object from MCMC sampling

###### predictFn

`Function`

Function that takes params and returns predictions

###### credibleInterval?

`number` = `0.95`

Credible interval (e.g., 0.95 for 95%)

###### Returns

`Object`

{mean, lower, upper} predictions

##### summary()

```ts
summary(): string;
```

Defined in: [model.js:335](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/model.js#L335)

Create a summary of the model

###### Returns

`string`

Model summary

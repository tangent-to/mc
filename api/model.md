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

Defined in: [model.js:52](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L52)

Bayesian probabilistic model: a DAG of random variables (priors), observed
likelihoods, generic [Model#potential](#potential) log-density terms, and named
[Model#deterministic](#deterministic) transforms, exposing the joint log-probability
and its gradient for the MCMC samplers.

#### Constructors

##### Constructor

```ts
new Model(name?): Model;
```

Defined in: [model.js:63](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L63)

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

Defined in: [model.js:67](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L67)

##### variables

```ts
variables: Map<any, any>;
```

Defined in: [model.js:68](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L68)

##### observedVars

```ts
observedVars: Map<any, any>;
```

Defined in: [model.js:69](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L69)

##### potentials

```ts
potentials: Map<any, any>;
```

Defined in: [model.js:70](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L70)

##### potentialGrads

```ts
potentialGrads: Map<any, any>;
```

Defined in: [model.js:71](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L71)

##### deterministics

```ts
deterministics: Map<any, any>;
```

Defined in: [model.js:72](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L72)

##### compiledTerms

```ts
compiledTerms: Map<any, any>;
```

Defined in: [model.js:75](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L75)

##### observedTerms

```ts
observedTerms: Map<any, any>;
```

Defined in: [model.js:79](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L79)

##### \_transformCache

```ts
_transformCache: Map<any, any> | null | undefined;
```

Defined in: [model.js:271](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L271)

#### Methods

##### potential()

```ts
potential(
   name, 
   fn, 
   gradFn?): Model;
```

Defined in: [model.js:124](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L124)

Register a generic log-density term (a "potential" / factor) contributing to
the joint log-probability. `fn(params)` receives the current free-variable
values as plain numbers (or arrays) keyed by name and must return a number
or an array of log-density values (which are summed into the total).

This is the general mechanism for likelihoods whose parameters are arbitrary
deterministic functions of the latent variables and data - the deterministic
expression is computed inside `fn` with ordinary JavaScript math:

```js
model.potential('y', (v) =>
  new Normal(xData.map((x) => v.slope * x + v.intercept), v.sigma).logProb(yData));
```

Gradients of potentials are estimated by central finite differences by
default; priors added with [Model#addVariable](#addvariable) get analytic gradients.
For a large data term this finite-difference cost (2·(#free params) extra
evaluations of `fn` per gradient) dominates NUTS/HMC — and, more seriously,
finite-difference error costs the leapfrog integrator its symplectic
property, degrading the acceptance rate. Two ways to avoid it:

[Model#autoPotential](#autopotential) writes the term in `@tangent.to/grad` ops and
differentiates it exactly, with no derivation by hand. Prefer it.

Otherwise pass an explicit `gradFn` returning the analytic gradient:

```js
model.potential('y', (v) => new Normal(mu(v), v.sigma).logProb(yData),
  (v) => ({ slope: dSlope(v), intercept: dIntercept(v), sigma: dSigma(v) }));
```

`gradFn(params)` must return an object mapping each free-variable name to the
partial derivative of THIS term's log-density with respect to it (a number,
or an array for a vector-valued variable). It is added to the analytic prior
gradients; omit an entry whose partial is zero.

###### Parameters

###### name

`string`

Identifier for the term

###### fn

(`params`) => `number` \| `number`[]

Returns log-density value(s)

###### gradFn?

(`params`) => `Object`

Optional analytic gradient of `fn`

###### Returns

[`Model`](#model)

this

##### autoPotential()

```ts
autoPotential(
   name, 
   fn, 
   options?): Model;
```

Defined in: [model.js:187](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L187)

Register a potential written in `@tangent.to/grad` ops, differentiated
exactly by reverse-mode autodiff.

The same term as [Model#potential](#potential), but `fn` builds its log-density
from grad's ops instead of plain arithmetic, and returns that expression
rather than a number. No gradient is derived by hand and none is
approximated:

```js
import { add, mul, sub, div, log, square, sum, matmul } from '@tangent.to/grad';

model.autoPotential('y', (v) => {
  const z = div(sub(yData, matmul(X, v.beta)), v.sigma);
  return sub(mul(-0.5, sum(square(z))), mul(yData.length, log(v.sigma)));
});
```

`add` and `mul` take any number of operands, which is what keeps a mean
with several terms readable. JavaScript cannot overload `+`, so this is as
close as the language gets to PyMC's `mu0 + tau * z + gamma`:

```js
const mu = add(v.mu0, mul(tau, matmul(Z, v.z)), matmul(C, v.cyc));
```

Against the finite-difference fallback on a 21-parameter regression with
300 observations: one likelihood evaluation per gradient instead of 2·P,
NUTS 7.7× faster end to end, and the same posterior. The gradient matches
a hand-derived closed form to ~1e-13, where central differences are off by
~2e-7.

The value and gradient share one evaluation, so the sampler's
value-and-gradient path sweeps the data once rather than twice.

The tape is COMPILED by default: built once and replayed at each new set of
parameters, rather than reconstructed per call. That is worth roughly 6x on
a real model, and it is safe here because of the contract above. `fn`
builds an expression out of grad's ops, so its graph is fixed by the way it
is written; and a sampler holds every parameter's shape constant for the
length of a run, so nothing can change underneath the plan.

Stepping outside that contract breaks the assumption, and the two ways to
do it both take deliberate effort: branching on a parameter's numeric value
by reaching into `.data`, so different draws take different paths through
`fn`, or closing over data that is mutated while the sampler runs. Neither
is an expression built from ops, which is why the default is what it is.
Pass `{ compile: false }` if you need one of them.

###### Parameters

###### name

`string`

Identifier for the term

###### fn

(`params`) => `Object`

Builds the log-density as a grad
  expression; receives the free variables as grad `Var`s keyed by name

###### options?

###### compile?

`boolean`

reuse the tape across calls

###### Returns

[`Model`](#model)

this

##### observe()

```ts
observe(
   name, 
   factory, 
   observed, 
   options?): Model;
```

Defined in: [model.js:226](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L226)

Declare an observed random variable: the likelihood, derived from a
distribution instead of written out.

`factory` receives the free variables as grad `Var`s and returns a
distribution whose parameters are expressions in them. The term added to
the model is that distribution's `logDensity` at `observed`, differentiated
exactly and compiled, so this is [Model#autoPotential](#autopotential) with the
density supplied by the distribution rather than by you. What that
removes from a model is everything a PyMC user never writes: the kernel,
the `-n log sigma`, the normalizing constant.

```js
const { add, mul } = mc.ops;
model.addVariable('a', new Normal(0, 5));
model.addVariable('b', new Normal(0, 5));
model.addVariable('sigma', new HalfNormal(2));
model.observe('y', (v) => new Normal(add(v.a, mul(v.b, xData)), v.sigma), yData);
```

The seven built-in distributions can be observed. A user-defined one
cannot be differentiated and is refused here; write its term with
`autoPotential`.

###### Parameters

###### name

`string`

Identifier for the term

###### factory

(`v`) => [`Distribution`](/api/distributions#distribution)

Builds the observation
  distribution from the free variables

###### observed

`number` \| `any`[]

The data

###### options?

`Object` = `{}`

As for [Model#autoPotential](#autopotential)

###### Returns

[`Model`](#model)

this

##### deterministic()

```ts
deterministic(name, fn): Model;
```

Defined in: [model.js:257](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L257)

Register a named deterministic transform of the parameters for recording in
the trace (computed post-hoc from posterior draws). Deterministics do NOT
affect the log-probability - use [Model#potential](#potential) for likelihood or
factor terms.

###### Parameters

###### name

`string`

Identifier for the transform

###### fn

(`params`) => `number` \| `any`[]

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

Defined in: [model.js:269](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L269)

Add a random variable to the model

###### Parameters

###### name

`string`

Name of the variable

###### distribution

[`Distribution`](/api/distributions#distribution)

Distribution of the variable

###### observed?

`any` = `null`

Observed data (optional)

###### Returns

[`Distribution`](/api/distributions#distribution)

The distribution

##### getVariable()

```ts
getVariable(name): Distribution;
```

Defined in: [model.js:286](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L286)

Get a variable from the model

###### Parameters

###### name

`string`

Name of the variable

###### Returns

[`Distribution`](/api/distributions#distribution)

The distribution

##### \_potentialSum()

```ts
_potentialSum(params): number;
```

Defined in: [model.js:291](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L291)

Sum of all potential terms at the given parameter values.

###### Parameters

###### params

`any`

###### Returns

`number`

##### logProb()

```ts
logProb(params): number;
```

Defined in: [model.js:304](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L304)

Compute the log probability of the model given parameter values

###### Parameters

###### params

`Object`

Parameter values as {name: number|Array} pairs

###### Returns

`number`

Log probability (scalar)

##### logProbAndGradient()

```ts
logProbAndGradient(params): object;
```

Defined in: [model.js:338](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L338)

Compute the log probability and its gradient with respect to parameters.

Prior terms are differentiated analytically (proba dlogpdf); potential
terms by central finite differences with step h = 1e-6 * max(1, |x|)
per scalar component.

###### Parameters

###### params

`Object`

Parameter values as {name: number|Array} pairs

###### Returns

`object`

The scalar log probability
  and a `{name: number|Array}` map of gradients, one per parameter

###### logProb

```ts
logProb: number;
```

###### gradients

```ts
gradients: Object;
```

##### gradientsOnly()

```ts
gradientsOnly(params): Object;
```

Defined in: [model.js:377](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L377)

Gradient of the joint log-probability WITHOUT its value — exactly
`logProbAndGradient(params).gradients`, skipping the potential-value pass.

Samplers' leapfrog steps only consume the gradient, but for a model with
an analytic potential gradient, computing the discarded value costs a
full extra pass over the data at every leapfrog step. This method
omits it; the returned gradients are identical.

###### Parameters

###### params

`Object`

Parameter values as {name: number|Array} pairs

###### Returns

`Object`

`{name: number|Array}` map of gradients

##### hasConstrainedVariables()

```ts
hasConstrainedVariables(): boolean;
```

Defined in: [model.js:420](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L420)

Does any free variable have a bounded support?

When nothing is constrained the unconstrained space IS the constrained one
and every transform is the identity, so a sampler can skip the mapping
entirely and behave exactly as it did before this existed.

###### Returns

`boolean`

##### toUnconstrained()

```ts
toUnconstrained(params): Object;
```

Defined in: [model.js:434](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L434)

Map constrained parameter values into the unconstrained space a gradient
sampler should move through.

###### Parameters

###### params

`Object`

`{name: number|Array}` in the model's own units

###### Returns

`Object`

the same shape, unconstrained

##### toConstrained()

```ts
toConstrained(uparams): Object;
```

Defined in: [model.js:451](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L451)

Map unconstrained values back into the model's units.

###### Parameters

###### uparams

`Object`

`{name: number|Array}`, unconstrained

###### Returns

`Object`

the same shape, constrained

##### logProbAndGradientUnconstrained()

```ts
logProbAndGradientUnconstrained(uparams): object;
```

Defined in: [model.js:477](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L477)

Joint log-probability and gradient in UNCONSTRAINED space.

The change of variables adds Σ log|dx/du| to the log-density, which is
what keeps the posterior invariant: without it the sampler would explore
the transformed density, not the one you wrote. The gradient is chained
through the same derivative, plus the d/du of that Jacobian term.

For a lower-bounded parameter x = a + eᵘ the Jacobian term is just u, so
its derivative is 1 — the "+1" below. For a doubly-bounded one it is
1 − 2σ(u).

###### Parameters

###### uparams

`Object`

`{name: number|Array}`, unconstrained

###### Returns

`object`

both in unconstrained terms

###### logProb

```ts
logProb: number;
```

###### gradients

```ts
gradients: Object;
```

##### gradientsOnlyUnconstrained()

```ts
gradientsOnlyUnconstrained(uparams): Object;
```

Defined in: [model.js:509](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L509)

Gradient only, in unconstrained space — the leapfrog hot path.

###### Parameters

###### uparams

`Object`

###### Returns

`Object`

`{name: number|Array}`

##### samplePrior()

```ts
samplePrior(nSamples?): Object;
```

Defined in: [model.js:585](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L585)

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

Defined in: [model.js:601](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L601)

Get list of unobserved variable names

###### Returns

`string`[]

Variable names

##### serializable()

```ts
serializable(): 
  | {
  ok: true;
}
  | {
  ok: false;
  reason: string;
};
```

Defined in: [model.js:634](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L634)

Can this model be written out as data and rebuilt elsewhere?

It can when every variable is one of mc's own distributions, so it
travels as a class name and parameters, and every potential is a compiled
grad term, so it travels as a plan. A `potential` written over plain
numbers is a function and cannot; a user-defined distribution cannot. The
answer names the first thing that stands in the way, since it is what the
user would have to change to get parallel chains.

###### Returns

  \| \{
  `ok`: `true`;
\}
  \| \{
  `ok`: `false`;
  `reason`: `string`;
\}

##### toJSON()

```ts
toJSON(at): Object;
```

Defined in: [model.js:664](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L664)

The model as data. Every variable as its distribution's class name and
parameters; every term as a compiled plan; deterministics left out, since
they run on the trace afterwards and stay on the calling thread.

A plan exists only once its graph has been traced, which needs a point to
trace at; `at` is that point, in the model's own (constrained)
parameterization, the same values a sampler is initialized with.

###### Parameters

###### at

`Object`

a full set of parameter values

###### Returns

`Object`

structured-clonable

##### fromJSON()

```ts
static fromJSON(json): Model;
```

Defined in: [model.js:698](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L698)

Rebuild a model from [Model#toJSON](#tojson)'s output, on any thread.

The terms come back as compiled plans, bound to the shapes they were
traced at. Deterministics do not travel; add them on the calling side
and apply them to the trace when the chains return.

###### Parameters

###### json

`Object`

###### Returns

[`Model`](#model)

##### computeDeterministics()

```ts
computeDeterministics(trace): any;
```

Defined in: [model.js:715](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L715)

###### Parameters

###### trace

`any`

###### Returns

`any`

##### predictPosterior()

```ts
predictPosterior(
   trace, 
   predictFn, 
   nSamples?): any[];
```

Defined in: [model.js:747](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L747)

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

Defined in: [model.js:776](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L776)

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

Defined in: [model.js:825](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/model.js#L825)

Create a summary of the model

###### Returns

`string`

Model summary

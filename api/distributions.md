---
layout: default
title: distributions
parent: API Reference
nav_order: 1
permalink: /api/distributions
---
# distributions

## Classes

### Distribution

Defined in: [distributions/base.js:46](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L46)

Base class for probability distributions.

Subclasses set `this._dist` (a @tangent.to/proba distribution) in their
constructor and implement `_params()` returning the proba parameter
object (fields may be numbers or arrays of numbers).

#### Extended by

- [`Normal`](#normal)
- [`Uniform`](#uniform)
- [`Bernoulli`](#bernoulli)
- [`Beta`](#beta)
- [`Gamma`](#gamma)
- [`Lognormal`](#lognormal)
- [`HalfNormal`](#halfnormal)

#### Constructors

##### Constructor

```ts
new Distribution(name?): Distribution;
```

Defined in: [distributions/base.js:51](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L51)

Create a base distribution; subclasses set `this._dist` and parameters.

###### Parameters

###### name?

`string` = `'Distribution'`

Name of the distribution

###### Returns

[`Distribution`](#distribution)

#### Properties

##### name

```ts
name: string;
```

Defined in: [distributions/base.js:52](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L52)

##### observed

```ts
observed: any;
```

Defined in: [distributions/base.js:53](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L53)

#### Methods

##### \_params()

```ts
_params(): Object;
```

Defined in: [distributions/base.js:60](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L60)

The proba parameter object for this distribution; subclasses must implement.

###### Returns

`Object`

proba parameter object (fields may be numbers or arrays)

##### \_len()

```ts
_len(value): number;
```

Defined in: [distributions/base.js:69](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L69)

Broadcast length across value and parameters (0 = all scalar).

###### Parameters

###### value

`number` \| `any`[]

Value(s) whose length participates in broadcasting

###### Returns

`number`

The broadcast length (0 when every input is scalar)

##### \_paramsAt()

```ts
_paramsAt(i): Object;
```

Defined in: [distributions/base.js:82](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L82)

The proba parameter object with each array parameter indexed at `i`.

###### Parameters

###### i

`number`

Broadcast index

###### Returns

`Object`

Per-element parameter object (scalars passed through)

##### logProb()

```ts
logProb(value): number | number[];
```

Defined in: [distributions/base.js:97](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L97)

Log probability density/mass function. Broadcasts over array values
and/or array parameters.

###### Parameters

###### value

`number` \| `Object` \| `any`[]

Value(s) to evaluate

###### Returns

`number` \| `number`[]

Log probability, elementwise for arrays

##### logDensity()

```ts
logDensity(_value): any;
```

Defined in: [distributions/base.js:135](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L135)

The log-density as a differentiable expression, SUMMED over elements.

Where [Distribution#logProb](#logprob) takes plain numbers and returns the
elementwise density, this takes parameters that may be grad `Var`s, built
from the model's free variables, and returns one scalar `Var`: the total
log-density of `value` under this distribution, differentiable in every
parameter that is a `Var`. It is what `Model#observe` evaluates, so that a
likelihood is derived from the distribution rather than written by hand.

The seven built-in distributions implement it. A subclass that does not is
still a valid prior and a valid `logProb`; it is simply not differentiable,
and `observe` will say so.

###### Parameters

###### \_value

`any`

###### Returns

`any`

scalar

##### logpdf()

```ts
logpdf(value): number | number[];
```

Defined in: [distributions/base.js:150](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L150)

Alias for [Distribution#logProb](#logprob), matching the `@tangent.to/proba`
distribution contract (which names the method `logpdf`). Lets code written
against proba's distributions work unchanged on mc's.

###### Parameters

###### value

`number` \| `Object` \| `any`[]

Value(s) to evaluate

###### Returns

`number` \| `number`[]

##### dlogProbDx()

```ts
dlogProbDx(value): number | number[];
```

Defined in: [distributions/base.js:162](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L162)

Derivative of logProb with respect to the value, elementwise.
Used by Model.logProbAndGradient for analytic prior gradients.
Discrete distributions return 0 (no dx in their gradient contract).

###### Parameters

###### value

`number` \| `any`[]

Value(s) at which to differentiate

###### Returns

`number` \| `number`[]

##### pdf()

```ts
pdf(value): number | number[];
```

Defined in: [distributions/base.js:182](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L182)

Probability density/mass function, `exp(logProb(value))`.

###### Parameters

###### value

`number` \| `any`[]

Value(s) to evaluate

###### Returns

`number` \| `number`[]

##### cdf()

```ts
cdf(value): number;
```

Defined in: [distributions/base.js:192](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L192)

Cumulative distribution function (scalar parameters).

###### Parameters

###### value

`number`

###### Returns

`number`

##### quantile()

```ts
quantile(p): number;
```

Defined in: [distributions/base.js:201](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L201)

Quantile (inverse cdf) function (scalar parameters).

###### Parameters

###### p

`number`

Probability in [0, 1]

###### Returns

`number`

##### sample()

```ts
sample(shape?): number | number[];
```

Defined in: [distributions/base.js:213](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L213)

Sample from the distribution using the package RNG (see setRandomSeed).
`sample()` / `sample([])` return a number; `sample(n)` / `sample([n])`
return an Array of n draws.

###### Parameters

###### shape?

`number` \| `number`[]

Number of samples

###### Returns

`number` \| `number`[]

##### observe()

```ts
observe(data): Distribution;
```

Defined in: [distributions/base.js:224](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L224)

Set observed data for this distribution

###### Parameters

###### data

`number` \| `any`[]

Observed data

###### Returns

[`Distribution`](#distribution)

this, for chaining

##### mean()

```ts
mean(): number | number[];
```

Defined in: [distributions/base.js:233](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L233)

Get the mean of the distribution

###### Returns

`number` \| `number`[]

The mean

##### variance()

```ts
variance(): number | number[];
```

Defined in: [distributions/base.js:243](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L243)

Get the variance of the distribution

###### Returns

`number` \| `number`[]

The variance

##### getParams()

```ts
getParams(): Object;
```

Defined in: [distributions/base.js:254](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L254)

Get the distribution's parameters as a plain object.
Subclasses override to expose their specific parameters.

###### Returns

`Object`

Parameters

***

### Bernoulli

Defined in: [distributions/bernoulli.js:8](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/bernoulli.js#L8)

Bernoulli distribution for binary outcomes.

#### Extends

- [`Distribution`](#distribution)

#### Constructors

##### Constructor

```ts
new Bernoulli(p?, name?): Bernoulli;
```

Defined in: [distributions/bernoulli.js:15](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/bernoulli.js#L15)

Create a Bernoulli distribution.

###### Parameters

###### p?

`number` \| `Object` \| `any`[]

Probability of success in [0, 1], or an
  options object `{ p, name }`

###### name?

`string` = `'Bernoulli'`

Name of the distribution

###### Returns

[`Bernoulli`](#bernoulli)

###### Overrides

[`Distribution`](#distribution).[`constructor`](#constructor)

#### Properties

##### observed

```ts
observed: any;
```

Defined in: [distributions/base.js:53](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L53)

###### Inherited from

[`Distribution`](#distribution).[`observed`](#observed)

##### name

```ts
name: any;
```

Defined in: [distributions/bernoulli.js:19](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/bernoulli.js#L19)

###### Inherited from

[`Distribution`](#distribution).[`name`](#name)

##### p

```ts
p: number | Object | any[];
```

Defined in: [distributions/bernoulli.js:22](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/bernoulli.js#L22)

##### \_dist

```ts
_dist: any;
```

Defined in: [distributions/bernoulli.js:23](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/bernoulli.js#L23)

#### Methods

##### \_len()

```ts
_len(value): number;
```

Defined in: [distributions/base.js:69](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L69)

Broadcast length across value and parameters (0 = all scalar).

###### Parameters

###### value

`number` \| `any`[]

Value(s) whose length participates in broadcasting

###### Returns

`number`

The broadcast length (0 when every input is scalar)

###### Inherited from

[`Distribution`](#distribution).[`_len`](#_len)

##### \_paramsAt()

```ts
_paramsAt(i): Object;
```

Defined in: [distributions/base.js:82](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L82)

The proba parameter object with each array parameter indexed at `i`.

###### Parameters

###### i

`number`

Broadcast index

###### Returns

`Object`

Per-element parameter object (scalars passed through)

###### Inherited from

[`Distribution`](#distribution).[`_paramsAt`](#_paramsat)

##### logProb()

```ts
logProb(value): number | number[];
```

Defined in: [distributions/base.js:97](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L97)

Log probability density/mass function. Broadcasts over array values
and/or array parameters.

###### Parameters

###### value

`number` \| `Object` \| `any`[]

Value(s) to evaluate

###### Returns

`number` \| `number`[]

Log probability, elementwise for arrays

###### Inherited from

[`Distribution`](#distribution).[`logProb`](#logprob)

##### logpdf()

```ts
logpdf(value): number | number[];
```

Defined in: [distributions/base.js:150](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L150)

Alias for [Distribution#logProb](#logprob), matching the `@tangent.to/proba`
distribution contract (which names the method `logpdf`). Lets code written
against proba's distributions work unchanged on mc's.

###### Parameters

###### value

`number` \| `Object` \| `any`[]

Value(s) to evaluate

###### Returns

`number` \| `number`[]

###### Inherited from

[`Distribution`](#distribution).[`logpdf`](#logpdf)

##### dlogProbDx()

```ts
dlogProbDx(value): number | number[];
```

Defined in: [distributions/base.js:162](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L162)

Derivative of logProb with respect to the value, elementwise.
Used by Model.logProbAndGradient for analytic prior gradients.
Discrete distributions return 0 (no dx in their gradient contract).

###### Parameters

###### value

`number` \| `any`[]

Value(s) at which to differentiate

###### Returns

`number` \| `number`[]

###### Inherited from

[`Distribution`](#distribution).[`dlogProbDx`](#dlogprobdx)

##### pdf()

```ts
pdf(value): number | number[];
```

Defined in: [distributions/base.js:182](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L182)

Probability density/mass function, `exp(logProb(value))`.

###### Parameters

###### value

`number` \| `any`[]

Value(s) to evaluate

###### Returns

`number` \| `number`[]

###### Inherited from

[`Distribution`](#distribution).[`pdf`](#pdf)

##### cdf()

```ts
cdf(value): number;
```

Defined in: [distributions/base.js:192](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L192)

Cumulative distribution function (scalar parameters).

###### Parameters

###### value

`number`

###### Returns

`number`

###### Inherited from

[`Distribution`](#distribution).[`cdf`](#cdf)

##### quantile()

```ts
quantile(p): number;
```

Defined in: [distributions/base.js:201](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L201)

Quantile (inverse cdf) function (scalar parameters).

###### Parameters

###### p

`number`

Probability in [0, 1]

###### Returns

`number`

###### Inherited from

[`Distribution`](#distribution).[`quantile`](#quantile)

##### sample()

```ts
sample(shape?): number | number[];
```

Defined in: [distributions/base.js:213](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L213)

Sample from the distribution using the package RNG (see setRandomSeed).
`sample()` / `sample([])` return a number; `sample(n)` / `sample([n])`
return an Array of n draws.

###### Parameters

###### shape?

`number` \| `number`[]

Number of samples

###### Returns

`number` \| `number`[]

###### Inherited from

[`Distribution`](#distribution).[`sample`](#sample)

##### observe()

```ts
observe(data): Distribution;
```

Defined in: [distributions/base.js:224](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L224)

Set observed data for this distribution

###### Parameters

###### data

`number` \| `any`[]

Observed data

###### Returns

[`Distribution`](#distribution)

this, for chaining

###### Inherited from

[`Distribution`](#distribution).[`observe`](#observe)

##### mean()

```ts
mean(): number | number[];
```

Defined in: [distributions/base.js:233](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L233)

Get the mean of the distribution

###### Returns

`number` \| `number`[]

The mean

###### Inherited from

[`Distribution`](#distribution).[`mean`](#mean)

##### variance()

```ts
variance(): number | number[];
```

Defined in: [distributions/base.js:243](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L243)

Get the variance of the distribution

###### Returns

`number` \| `number`[]

The variance

###### Inherited from

[`Distribution`](#distribution).[`variance`](#variance)

##### \_params()

```ts
_params(): object;
```

Defined in: [distributions/bernoulli.js:30](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/bernoulli.js#L30)

The proba parameter object for this distribution.

###### Returns

`object`

###### p

```ts
p: number | any[];
```

###### Overrides

[`Distribution`](#distribution).[`_params`](#_params)

##### logDensity()

```ts
logDensity(value): any;
```

Defined in: [distributions/bernoulli.js:34](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/bernoulli.js#L34)

The log-density as a differentiable expression, SUMMED over elements.

Where [Distribution#logProb](#logprob) takes plain numbers and returns the
elementwise density, this takes parameters that may be grad `Var`s, built
from the model's free variables, and returns one scalar `Var`: the total
log-density of `value` under this distribution, differentiable in every
parameter that is a `Var`. It is what `Model#observe` evaluates, so that a
likelihood is derived from the distribution rather than written by hand.

The seven built-in distributions implement it. A subclass that does not is
still a valid prior and a valid `logProb`; it is simply not differentiable,
and `observe` will say so.

###### Parameters

###### value

`any`

observed value(s), plain numbers

###### Returns

`any`

scalar

###### Overrides

[`Distribution`](#distribution).[`logDensity`](#logdensity)

##### getParams()

```ts
getParams(): object;
```

Defined in: [distributions/bernoulli.js:43](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/bernoulli.js#L43)

Get the distribution's parameters.

###### Returns

`object`

###### p

```ts
p: number | any[];
```

###### Overrides

[`Distribution`](#distribution).[`getParams`](#getparams)

***

### Beta

Defined in: [distributions/beta.js:8](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/beta.js#L8)

Beta distribution on (0, 1).

#### Extends

- [`Distribution`](#distribution)

#### Constructors

##### Constructor

```ts
new Beta(
   alpha?, 
   beta?, 
   name?): Beta;
```

Defined in: [distributions/beta.js:16](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/beta.js#L16)

Create a Beta distribution.

###### Parameters

###### alpha?

`number` \| `Object` \| `any`[]

First shape, or an options object
  `{ alpha, beta, name }`

###### beta?

`number` \| `any`[]

Second shape

###### name?

`string` = `'Beta'`

Name of the distribution

###### Returns

[`Beta`](#beta)

###### Overrides

[`Distribution`](#distribution).[`constructor`](#constructor)

#### Properties

##### observed

```ts
observed: any;
```

Defined in: [distributions/base.js:53](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L53)

###### Inherited from

[`Distribution`](#distribution).[`observed`](#observed)

##### name

```ts
name: any;
```

Defined in: [distributions/beta.js:20](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/beta.js#L20)

###### Inherited from

[`Distribution`](#distribution).[`name`](#name)

##### alpha

```ts
alpha: number | Object | any[];
```

Defined in: [distributions/beta.js:24](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/beta.js#L24)

##### beta

```ts
beta: number | any[] | undefined;
```

Defined in: [distributions/beta.js:25](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/beta.js#L25)

##### \_dist

```ts
_dist: any;
```

Defined in: [distributions/beta.js:26](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/beta.js#L26)

#### Methods

##### \_len()

```ts
_len(value): number;
```

Defined in: [distributions/base.js:69](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L69)

Broadcast length across value and parameters (0 = all scalar).

###### Parameters

###### value

`number` \| `any`[]

Value(s) whose length participates in broadcasting

###### Returns

`number`

The broadcast length (0 when every input is scalar)

###### Inherited from

[`Distribution`](#distribution).[`_len`](#_len)

##### \_paramsAt()

```ts
_paramsAt(i): Object;
```

Defined in: [distributions/base.js:82](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L82)

The proba parameter object with each array parameter indexed at `i`.

###### Parameters

###### i

`number`

Broadcast index

###### Returns

`Object`

Per-element parameter object (scalars passed through)

###### Inherited from

[`Distribution`](#distribution).[`_paramsAt`](#_paramsat)

##### logProb()

```ts
logProb(value): number | number[];
```

Defined in: [distributions/base.js:97](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L97)

Log probability density/mass function. Broadcasts over array values
and/or array parameters.

###### Parameters

###### value

`number` \| `Object` \| `any`[]

Value(s) to evaluate

###### Returns

`number` \| `number`[]

Log probability, elementwise for arrays

###### Inherited from

[`Distribution`](#distribution).[`logProb`](#logprob)

##### logpdf()

```ts
logpdf(value): number | number[];
```

Defined in: [distributions/base.js:150](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L150)

Alias for [Distribution#logProb](#logprob), matching the `@tangent.to/proba`
distribution contract (which names the method `logpdf`). Lets code written
against proba's distributions work unchanged on mc's.

###### Parameters

###### value

`number` \| `Object` \| `any`[]

Value(s) to evaluate

###### Returns

`number` \| `number`[]

###### Inherited from

[`Distribution`](#distribution).[`logpdf`](#logpdf)

##### dlogProbDx()

```ts
dlogProbDx(value): number | number[];
```

Defined in: [distributions/base.js:162](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L162)

Derivative of logProb with respect to the value, elementwise.
Used by Model.logProbAndGradient for analytic prior gradients.
Discrete distributions return 0 (no dx in their gradient contract).

###### Parameters

###### value

`number` \| `any`[]

Value(s) at which to differentiate

###### Returns

`number` \| `number`[]

###### Inherited from

[`Distribution`](#distribution).[`dlogProbDx`](#dlogprobdx)

##### pdf()

```ts
pdf(value): number | number[];
```

Defined in: [distributions/base.js:182](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L182)

Probability density/mass function, `exp(logProb(value))`.

###### Parameters

###### value

`number` \| `any`[]

Value(s) to evaluate

###### Returns

`number` \| `number`[]

###### Inherited from

[`Distribution`](#distribution).[`pdf`](#pdf)

##### cdf()

```ts
cdf(value): number;
```

Defined in: [distributions/base.js:192](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L192)

Cumulative distribution function (scalar parameters).

###### Parameters

###### value

`number`

###### Returns

`number`

###### Inherited from

[`Distribution`](#distribution).[`cdf`](#cdf)

##### quantile()

```ts
quantile(p): number;
```

Defined in: [distributions/base.js:201](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L201)

Quantile (inverse cdf) function (scalar parameters).

###### Parameters

###### p

`number`

Probability in [0, 1]

###### Returns

`number`

###### Inherited from

[`Distribution`](#distribution).[`quantile`](#quantile)

##### sample()

```ts
sample(shape?): number | number[];
```

Defined in: [distributions/base.js:213](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L213)

Sample from the distribution using the package RNG (see setRandomSeed).
`sample()` / `sample([])` return a number; `sample(n)` / `sample([n])`
return an Array of n draws.

###### Parameters

###### shape?

`number` \| `number`[]

Number of samples

###### Returns

`number` \| `number`[]

###### Inherited from

[`Distribution`](#distribution).[`sample`](#sample)

##### observe()

```ts
observe(data): Distribution;
```

Defined in: [distributions/base.js:224](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L224)

Set observed data for this distribution

###### Parameters

###### data

`number` \| `any`[]

Observed data

###### Returns

[`Distribution`](#distribution)

this, for chaining

###### Inherited from

[`Distribution`](#distribution).[`observe`](#observe)

##### mean()

```ts
mean(): number | number[];
```

Defined in: [distributions/base.js:233](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L233)

Get the mean of the distribution

###### Returns

`number` \| `number`[]

The mean

###### Inherited from

[`Distribution`](#distribution).[`mean`](#mean)

##### variance()

```ts
variance(): number | number[];
```

Defined in: [distributions/base.js:243](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L243)

Get the variance of the distribution

###### Returns

`number` \| `number`[]

The variance

###### Inherited from

[`Distribution`](#distribution).[`variance`](#variance)

##### \_params()

```ts
_params(): object;
```

Defined in: [distributions/beta.js:33](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/beta.js#L33)

The proba parameter object for this distribution.

###### Returns

`object`

###### alpha

```ts
alpha: number | any[];
```

###### beta

```ts
beta: number | any[];
```

###### Overrides

[`Distribution`](#distribution).[`_params`](#_params)

##### logDensity()

```ts
logDensity(value): any;
```

Defined in: [distributions/beta.js:37](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/beta.js#L37)

The log-density as a differentiable expression, SUMMED over elements.

Where [Distribution#logProb](#logprob) takes plain numbers and returns the
elementwise density, this takes parameters that may be grad `Var`s, built
from the model's free variables, and returns one scalar `Var`: the total
log-density of `value` under this distribution, differentiable in every
parameter that is a `Var`. It is what `Model#observe` evaluates, so that a
likelihood is derived from the distribution rather than written by hand.

The seven built-in distributions implement it. A subclass that does not is
still a valid prior and a valid `logProb`; it is simply not differentiable,
and `observe` will say so.

###### Parameters

###### value

`any`

observed value(s), plain numbers

###### Returns

`any`

scalar

###### Overrides

[`Distribution`](#distribution).[`logDensity`](#logdensity)

##### getParams()

```ts
getParams(): object;
```

Defined in: [distributions/beta.js:52](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/beta.js#L52)

Get the distribution's parameters.

###### Returns

`object`

###### alpha

```ts
alpha: number | any[];
```

###### beta

```ts
beta: number | any[];
```

###### Overrides

[`Distribution`](#distribution).[`getParams`](#getparams)

***

### Gamma

Defined in: [distributions/gamma.js:9](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/gamma.js#L9)

Gamma distribution (shape/rate parameterization, PyMC convention):
mean = alpha / beta.

#### Extends

- [`Distribution`](#distribution)

#### Constructors

##### Constructor

```ts
new Gamma(
   alpha?, 
   beta?, 
   name?): Gamma;
```

Defined in: [distributions/gamma.js:21](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/gamma.js#L21)

Shape/RATE parameterization (PyMC/Stan convention): mean = alpha / beta.
Note this differs from R and `@tangent.to/ds`, which use shape/SCALE
(scale = 1 / rate). A `scale` key is therefore rejected here rather than
silently misread as a rate — pass `rate` (or `beta`) explicitly.

###### Parameters

###### alpha?

`number` \| `Object` \| `any`[]

Shape, or an options object
  `{ alpha | shape, beta | rate, name }`

###### beta?

`number` \| `any`[]

Rate (NOT scale)

###### name?

`string` = `'Gamma'`

Name of the distribution

###### Returns

[`Gamma`](#gamma)

###### Overrides

[`Distribution`](#distribution).[`constructor`](#constructor)

#### Properties

##### observed

```ts
observed: any;
```

Defined in: [distributions/base.js:53](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L53)

###### Inherited from

[`Distribution`](#distribution).[`observed`](#observed)

##### name

```ts
name: any;
```

Defined in: [distributions/gamma.js:31](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/gamma.js#L31)

###### Inherited from

[`Distribution`](#distribution).[`name`](#name)

##### alpha

```ts
alpha: number | Object | any[];
```

Defined in: [distributions/gamma.js:35](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/gamma.js#L35)

##### beta

```ts
beta: number | any[] | undefined;
```

Defined in: [distributions/gamma.js:36](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/gamma.js#L36)

##### \_dist

```ts
_dist: any;
```

Defined in: [distributions/gamma.js:37](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/gamma.js#L37)

#### Methods

##### \_len()

```ts
_len(value): number;
```

Defined in: [distributions/base.js:69](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L69)

Broadcast length across value and parameters (0 = all scalar).

###### Parameters

###### value

`number` \| `any`[]

Value(s) whose length participates in broadcasting

###### Returns

`number`

The broadcast length (0 when every input is scalar)

###### Inherited from

[`Distribution`](#distribution).[`_len`](#_len)

##### \_paramsAt()

```ts
_paramsAt(i): Object;
```

Defined in: [distributions/base.js:82](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L82)

The proba parameter object with each array parameter indexed at `i`.

###### Parameters

###### i

`number`

Broadcast index

###### Returns

`Object`

Per-element parameter object (scalars passed through)

###### Inherited from

[`Distribution`](#distribution).[`_paramsAt`](#_paramsat)

##### logProb()

```ts
logProb(value): number | number[];
```

Defined in: [distributions/base.js:97](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L97)

Log probability density/mass function. Broadcasts over array values
and/or array parameters.

###### Parameters

###### value

`number` \| `Object` \| `any`[]

Value(s) to evaluate

###### Returns

`number` \| `number`[]

Log probability, elementwise for arrays

###### Inherited from

[`Distribution`](#distribution).[`logProb`](#logprob)

##### logpdf()

```ts
logpdf(value): number | number[];
```

Defined in: [distributions/base.js:150](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L150)

Alias for [Distribution#logProb](#logprob), matching the `@tangent.to/proba`
distribution contract (which names the method `logpdf`). Lets code written
against proba's distributions work unchanged on mc's.

###### Parameters

###### value

`number` \| `Object` \| `any`[]

Value(s) to evaluate

###### Returns

`number` \| `number`[]

###### Inherited from

[`Distribution`](#distribution).[`logpdf`](#logpdf)

##### dlogProbDx()

```ts
dlogProbDx(value): number | number[];
```

Defined in: [distributions/base.js:162](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L162)

Derivative of logProb with respect to the value, elementwise.
Used by Model.logProbAndGradient for analytic prior gradients.
Discrete distributions return 0 (no dx in their gradient contract).

###### Parameters

###### value

`number` \| `any`[]

Value(s) at which to differentiate

###### Returns

`number` \| `number`[]

###### Inherited from

[`Distribution`](#distribution).[`dlogProbDx`](#dlogprobdx)

##### pdf()

```ts
pdf(value): number | number[];
```

Defined in: [distributions/base.js:182](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L182)

Probability density/mass function, `exp(logProb(value))`.

###### Parameters

###### value

`number` \| `any`[]

Value(s) to evaluate

###### Returns

`number` \| `number`[]

###### Inherited from

[`Distribution`](#distribution).[`pdf`](#pdf)

##### cdf()

```ts
cdf(value): number;
```

Defined in: [distributions/base.js:192](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L192)

Cumulative distribution function (scalar parameters).

###### Parameters

###### value

`number`

###### Returns

`number`

###### Inherited from

[`Distribution`](#distribution).[`cdf`](#cdf)

##### quantile()

```ts
quantile(p): number;
```

Defined in: [distributions/base.js:201](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L201)

Quantile (inverse cdf) function (scalar parameters).

###### Parameters

###### p

`number`

Probability in [0, 1]

###### Returns

`number`

###### Inherited from

[`Distribution`](#distribution).[`quantile`](#quantile)

##### sample()

```ts
sample(shape?): number | number[];
```

Defined in: [distributions/base.js:213](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L213)

Sample from the distribution using the package RNG (see setRandomSeed).
`sample()` / `sample([])` return a number; `sample(n)` / `sample([n])`
return an Array of n draws.

###### Parameters

###### shape?

`number` \| `number`[]

Number of samples

###### Returns

`number` \| `number`[]

###### Inherited from

[`Distribution`](#distribution).[`sample`](#sample)

##### observe()

```ts
observe(data): Distribution;
```

Defined in: [distributions/base.js:224](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L224)

Set observed data for this distribution

###### Parameters

###### data

`number` \| `any`[]

Observed data

###### Returns

[`Distribution`](#distribution)

this, for chaining

###### Inherited from

[`Distribution`](#distribution).[`observe`](#observe)

##### mean()

```ts
mean(): number | number[];
```

Defined in: [distributions/base.js:233](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L233)

Get the mean of the distribution

###### Returns

`number` \| `number`[]

The mean

###### Inherited from

[`Distribution`](#distribution).[`mean`](#mean)

##### variance()

```ts
variance(): number | number[];
```

Defined in: [distributions/base.js:243](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L243)

Get the variance of the distribution

###### Returns

`number` \| `number`[]

The variance

###### Inherited from

[`Distribution`](#distribution).[`variance`](#variance)

##### \_params()

```ts
_params(): object;
```

Defined in: [distributions/gamma.js:44](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/gamma.js#L44)

The proba parameter object for this distribution (shape/rate).

###### Returns

`object`

###### alpha

```ts
alpha: number | any[];
```

###### beta

```ts
beta: number | any[];
```

###### Overrides

[`Distribution`](#distribution).[`_params`](#_params)

##### logDensity()

```ts
logDensity(value): any;
```

Defined in: [distributions/gamma.js:48](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/gamma.js#L48)

The log-density as a differentiable expression, SUMMED over elements.

Where [Distribution#logProb](#logprob) takes plain numbers and returns the
elementwise density, this takes parameters that may be grad `Var`s, built
from the model's free variables, and returns one scalar `Var`: the total
log-density of `value` under this distribution, differentiable in every
parameter that is a `Var`. It is what `Model#observe` evaluates, so that a
likelihood is derived from the distribution rather than written by hand.

The seven built-in distributions implement it. A subclass that does not is
still a valid prior and a valid `logProb`; it is simply not differentiable,
and `observe` will say so.

###### Parameters

###### value

`any`

observed value(s), plain numbers

###### Returns

`any`

scalar

###### Overrides

[`Distribution`](#distribution).[`logDensity`](#logdensity)

##### getParams()

```ts
getParams(): object;
```

Defined in: [distributions/gamma.js:64](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/gamma.js#L64)

Get the distribution's parameters.

###### Returns

`object`

###### alpha

```ts
alpha: number | any[];
```

###### beta

```ts
beta: number | any[];
```

###### Overrides

[`Distribution`](#distribution).[`getParams`](#getparams)

***

### HalfNormal

Defined in: [distributions/halfnormal.js:10](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/halfnormal.js#L10)

Half-normal distribution on [0, Infinity) — the absolute value of a
Normal(0, sigma^2). A standard weakly-informative prior for scales.

#### Extends

- [`Distribution`](#distribution)

#### Constructors

##### Constructor

```ts
new HalfNormal(sigma?, name?): HalfNormal;
```

Defined in: [distributions/halfnormal.js:17](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/halfnormal.js#L17)

Create a half-normal distribution.

###### Parameters

###### sigma?

`number` \| `Object` \| `any`[]

Scale, or an options object
  `{ sigma | sd | std | scale, name }`

###### name?

`string` = `'HalfNormal'`

Name of the distribution

###### Returns

[`HalfNormal`](#halfnormal)

###### Overrides

[`Distribution`](#distribution).[`constructor`](#constructor)

#### Properties

##### observed

```ts
observed: any;
```

Defined in: [distributions/base.js:53](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L53)

###### Inherited from

[`Distribution`](#distribution).[`observed`](#observed)

##### name

```ts
name: any;
```

Defined in: [distributions/halfnormal.js:21](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/halfnormal.js#L21)

###### Inherited from

[`Distribution`](#distribution).[`name`](#name)

##### sigma

```ts
sigma: number | Object | any[];
```

Defined in: [distributions/halfnormal.js:24](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/halfnormal.js#L24)

##### \_dist

```ts
_dist: any;
```

Defined in: [distributions/halfnormal.js:25](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/halfnormal.js#L25)

#### Methods

##### \_len()

```ts
_len(value): number;
```

Defined in: [distributions/base.js:69](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L69)

Broadcast length across value and parameters (0 = all scalar).

###### Parameters

###### value

`number` \| `any`[]

Value(s) whose length participates in broadcasting

###### Returns

`number`

The broadcast length (0 when every input is scalar)

###### Inherited from

[`Distribution`](#distribution).[`_len`](#_len)

##### \_paramsAt()

```ts
_paramsAt(i): Object;
```

Defined in: [distributions/base.js:82](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L82)

The proba parameter object with each array parameter indexed at `i`.

###### Parameters

###### i

`number`

Broadcast index

###### Returns

`Object`

Per-element parameter object (scalars passed through)

###### Inherited from

[`Distribution`](#distribution).[`_paramsAt`](#_paramsat)

##### logProb()

```ts
logProb(value): number | number[];
```

Defined in: [distributions/base.js:97](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L97)

Log probability density/mass function. Broadcasts over array values
and/or array parameters.

###### Parameters

###### value

`number` \| `Object` \| `any`[]

Value(s) to evaluate

###### Returns

`number` \| `number`[]

Log probability, elementwise for arrays

###### Inherited from

[`Distribution`](#distribution).[`logProb`](#logprob)

##### logpdf()

```ts
logpdf(value): number | number[];
```

Defined in: [distributions/base.js:150](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L150)

Alias for [Distribution#logProb](#logprob), matching the `@tangent.to/proba`
distribution contract (which names the method `logpdf`). Lets code written
against proba's distributions work unchanged on mc's.

###### Parameters

###### value

`number` \| `Object` \| `any`[]

Value(s) to evaluate

###### Returns

`number` \| `number`[]

###### Inherited from

[`Distribution`](#distribution).[`logpdf`](#logpdf)

##### dlogProbDx()

```ts
dlogProbDx(value): number | number[];
```

Defined in: [distributions/base.js:162](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L162)

Derivative of logProb with respect to the value, elementwise.
Used by Model.logProbAndGradient for analytic prior gradients.
Discrete distributions return 0 (no dx in their gradient contract).

###### Parameters

###### value

`number` \| `any`[]

Value(s) at which to differentiate

###### Returns

`number` \| `number`[]

###### Inherited from

[`Distribution`](#distribution).[`dlogProbDx`](#dlogprobdx)

##### pdf()

```ts
pdf(value): number | number[];
```

Defined in: [distributions/base.js:182](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L182)

Probability density/mass function, `exp(logProb(value))`.

###### Parameters

###### value

`number` \| `any`[]

Value(s) to evaluate

###### Returns

`number` \| `number`[]

###### Inherited from

[`Distribution`](#distribution).[`pdf`](#pdf)

##### cdf()

```ts
cdf(value): number;
```

Defined in: [distributions/base.js:192](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L192)

Cumulative distribution function (scalar parameters).

###### Parameters

###### value

`number`

###### Returns

`number`

###### Inherited from

[`Distribution`](#distribution).[`cdf`](#cdf)

##### quantile()

```ts
quantile(p): number;
```

Defined in: [distributions/base.js:201](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L201)

Quantile (inverse cdf) function (scalar parameters).

###### Parameters

###### p

`number`

Probability in [0, 1]

###### Returns

`number`

###### Inherited from

[`Distribution`](#distribution).[`quantile`](#quantile)

##### sample()

```ts
sample(shape?): number | number[];
```

Defined in: [distributions/base.js:213](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L213)

Sample from the distribution using the package RNG (see setRandomSeed).
`sample()` / `sample([])` return a number; `sample(n)` / `sample([n])`
return an Array of n draws.

###### Parameters

###### shape?

`number` \| `number`[]

Number of samples

###### Returns

`number` \| `number`[]

###### Inherited from

[`Distribution`](#distribution).[`sample`](#sample)

##### observe()

```ts
observe(data): Distribution;
```

Defined in: [distributions/base.js:224](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L224)

Set observed data for this distribution

###### Parameters

###### data

`number` \| `any`[]

Observed data

###### Returns

[`Distribution`](#distribution)

this, for chaining

###### Inherited from

[`Distribution`](#distribution).[`observe`](#observe)

##### mean()

```ts
mean(): number | number[];
```

Defined in: [distributions/base.js:233](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L233)

Get the mean of the distribution

###### Returns

`number` \| `number`[]

The mean

###### Inherited from

[`Distribution`](#distribution).[`mean`](#mean)

##### variance()

```ts
variance(): number | number[];
```

Defined in: [distributions/base.js:243](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L243)

Get the variance of the distribution

###### Returns

`number` \| `number`[]

The variance

###### Inherited from

[`Distribution`](#distribution).[`variance`](#variance)

##### \_params()

```ts
_params(): object;
```

Defined in: [distributions/halfnormal.js:32](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/halfnormal.js#L32)

The proba parameter object for this distribution.

###### Returns

`object`

###### sigma

```ts
sigma: number | any[];
```

###### Overrides

[`Distribution`](#distribution).[`_params`](#_params)

##### logDensity()

```ts
logDensity(value): any;
```

Defined in: [distributions/halfnormal.js:36](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/halfnormal.js#L36)

The log-density as a differentiable expression, SUMMED over elements.

Where [Distribution#logProb](#logprob) takes plain numbers and returns the
elementwise density, this takes parameters that may be grad `Var`s, built
from the model's free variables, and returns one scalar `Var`: the total
log-density of `value` under this distribution, differentiable in every
parameter that is a `Var`. It is what `Model#observe` evaluates, so that a
likelihood is derived from the distribution rather than written by hand.

The seven built-in distributions implement it. A subclass that does not is
still a valid prior and a valid `logProb`; it is simply not differentiable,
and `observe` will say so.

###### Parameters

###### value

`any`

observed value(s), plain numbers

###### Returns

`any`

scalar

###### Overrides

[`Distribution`](#distribution).[`logDensity`](#logdensity)

##### getParams()

```ts
getParams(): object;
```

Defined in: [distributions/halfnormal.js:47](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/halfnormal.js#L47)

Get the distribution's parameters.

###### Returns

`object`

###### sigma

```ts
sigma: number | any[];
```

###### Overrides

[`Distribution`](#distribution).[`getParams`](#getparams)

***

### Lognormal

Defined in: [distributions/lognormal.js:10](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/lognormal.js#L10)

Log-normal distribution: if log X ~ Normal(mu, sigma^2) then
X ~ LogNormal(mu, sigma). Parameters are on the log scale.

#### Extends

- [`Distribution`](#distribution)

#### Constructors

##### Constructor

```ts
new Lognormal(
   mu?, 
   sigma?, 
   name?): Lognormal;
```

Defined in: [distributions/lognormal.js:18](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/lognormal.js#L18)

Create a log-normal distribution (parameters on the log scale).

###### Parameters

###### mu?

`number` \| `Object` \| `any`[]

Log-scale location, or an options object
  `{ mu | mean, sigma | sd | std, name }`

###### sigma?

`number` \| `any`[]

Log-scale standard deviation

###### name?

`string` = `'Lognormal'`

Name of the distribution

###### Returns

[`Lognormal`](#lognormal)

###### Overrides

[`Distribution`](#distribution).[`constructor`](#constructor)

#### Properties

##### observed

```ts
observed: any;
```

Defined in: [distributions/base.js:53](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L53)

###### Inherited from

[`Distribution`](#distribution).[`observed`](#observed)

##### name

```ts
name: any;
```

Defined in: [distributions/lognormal.js:22](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/lognormal.js#L22)

###### Inherited from

[`Distribution`](#distribution).[`name`](#name)

##### mu

```ts
mu: number | Object | any[];
```

Defined in: [distributions/lognormal.js:26](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/lognormal.js#L26)

##### sigma

```ts
sigma: number | any[] | undefined;
```

Defined in: [distributions/lognormal.js:27](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/lognormal.js#L27)

##### \_dist

```ts
_dist: any;
```

Defined in: [distributions/lognormal.js:28](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/lognormal.js#L28)

#### Methods

##### \_len()

```ts
_len(value): number;
```

Defined in: [distributions/base.js:69](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L69)

Broadcast length across value and parameters (0 = all scalar).

###### Parameters

###### value

`number` \| `any`[]

Value(s) whose length participates in broadcasting

###### Returns

`number`

The broadcast length (0 when every input is scalar)

###### Inherited from

[`Distribution`](#distribution).[`_len`](#_len)

##### \_paramsAt()

```ts
_paramsAt(i): Object;
```

Defined in: [distributions/base.js:82](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L82)

The proba parameter object with each array parameter indexed at `i`.

###### Parameters

###### i

`number`

Broadcast index

###### Returns

`Object`

Per-element parameter object (scalars passed through)

###### Inherited from

[`Distribution`](#distribution).[`_paramsAt`](#_paramsat)

##### logProb()

```ts
logProb(value): number | number[];
```

Defined in: [distributions/base.js:97](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L97)

Log probability density/mass function. Broadcasts over array values
and/or array parameters.

###### Parameters

###### value

`number` \| `Object` \| `any`[]

Value(s) to evaluate

###### Returns

`number` \| `number`[]

Log probability, elementwise for arrays

###### Inherited from

[`Distribution`](#distribution).[`logProb`](#logprob)

##### logpdf()

```ts
logpdf(value): number | number[];
```

Defined in: [distributions/base.js:150](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L150)

Alias for [Distribution#logProb](#logprob), matching the `@tangent.to/proba`
distribution contract (which names the method `logpdf`). Lets code written
against proba's distributions work unchanged on mc's.

###### Parameters

###### value

`number` \| `Object` \| `any`[]

Value(s) to evaluate

###### Returns

`number` \| `number`[]

###### Inherited from

[`Distribution`](#distribution).[`logpdf`](#logpdf)

##### dlogProbDx()

```ts
dlogProbDx(value): number | number[];
```

Defined in: [distributions/base.js:162](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L162)

Derivative of logProb with respect to the value, elementwise.
Used by Model.logProbAndGradient for analytic prior gradients.
Discrete distributions return 0 (no dx in their gradient contract).

###### Parameters

###### value

`number` \| `any`[]

Value(s) at which to differentiate

###### Returns

`number` \| `number`[]

###### Inherited from

[`Distribution`](#distribution).[`dlogProbDx`](#dlogprobdx)

##### pdf()

```ts
pdf(value): number | number[];
```

Defined in: [distributions/base.js:182](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L182)

Probability density/mass function, `exp(logProb(value))`.

###### Parameters

###### value

`number` \| `any`[]

Value(s) to evaluate

###### Returns

`number` \| `number`[]

###### Inherited from

[`Distribution`](#distribution).[`pdf`](#pdf)

##### cdf()

```ts
cdf(value): number;
```

Defined in: [distributions/base.js:192](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L192)

Cumulative distribution function (scalar parameters).

###### Parameters

###### value

`number`

###### Returns

`number`

###### Inherited from

[`Distribution`](#distribution).[`cdf`](#cdf)

##### quantile()

```ts
quantile(p): number;
```

Defined in: [distributions/base.js:201](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L201)

Quantile (inverse cdf) function (scalar parameters).

###### Parameters

###### p

`number`

Probability in [0, 1]

###### Returns

`number`

###### Inherited from

[`Distribution`](#distribution).[`quantile`](#quantile)

##### sample()

```ts
sample(shape?): number | number[];
```

Defined in: [distributions/base.js:213](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L213)

Sample from the distribution using the package RNG (see setRandomSeed).
`sample()` / `sample([])` return a number; `sample(n)` / `sample([n])`
return an Array of n draws.

###### Parameters

###### shape?

`number` \| `number`[]

Number of samples

###### Returns

`number` \| `number`[]

###### Inherited from

[`Distribution`](#distribution).[`sample`](#sample)

##### observe()

```ts
observe(data): Distribution;
```

Defined in: [distributions/base.js:224](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L224)

Set observed data for this distribution

###### Parameters

###### data

`number` \| `any`[]

Observed data

###### Returns

[`Distribution`](#distribution)

this, for chaining

###### Inherited from

[`Distribution`](#distribution).[`observe`](#observe)

##### mean()

```ts
mean(): number | number[];
```

Defined in: [distributions/base.js:233](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L233)

Get the mean of the distribution

###### Returns

`number` \| `number`[]

The mean

###### Inherited from

[`Distribution`](#distribution).[`mean`](#mean)

##### variance()

```ts
variance(): number | number[];
```

Defined in: [distributions/base.js:243](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L243)

Get the variance of the distribution

###### Returns

`number` \| `number`[]

The variance

###### Inherited from

[`Distribution`](#distribution).[`variance`](#variance)

##### \_params()

```ts
_params(): object;
```

Defined in: [distributions/lognormal.js:35](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/lognormal.js#L35)

The proba parameter object for this distribution.

###### Returns

`object`

###### mu

```ts
mu: number | any[];
```

###### sigma

```ts
sigma: number | any[];
```

###### Overrides

[`Distribution`](#distribution).[`_params`](#_params)

##### logDensity()

```ts
logDensity(value): any;
```

Defined in: [distributions/lognormal.js:39](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/lognormal.js#L39)

The log-density as a differentiable expression, SUMMED over elements.

Where [Distribution#logProb](#logprob) takes plain numbers and returns the
elementwise density, this takes parameters that may be grad `Var`s, built
from the model's free variables, and returns one scalar `Var`: the total
log-density of `value` under this distribution, differentiable in every
parameter that is a `Var`. It is what `Model#observe` evaluates, so that a
likelihood is derived from the distribution rather than written by hand.

The seven built-in distributions implement it. A subclass that does not is
still a valid prior and a valid `logProb`; it is simply not differentiable,
and `observe` will say so.

###### Parameters

###### value

`any`

observed value(s), plain numbers

###### Returns

`any`

scalar

###### Overrides

[`Distribution`](#distribution).[`logDensity`](#logdensity)

##### getParams()

```ts
getParams(): object;
```

Defined in: [distributions/lognormal.js:50](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/lognormal.js#L50)

Get the distribution's parameters.

###### Returns

`object`

###### mu

```ts
mu: number | any[];
```

###### sigma

```ts
sigma: number | any[];
```

###### Overrides

[`Distribution`](#distribution).[`getParams`](#getparams)

***

### Normal

Defined in: [distributions/normal.js:13](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/normal.js#L13)

Normal (Gaussian) distribution

$$ p(x | \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) $$

#### See

[Distribution](https://en.wikipedia.org/wiki/Normal_distribution|Normal)

#### Extends

- [`Distribution`](#distribution)

#### Constructors

##### Constructor

```ts
new Normal(
   mu?, 
   sigma?, 
   name?): Normal;
```

Defined in: [distributions/normal.js:28](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/normal.js#L28)

Accepts either positional arguments or a single options object, matching the
dual-constructor convention of `@tangent.to/ds`.

###### Parameters

###### mu?

`number` \| `Object` \| `any`[]

Mean parameter, or an options object
  `{ mu | mean, sigma | sd | std, name }`

###### sigma?

`number` \| `any`[]

Standard deviation, sigma > 0

###### name?

`string` = `'Normal'`

Name of the distribution

###### Returns

[`Normal`](#normal)

###### Examples

```ts
new Normal(0, 1)
```

```ts
new Normal({ mean: 0, sd: 1 })
```

###### Overrides

[`Distribution`](#distribution).[`constructor`](#constructor)

#### Properties

##### observed

```ts
observed: any;
```

Defined in: [distributions/base.js:53](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L53)

###### Inherited from

[`Distribution`](#distribution).[`observed`](#observed)

##### name

```ts
name: any;
```

Defined in: [distributions/normal.js:32](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/normal.js#L32)

###### Inherited from

[`Distribution`](#distribution).[`name`](#name)

##### mu

```ts
mu: number | Object | any[];
```

Defined in: [distributions/normal.js:36](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/normal.js#L36)

##### sigma

```ts
sigma: number | any[] | undefined;
```

Defined in: [distributions/normal.js:37](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/normal.js#L37)

##### \_dist

```ts
_dist: any;
```

Defined in: [distributions/normal.js:38](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/normal.js#L38)

#### Methods

##### \_len()

```ts
_len(value): number;
```

Defined in: [distributions/base.js:69](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L69)

Broadcast length across value and parameters (0 = all scalar).

###### Parameters

###### value

`number` \| `any`[]

Value(s) whose length participates in broadcasting

###### Returns

`number`

The broadcast length (0 when every input is scalar)

###### Inherited from

[`Distribution`](#distribution).[`_len`](#_len)

##### \_paramsAt()

```ts
_paramsAt(i): Object;
```

Defined in: [distributions/base.js:82](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L82)

The proba parameter object with each array parameter indexed at `i`.

###### Parameters

###### i

`number`

Broadcast index

###### Returns

`Object`

Per-element parameter object (scalars passed through)

###### Inherited from

[`Distribution`](#distribution).[`_paramsAt`](#_paramsat)

##### logProb()

```ts
logProb(value): number | number[];
```

Defined in: [distributions/base.js:97](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L97)

Log probability density/mass function. Broadcasts over array values
and/or array parameters.

###### Parameters

###### value

`number` \| `Object` \| `any`[]

Value(s) to evaluate

###### Returns

`number` \| `number`[]

Log probability, elementwise for arrays

###### Inherited from

[`Distribution`](#distribution).[`logProb`](#logprob)

##### logpdf()

```ts
logpdf(value): number | number[];
```

Defined in: [distributions/base.js:150](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L150)

Alias for [Distribution#logProb](#logprob), matching the `@tangent.to/proba`
distribution contract (which names the method `logpdf`). Lets code written
against proba's distributions work unchanged on mc's.

###### Parameters

###### value

`number` \| `Object` \| `any`[]

Value(s) to evaluate

###### Returns

`number` \| `number`[]

###### Inherited from

[`Distribution`](#distribution).[`logpdf`](#logpdf)

##### dlogProbDx()

```ts
dlogProbDx(value): number | number[];
```

Defined in: [distributions/base.js:162](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L162)

Derivative of logProb with respect to the value, elementwise.
Used by Model.logProbAndGradient for analytic prior gradients.
Discrete distributions return 0 (no dx in their gradient contract).

###### Parameters

###### value

`number` \| `any`[]

Value(s) at which to differentiate

###### Returns

`number` \| `number`[]

###### Inherited from

[`Distribution`](#distribution).[`dlogProbDx`](#dlogprobdx)

##### pdf()

```ts
pdf(value): number | number[];
```

Defined in: [distributions/base.js:182](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L182)

Probability density/mass function, `exp(logProb(value))`.

###### Parameters

###### value

`number` \| `any`[]

Value(s) to evaluate

###### Returns

`number` \| `number`[]

###### Inherited from

[`Distribution`](#distribution).[`pdf`](#pdf)

##### cdf()

```ts
cdf(value): number;
```

Defined in: [distributions/base.js:192](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L192)

Cumulative distribution function (scalar parameters).

###### Parameters

###### value

`number`

###### Returns

`number`

###### Inherited from

[`Distribution`](#distribution).[`cdf`](#cdf)

##### quantile()

```ts
quantile(p): number;
```

Defined in: [distributions/base.js:201](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L201)

Quantile (inverse cdf) function (scalar parameters).

###### Parameters

###### p

`number`

Probability in [0, 1]

###### Returns

`number`

###### Inherited from

[`Distribution`](#distribution).[`quantile`](#quantile)

##### sample()

```ts
sample(shape?): number | number[];
```

Defined in: [distributions/base.js:213](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L213)

Sample from the distribution using the package RNG (see setRandomSeed).
`sample()` / `sample([])` return a number; `sample(n)` / `sample([n])`
return an Array of n draws.

###### Parameters

###### shape?

`number` \| `number`[]

Number of samples

###### Returns

`number` \| `number`[]

###### Inherited from

[`Distribution`](#distribution).[`sample`](#sample)

##### observe()

```ts
observe(data): Distribution;
```

Defined in: [distributions/base.js:224](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L224)

Set observed data for this distribution

###### Parameters

###### data

`number` \| `any`[]

Observed data

###### Returns

[`Distribution`](#distribution)

this, for chaining

###### Inherited from

[`Distribution`](#distribution).[`observe`](#observe)

##### mean()

```ts
mean(): number | number[];
```

Defined in: [distributions/base.js:233](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L233)

Get the mean of the distribution

###### Returns

`number` \| `number`[]

The mean

###### Inherited from

[`Distribution`](#distribution).[`mean`](#mean)

##### variance()

```ts
variance(): number | number[];
```

Defined in: [distributions/base.js:243](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L243)

Get the variance of the distribution

###### Returns

`number` \| `number`[]

The variance

###### Inherited from

[`Distribution`](#distribution).[`variance`](#variance)

##### \_params()

```ts
_params(): object;
```

Defined in: [distributions/normal.js:45](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/normal.js#L45)

The proba parameter object for this distribution.

###### Returns

`object`

###### mu

```ts
mu: number | any[];
```

###### sigma

```ts
sigma: number | any[];
```

###### Overrides

[`Distribution`](#distribution).[`_params`](#_params)

##### logDensity()

```ts
logDensity(value): any;
```

Defined in: [distributions/normal.js:49](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/normal.js#L49)

The log-density as a differentiable expression, SUMMED over elements.

Where [Distribution#logProb](#logprob) takes plain numbers and returns the
elementwise density, this takes parameters that may be grad `Var`s, built
from the model's free variables, and returns one scalar `Var`: the total
log-density of `value` under this distribution, differentiable in every
parameter that is a `Var`. It is what `Model#observe` evaluates, so that a
likelihood is derived from the distribution rather than written by hand.

The seven built-in distributions implement it. A subclass that does not is
still a valid prior and a valid `logProb`; it is simply not differentiable,
and `observe` will say so.

###### Parameters

###### value

`any`

observed value(s), plain numbers

###### Returns

`any`

scalar

###### Overrides

[`Distribution`](#distribution).[`logDensity`](#logdensity)

##### getParams()

```ts
getParams(): object;
```

Defined in: [distributions/normal.js:61](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/normal.js#L61)

Get the distribution's parameters.

###### Returns

`object`

###### mu

```ts
mu: number | any[];
```

###### sigma

```ts
sigma: number | any[];
```

###### Overrides

[`Distribution`](#distribution).[`getParams`](#getparams)

***

### Uniform

Defined in: [distributions/uniform.js:8](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/uniform.js#L8)

Continuous uniform distribution on [lower, upper].

#### Extends

- [`Distribution`](#distribution)

#### Constructors

##### Constructor

```ts
new Uniform(
   lower?, 
   upper?, 
   name?): Uniform;
```

Defined in: [distributions/uniform.js:16](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/uniform.js#L16)

Create a continuous uniform distribution on [lower, upper].

###### Parameters

###### lower?

`number` \| `Object` \| `any`[]

Lower bound, or an options object
  `{ lower | min, upper | max, name }`

###### upper?

`number` \| `any`[]

Upper bound

###### name?

`string` = `'Uniform'`

Name of the distribution

###### Returns

[`Uniform`](#uniform)

###### Overrides

[`Distribution`](#distribution).[`constructor`](#constructor)

#### Properties

##### observed

```ts
observed: any;
```

Defined in: [distributions/base.js:53](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L53)

###### Inherited from

[`Distribution`](#distribution).[`observed`](#observed)

##### name

```ts
name: any;
```

Defined in: [distributions/uniform.js:20](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/uniform.js#L20)

###### Inherited from

[`Distribution`](#distribution).[`name`](#name)

##### lower

```ts
lower: number | Object | any[];
```

Defined in: [distributions/uniform.js:24](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/uniform.js#L24)

##### upper

```ts
upper: number | any[] | undefined;
```

Defined in: [distributions/uniform.js:25](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/uniform.js#L25)

##### \_dist

```ts
_dist: any;
```

Defined in: [distributions/uniform.js:26](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/uniform.js#L26)

#### Methods

##### \_len()

```ts
_len(value): number;
```

Defined in: [distributions/base.js:69](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L69)

Broadcast length across value and parameters (0 = all scalar).

###### Parameters

###### value

`number` \| `any`[]

Value(s) whose length participates in broadcasting

###### Returns

`number`

The broadcast length (0 when every input is scalar)

###### Inherited from

[`Distribution`](#distribution).[`_len`](#_len)

##### \_paramsAt()

```ts
_paramsAt(i): Object;
```

Defined in: [distributions/base.js:82](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L82)

The proba parameter object with each array parameter indexed at `i`.

###### Parameters

###### i

`number`

Broadcast index

###### Returns

`Object`

Per-element parameter object (scalars passed through)

###### Inherited from

[`Distribution`](#distribution).[`_paramsAt`](#_paramsat)

##### logProb()

```ts
logProb(value): number | number[];
```

Defined in: [distributions/base.js:97](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L97)

Log probability density/mass function. Broadcasts over array values
and/or array parameters.

###### Parameters

###### value

`number` \| `Object` \| `any`[]

Value(s) to evaluate

###### Returns

`number` \| `number`[]

Log probability, elementwise for arrays

###### Inherited from

[`Distribution`](#distribution).[`logProb`](#logprob)

##### logpdf()

```ts
logpdf(value): number | number[];
```

Defined in: [distributions/base.js:150](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L150)

Alias for [Distribution#logProb](#logprob), matching the `@tangent.to/proba`
distribution contract (which names the method `logpdf`). Lets code written
against proba's distributions work unchanged on mc's.

###### Parameters

###### value

`number` \| `Object` \| `any`[]

Value(s) to evaluate

###### Returns

`number` \| `number`[]

###### Inherited from

[`Distribution`](#distribution).[`logpdf`](#logpdf)

##### dlogProbDx()

```ts
dlogProbDx(value): number | number[];
```

Defined in: [distributions/base.js:162](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L162)

Derivative of logProb with respect to the value, elementwise.
Used by Model.logProbAndGradient for analytic prior gradients.
Discrete distributions return 0 (no dx in their gradient contract).

###### Parameters

###### value

`number` \| `any`[]

Value(s) at which to differentiate

###### Returns

`number` \| `number`[]

###### Inherited from

[`Distribution`](#distribution).[`dlogProbDx`](#dlogprobdx)

##### pdf()

```ts
pdf(value): number | number[];
```

Defined in: [distributions/base.js:182](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L182)

Probability density/mass function, `exp(logProb(value))`.

###### Parameters

###### value

`number` \| `any`[]

Value(s) to evaluate

###### Returns

`number` \| `number`[]

###### Inherited from

[`Distribution`](#distribution).[`pdf`](#pdf)

##### cdf()

```ts
cdf(value): number;
```

Defined in: [distributions/base.js:192](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L192)

Cumulative distribution function (scalar parameters).

###### Parameters

###### value

`number`

###### Returns

`number`

###### Inherited from

[`Distribution`](#distribution).[`cdf`](#cdf)

##### quantile()

```ts
quantile(p): number;
```

Defined in: [distributions/base.js:201](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L201)

Quantile (inverse cdf) function (scalar parameters).

###### Parameters

###### p

`number`

Probability in [0, 1]

###### Returns

`number`

###### Inherited from

[`Distribution`](#distribution).[`quantile`](#quantile)

##### sample()

```ts
sample(shape?): number | number[];
```

Defined in: [distributions/base.js:213](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L213)

Sample from the distribution using the package RNG (see setRandomSeed).
`sample()` / `sample([])` return a number; `sample(n)` / `sample([n])`
return an Array of n draws.

###### Parameters

###### shape?

`number` \| `number`[]

Number of samples

###### Returns

`number` \| `number`[]

###### Inherited from

[`Distribution`](#distribution).[`sample`](#sample)

##### observe()

```ts
observe(data): Distribution;
```

Defined in: [distributions/base.js:224](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L224)

Set observed data for this distribution

###### Parameters

###### data

`number` \| `any`[]

Observed data

###### Returns

[`Distribution`](#distribution)

this, for chaining

###### Inherited from

[`Distribution`](#distribution).[`observe`](#observe)

##### mean()

```ts
mean(): number | number[];
```

Defined in: [distributions/base.js:233](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L233)

Get the mean of the distribution

###### Returns

`number` \| `number`[]

The mean

###### Inherited from

[`Distribution`](#distribution).[`mean`](#mean)

##### variance()

```ts
variance(): number | number[];
```

Defined in: [distributions/base.js:243](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/base.js#L243)

Get the variance of the distribution

###### Returns

`number` \| `number`[]

The variance

###### Inherited from

[`Distribution`](#distribution).[`variance`](#variance)

##### \_params()

```ts
_params(): object;
```

Defined in: [distributions/uniform.js:33](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/uniform.js#L33)

The proba parameter object for this distribution (proba `{low, high}` keys).

###### Returns

`object`

###### low

```ts
low: number | any[];
```

###### high

```ts
high: number | any[];
```

###### Overrides

[`Distribution`](#distribution).[`_params`](#_params)

##### logDensity()

```ts
logDensity(value): any;
```

Defined in: [distributions/uniform.js:37](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/uniform.js#L37)

The log-density as a differentiable expression, SUMMED over elements.

Where [Distribution#logProb](#logprob) takes plain numbers and returns the
elementwise density, this takes parameters that may be grad `Var`s, built
from the model's free variables, and returns one scalar `Var`: the total
log-density of `value` under this distribution, differentiable in every
parameter that is a `Var`. It is what `Model#observe` evaluates, so that a
likelihood is derived from the distribution rather than written by hand.

The seven built-in distributions implement it. A subclass that does not is
still a valid prior and a valid `logProb`; it is simply not differentiable,
and `observe` will say so.

###### Parameters

###### value

`any`

observed value(s), plain numbers

###### Returns

`any`

scalar

###### Overrides

[`Distribution`](#distribution).[`logDensity`](#logdensity)

##### getParams()

```ts
getParams(): object;
```

Defined in: [distributions/uniform.js:54](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/distributions/uniform.js#L54)

Get the distribution's parameters.

###### Returns

`object`

###### lower

```ts
lower: number | any[];
```

###### upper

```ts
upper: number | any[];
```

###### Overrides

[`Distribution`](#distribution).[`getParams`](#getparams)

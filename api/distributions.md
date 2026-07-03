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

Defined in: [distributions/base.js:27](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/base.js#L27)

Base class for probability distributions.
Provides common interface for all distributions.

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

Defined in: [distributions/base.js:28](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/base.js#L28)

###### Parameters

###### name?

`string` = `'Distribution'`

###### Returns

[`Distribution`](#distribution)

#### Properties

##### name

```ts
name: string;
```

Defined in: [distributions/base.js:29](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/base.js#L29)

##### observed

```ts
observed: Tensor<Rank> | null;
```

Defined in: [distributions/base.js:30](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/base.js#L30)

#### Methods

##### logProb()

```ts
logProb(value): Tensor<Rank>;
```

Defined in: [distributions/base.js:38](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/base.js#L38)

Log probability density/mass function

###### Parameters

###### value

`number` \| `Tensor`\<`Rank`\>

Value to evaluate

###### Returns

`Tensor`\<`Rank`\>

Log probability

##### pdf()

```ts
pdf(value): Tensor<Rank>;
```

Defined in: [distributions/base.js:51](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/base.js#L51)

Probability density/mass function

Computed as `exp(logProb(value))`. Provided for parity with the
`@tangent.to/ds` distribution interface (`pdf`/`cdf`/`quantile`).

###### Parameters

###### value

`number` \| `Tensor`\<`Rank`\>

Value to evaluate

###### Returns

`Tensor`\<`Rank`\>

Probability density/mass

##### sample()

```ts
sample(shape?): Tensor<Rank>;
```

Defined in: [distributions/base.js:60](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/base.js#L60)

Sample from the distribution

###### Parameters

###### shape?

`number` \| `number`[]

Shape of samples to generate

###### Returns

`Tensor`\<`Rank`\>

Samples

##### observe()

```ts
observe(data): Distribution;
```

Defined in: [distributions/base.js:68](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/base.js#L68)

Set observed data for this distribution

###### Parameters

###### data

`number` \| `any`[] \| `Tensor`\<`Rank`\>

Observed data

###### Returns

[`Distribution`](#distribution)

##### getParams()

```ts
getParams(): Object;
```

Defined in: [distributions/base.js:78](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/base.js#L78)

Get the distribution's parameters as a plain object.
Subclasses override to expose their specific parameters.

###### Returns

`Object`

Parameters

***

### Bernoulli

Defined in: [distributions/bernoulli.js:7](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/bernoulli.js#L7)

Bernoulli distribution for binary outcomes

#### Extends

- [`Distribution`](#distribution)

#### Constructors

##### Constructor

```ts
new Bernoulli(p?, name?): Bernoulli;
```

Defined in: [distributions/bernoulli.js:20](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/bernoulli.js#L20)

Accepts either positional arguments or a single options object.

###### Parameters

###### p?

`number` \| `Object` \| `Tensor`\<`Rank`\>

Probability of success in [0, 1], or an
  options object `{ p, name }`

###### name?

`string` = `'Bernoulli'`

Name of the distribution

###### Returns

[`Bernoulli`](#bernoulli)

###### Examples

```ts
new Bernoulli(0.7)
```

```ts
new Bernoulli({ p: 0.7 })
```

###### Overrides

[`Distribution`](#distribution).[`constructor`](#constructor)

#### Properties

##### observed

```ts
observed: Tensor<Rank> | null;
```

Defined in: [distributions/base.js:30](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/base.js#L30)

###### Inherited from

[`Distribution`](#distribution).[`observed`](#observed)

##### name

```ts
name: any;
```

Defined in: [distributions/bernoulli.js:24](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/bernoulli.js#L24)

###### Inherited from

[`Distribution`](#distribution).[`name`](#name)

##### p

```ts
p: Object | Tensor<Rank>;
```

Defined in: [distributions/bernoulli.js:27](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/bernoulli.js#L27)

#### Methods

##### pdf()

```ts
pdf(value): Tensor<Rank>;
```

Defined in: [distributions/base.js:51](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/base.js#L51)

Probability density/mass function

Computed as `exp(logProb(value))`. Provided for parity with the
`@tangent.to/ds` distribution interface (`pdf`/`cdf`/`quantile`).

###### Parameters

###### value

`number` \| `Tensor`\<`Rank`\>

Value to evaluate

###### Returns

`Tensor`\<`Rank`\>

Probability density/mass

###### Inherited from

[`Distribution`](#distribution).[`pdf`](#pdf)

##### observe()

```ts
observe(data): Bernoulli;
```

Defined in: [distributions/base.js:68](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/base.js#L68)

Set observed data for this distribution

###### Parameters

###### data

`number` \| `any`[] \| `Tensor`\<`Rank`\>

Observed data

###### Returns

[`Bernoulli`](#bernoulli)

###### Inherited from

[`Distribution`](#distribution).[`observe`](#observe)

##### logProb()

```ts
logProb(value): Tensor<Rank>;
```

Defined in: [distributions/bernoulli.js:35](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/bernoulli.js#L35)

Log probability mass function

###### Parameters

###### value

`number` \| `Tensor`\<`Rank`\>

Value to evaluate (0 or 1)

###### Returns

`Tensor`\<`Rank`\>

Log probability

###### Overrides

[`Distribution`](#distribution).[`logProb`](#logprob)

##### sample()

```ts
sample(shape?): Tensor<Rank>;
```

Defined in: [distributions/bernoulli.js:55](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/bernoulli.js#L55)

Sample from the Bernoulli distribution

###### Parameters

###### shape?

`number` \| `number`[]

Shape of samples to generate

###### Returns

`Tensor`\<`Rank`\>

Samples (0 or 1)

###### Overrides

[`Distribution`](#distribution).[`sample`](#sample)

##### mean()

```ts
mean(): Object | Tensor<Rank>;
```

Defined in: [distributions/bernoulli.js:66](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/bernoulli.js#L66)

Get the mean of the distribution

###### Returns

`Object` \| `Tensor`\<`Rank`\>

##### variance()

```ts
variance(): Tensor<Rank>;
```

Defined in: [distributions/bernoulli.js:73](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/bernoulli.js#L73)

Get the variance of the distribution

###### Returns

`Tensor`\<`Rank`\>

##### getParams()

```ts
getParams(): object;
```

Defined in: [distributions/bernoulli.js:81](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/bernoulli.js#L81)

Get the distribution's parameters.

###### Returns

`object`

###### p

```ts
p: number;
```

###### Overrides

[`Distribution`](#distribution).[`getParams`](#getparams)

***

### Beta

Defined in: [distributions/beta.js:8](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/beta.js#L8)

Beta distribution (useful for modeling probabilities)

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

Defined in: [distributions/beta.js:22](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/beta.js#L22)

Accepts either positional arguments or a single options object.

###### Parameters

###### alpha?

`number` \| `Object` \| `Tensor`\<`Rank`\>

Shape parameter (> 0), or an options
  object `{ alpha, beta, name }`

###### beta?

`number` \| `Tensor`\<`Rank`\>

Shape parameter (must be > 0)

###### name?

`string` = `'Beta'`

Name of the distribution

###### Returns

[`Beta`](#beta)

###### Examples

```ts
new Beta(2, 5)
```

```ts
new Beta({ alpha: 2, beta: 5 })
```

###### Overrides

[`Distribution`](#distribution).[`constructor`](#constructor)

#### Properties

##### observed

```ts
observed: Tensor<Rank> | null;
```

Defined in: [distributions/base.js:30](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/base.js#L30)

###### Inherited from

[`Distribution`](#distribution).[`observed`](#observed)

##### name

```ts
name: any;
```

Defined in: [distributions/beta.js:26](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/beta.js#L26)

###### Inherited from

[`Distribution`](#distribution).[`name`](#name)

##### alpha

```ts
alpha: Object | Tensor<Rank>;
```

Defined in: [distributions/beta.js:30](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/beta.js#L30)

##### beta

```ts
beta: Tensor<Rank> | undefined;
```

Defined in: [distributions/beta.js:31](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/beta.js#L31)

#### Methods

##### pdf()

```ts
pdf(value): Tensor<Rank>;
```

Defined in: [distributions/base.js:51](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/base.js#L51)

Probability density/mass function

Computed as `exp(logProb(value))`. Provided for parity with the
`@tangent.to/ds` distribution interface (`pdf`/`cdf`/`quantile`).

###### Parameters

###### value

`number` \| `Tensor`\<`Rank`\>

Value to evaluate

###### Returns

`Tensor`\<`Rank`\>

Probability density/mass

###### Inherited from

[`Distribution`](#distribution).[`pdf`](#pdf)

##### observe()

```ts
observe(data): Beta;
```

Defined in: [distributions/base.js:68](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/base.js#L68)

Set observed data for this distribution

###### Parameters

###### data

`number` \| `any`[] \| `Tensor`\<`Rank`\>

Observed data

###### Returns

[`Beta`](#beta)

###### Inherited from

[`Distribution`](#distribution).[`observe`](#observe)

##### logProb()

```ts
logProb(value): Tensor<Rank>;
```

Defined in: [distributions/beta.js:39](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/beta.js#L39)

Log probability density function

###### Parameters

###### value

`number` \| `Tensor`\<`Rank`\>

Value to evaluate (must be in [0, 1])

###### Returns

`Tensor`\<`Rank`\>

Log probability

###### Overrides

[`Distribution`](#distribution).[`logProb`](#logprob)

##### sample()

```ts
sample(shape?): Tensor<Rank>;
```

Defined in: [distributions/beta.js:69](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/beta.js#L69)

Sample from the beta distribution
Uses the relationship: if X ~ Gamma(α) and Y ~ Gamma(β), then X/(X+Y) ~ Beta(α, β)

###### Parameters

###### shape?

`number` \| `number`[]

Shape of samples to generate

###### Returns

`Tensor`\<`Rank`\>

Samples

###### Overrides

[`Distribution`](#distribution).[`sample`](#sample)

##### mean()

```ts
mean(): Tensor<Rank>;
```

Defined in: [distributions/beta.js:90](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/beta.js#L90)

Get the mean of the distribution

###### Returns

`Tensor`\<`Rank`\>

##### variance()

```ts
variance(): Tensor<Rank>;
```

Defined in: [distributions/beta.js:97](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/beta.js#L97)

Get the variance of the distribution

###### Returns

`Tensor`\<`Rank`\>

##### getParams()

```ts
getParams(): object;
```

Defined in: [distributions/beta.js:110](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/beta.js#L110)

Get the distribution's parameters.

###### Returns

`object`

###### alpha

```ts
alpha: number;
```

###### beta

```ts
beta: number;
```

###### Overrides

[`Distribution`](#distribution).[`getParams`](#getparams)

***

### Gamma

Defined in: [distributions/gamma.js:8](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/gamma.js#L8)

Gamma distribution (useful for modeling positive continuous values)

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

Defined in: [distributions/gamma.js:22](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/gamma.js#L22)

Accepts either positional arguments or a single options object.

###### Parameters

###### alpha?

`number` \| `Object` \| `Tensor`\<`Rank`\>

Shape parameter (> 0), or an options
  object `{ alpha | shape, beta | rate, name }`

###### beta?

`number` \| `Tensor`\<`Rank`\>

Rate parameter (must be > 0)

###### name?

`string` = `'Gamma'`

Name of the distribution

###### Returns

[`Gamma`](#gamma)

###### Examples

```ts
new Gamma(2, 1)
```

```ts
new Gamma({ shape: 2, rate: 1 })
```

###### Overrides

[`Distribution`](#distribution).[`constructor`](#constructor)

#### Properties

##### observed

```ts
observed: Tensor<Rank> | null;
```

Defined in: [distributions/base.js:30](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/base.js#L30)

###### Inherited from

[`Distribution`](#distribution).[`observed`](#observed)

##### name

```ts
name: any;
```

Defined in: [distributions/gamma.js:26](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/gamma.js#L26)

###### Inherited from

[`Distribution`](#distribution).[`name`](#name)

##### alpha

```ts
alpha: Object | Tensor<Rank>;
```

Defined in: [distributions/gamma.js:30](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/gamma.js#L30)

##### beta

```ts
beta: Tensor<Rank> | undefined;
```

Defined in: [distributions/gamma.js:31](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/gamma.js#L31)

#### Methods

##### pdf()

```ts
pdf(value): Tensor<Rank>;
```

Defined in: [distributions/base.js:51](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/base.js#L51)

Probability density/mass function

Computed as `exp(logProb(value))`. Provided for parity with the
`@tangent.to/ds` distribution interface (`pdf`/`cdf`/`quantile`).

###### Parameters

###### value

`number` \| `Tensor`\<`Rank`\>

Value to evaluate

###### Returns

`Tensor`\<`Rank`\>

Probability density/mass

###### Inherited from

[`Distribution`](#distribution).[`pdf`](#pdf)

##### observe()

```ts
observe(data): Gamma;
```

Defined in: [distributions/base.js:68](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/base.js#L68)

Set observed data for this distribution

###### Parameters

###### data

`number` \| `any`[] \| `Tensor`\<`Rank`\>

Observed data

###### Returns

[`Gamma`](#gamma)

###### Inherited from

[`Distribution`](#distribution).[`observe`](#observe)

##### logProb()

```ts
logProb(value): Tensor<Rank>;
```

Defined in: [distributions/gamma.js:39](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/gamma.js#L39)

Log probability density function

###### Parameters

###### value

`number` \| `Tensor`\<`Rank`\>

Value to evaluate (must be > 0)

###### Returns

`Tensor`\<`Rank`\>

Log probability

###### Overrides

[`Distribution`](#distribution).[`logProb`](#logprob)

##### sample()

```ts
sample(shape?): Tensor<Rank>;
```

Defined in: [distributions/gamma.js:75](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/gamma.js#L75)

Sample from the gamma distribution

###### Parameters

###### shape?

`number` \| `number`[]

Shape of samples to generate

###### Returns

`Tensor`\<`Rank`\>

Samples

###### Overrides

[`Distribution`](#distribution).[`sample`](#sample)

##### mean()

```ts
mean(): Tensor<Rank>;
```

Defined in: [distributions/gamma.js:96](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/gamma.js#L96)

Get the mean of the distribution

###### Returns

`Tensor`\<`Rank`\>

##### variance()

```ts
variance(): Tensor<Rank>;
```

Defined in: [distributions/gamma.js:103](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/gamma.js#L103)

Get the variance of the distribution

###### Returns

`Tensor`\<`Rank`\>

##### getParams()

```ts
getParams(): object;
```

Defined in: [distributions/gamma.js:111](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/gamma.js#L111)

Get the distribution's parameters.

###### Returns

`object`

###### alpha

```ts
alpha: number;
```

###### beta

```ts
beta: number;
```

###### Overrides

[`Distribution`](#distribution).[`getParams`](#getparams)

***

### HalfNormal

Defined in: [distributions/halfnormal.js:19](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/halfnormal.js#L19)

Half-normal distribution

The distribution of $|Z|$ where $Z \sim \mathcal{N}(0, \sigma^2)$; a positive
variable concentrated near zero. Commonly used as a weakly-informative prior
for scale / standard-deviation parameters (variance components).

Probability density function (for $x \ge 0$):
$$
p(x \mid \sigma) = \frac{\sqrt{2}}{\sigma\sqrt{\pi}}
  \exp\!\left(-\frac{x^2}{2\sigma^2}\right)
$$

#### See

[distribution](https://en.wikipedia.org/wiki/Half-normal_distribution|Half-normal)

#### Extends

- [`Distribution`](#distribution)

#### Constructors

##### Constructor

```ts
new HalfNormal(sigma?, name?): HalfNormal;
```

Defined in: [distributions/halfnormal.js:24](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/halfnormal.js#L24)

###### Parameters

###### sigma?

`number` \| `Tensor`\<`Rank`\>

Scale parameter ($\sigma > 0$)

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
observed: Tensor<Rank> | null;
```

Defined in: [distributions/base.js:30](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/base.js#L30)

###### Inherited from

[`Distribution`](#distribution).[`observed`](#observed)

##### name

```ts
name: any;
```

Defined in: [distributions/halfnormal.js:28](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/halfnormal.js#L28)

###### Inherited from

[`Distribution`](#distribution).[`name`](#name)

##### sigma

```ts
sigma: Tensor<Rank>;
```

Defined in: [distributions/halfnormal.js:31](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/halfnormal.js#L31)

#### Methods

##### pdf()

```ts
pdf(value): Tensor<Rank>;
```

Defined in: [distributions/base.js:51](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/base.js#L51)

Probability density/mass function

Computed as `exp(logProb(value))`. Provided for parity with the
`@tangent.to/ds` distribution interface (`pdf`/`cdf`/`quantile`).

###### Parameters

###### value

`number` \| `Tensor`\<`Rank`\>

Value to evaluate

###### Returns

`Tensor`\<`Rank`\>

Probability density/mass

###### Inherited from

[`Distribution`](#distribution).[`pdf`](#pdf)

##### observe()

```ts
observe(data): HalfNormal;
```

Defined in: [distributions/base.js:68](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/base.js#L68)

Set observed data for this distribution

###### Parameters

###### data

`number` \| `any`[] \| `Tensor`\<`Rank`\>

Observed data

###### Returns

[`HalfNormal`](#halfnormal)

###### Inherited from

[`Distribution`](#distribution).[`observe`](#observe)

##### logProb()

```ts
logProb(value): Tensor<Rank>;
```

Defined in: [distributions/halfnormal.js:47](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/halfnormal.js#L47)

Log probability density function.

$$
\log p(x) = \tfrac{1}{2}\log\frac{2}{\pi} - \log\sigma - \frac{x^2}{2\sigma^2},
\quad x \ge 0
$$

Returns $-\infty$ for negative inputs.

###### Parameters

###### value

`number` \| `Tensor`\<`Rank`\>

Value to evaluate ($x \ge 0$)

###### Returns

`Tensor`\<`Rank`\>

Log probability density

###### Overrides

[`Distribution`](#distribution).[`logProb`](#logprob)

##### sample()

```ts
sample(shape?): Tensor<Rank>;
```

Defined in: [distributions/halfnormal.js:66](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/halfnormal.js#L66)

Sample from the half-normal distribution: $|\sigma Z|$, $Z \sim \mathcal{N}(0,1)$.

###### Parameters

###### shape?

`number` \| `number`[]

Shape of samples to generate

###### Returns

`Tensor`\<`Rank`\>

Samples

###### Overrides

[`Distribution`](#distribution).[`sample`](#sample)

##### mean()

```ts
mean(): Tensor<Rank>;
```

Defined in: [distributions/halfnormal.js:78](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/halfnormal.js#L78)

Mean of the distribution: $\sigma\sqrt{2/\pi}$.

###### Returns

`Tensor`\<`Rank`\>

The mean

##### variance()

```ts
variance(): Tensor<Rank>;
```

Defined in: [distributions/halfnormal.js:86](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/halfnormal.js#L86)

Variance of the distribution: sigma^2 * (1 - 2/pi).

###### Returns

`Tensor`\<`Rank`\>

The variance

##### getParams()

```ts
getParams(): object;
```

Defined in: [distributions/halfnormal.js:94](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/halfnormal.js#L94)

Get the distribution's parameters.

###### Returns

`object`

###### sigma

```ts
sigma: number;
```

###### Overrides

[`Distribution`](#distribution).[`getParams`](#getparams)

***

### Lognormal

Defined in: [distributions/lognormal.js:21](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/lognormal.js#L21)

Log-normal distribution

A positive random variable whose logarithm is normally distributed:
if $\log X \sim \mathcal{N}(\mu, \sigma^2)$ then $X \sim \text{LogNormal}(\mu, \sigma)$.

Probability density function (for $x > 0$):
$$
p(x \mid \mu, \sigma) = \frac{1}{x\,\sigma\sqrt{2\pi}}
  \exp\!\left(-\frac{(\log x - \mu)^2}{2\sigma^2}\right)
$$

Useful as a weakly-informative prior for strictly positive quantities
(rates, scales, plateaus).

#### See

[distribution](https://en.wikipedia.org/wiki/Log-normal_distribution|Log-normal)

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

Defined in: [distributions/lognormal.js:27](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/lognormal.js#L27)

###### Parameters

###### mu?

`number` \| `Tensor`\<`Rank`\>

Mean of the underlying normal (log-scale)

###### sigma?

`number` \| `Tensor`\<`Rank`\>

Std-dev of the underlying normal ($\sigma > 0$)

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
observed: Tensor<Rank> | null;
```

Defined in: [distributions/base.js:30](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/base.js#L30)

###### Inherited from

[`Distribution`](#distribution).[`observed`](#observed)

##### name

```ts
name: any;
```

Defined in: [distributions/lognormal.js:31](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/lognormal.js#L31)

###### Inherited from

[`Distribution`](#distribution).[`name`](#name)

##### mu

```ts
mu: Tensor<Rank>;
```

Defined in: [distributions/lognormal.js:35](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/lognormal.js#L35)

##### sigma

```ts
sigma: Tensor<Rank>;
```

Defined in: [distributions/lognormal.js:36](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/lognormal.js#L36)

#### Methods

##### pdf()

```ts
pdf(value): Tensor<Rank>;
```

Defined in: [distributions/base.js:51](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/base.js#L51)

Probability density/mass function

Computed as `exp(logProb(value))`. Provided for parity with the
`@tangent.to/ds` distribution interface (`pdf`/`cdf`/`quantile`).

###### Parameters

###### value

`number` \| `Tensor`\<`Rank`\>

Value to evaluate

###### Returns

`Tensor`\<`Rank`\>

Probability density/mass

###### Inherited from

[`Distribution`](#distribution).[`pdf`](#pdf)

##### observe()

```ts
observe(data): Lognormal;
```

Defined in: [distributions/base.js:68](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/base.js#L68)

Set observed data for this distribution

###### Parameters

###### data

`number` \| `any`[] \| `Tensor`\<`Rank`\>

Observed data

###### Returns

[`Lognormal`](#lognormal)

###### Inherited from

[`Distribution`](#distribution).[`observe`](#observe)

##### logProb()

```ts
logProb(value): Tensor<Rank>;
```

Defined in: [distributions/lognormal.js:50](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/lognormal.js#L50)

Log probability density function.

$$
\log p(x) = -\log x - \log \sigma - \tfrac{1}{2}\log(2\pi)
           - \frac{(\log x - \mu)^2}{2\sigma^2}, \quad x > 0
$$

###### Parameters

###### value

`number` \| `Tensor`\<`Rank`\>

Value to evaluate ($x > 0$)

###### Returns

`Tensor`\<`Rank`\>

Log probability density

###### Overrides

[`Distribution`](#distribution).[`logProb`](#logprob)

##### sample()

```ts
sample(shape?): Tensor<Rank>;
```

Defined in: [distributions/lognormal.js:69](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/lognormal.js#L69)

Sample from the log-normal distribution: $\exp(\mu + \sigma Z)$, $Z \sim \mathcal{N}(0,1)$.

###### Parameters

###### shape?

`number` \| `number`[]

Shape of samples to generate

###### Returns

`Tensor`\<`Rank`\>

Samples

###### Overrides

[`Distribution`](#distribution).[`sample`](#sample)

##### mean()

```ts
mean(): Tensor<Rank>;
```

Defined in: [distributions/lognormal.js:81](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/lognormal.js#L81)

Mean of the distribution: $\exp(\mu + \sigma^2/2)$.

###### Returns

`Tensor`\<`Rank`\>

The mean

##### variance()

```ts
variance(): Tensor<Rank>;
```

Defined in: [distributions/lognormal.js:89](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/lognormal.js#L89)

Variance of the distribution: (exp(sigma^2) - 1) * exp(2*mu + sigma^2).

###### Returns

`Tensor`\<`Rank`\>

The variance

##### getParams()

```ts
getParams(): object;
```

Defined in: [distributions/lognormal.js:100](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/lognormal.js#L100)

Get the distribution's parameters.

###### Returns

`object`

###### mu

```ts
mu: number;
```

###### sigma

```ts
sigma: number;
```

###### Overrides

[`Distribution`](#distribution).[`getParams`](#getparams)

***

### Normal

Defined in: [distributions/normal.js:14](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/normal.js#L14)

Normal (Gaussian) distribution

Probability density function:
$$
p(x | \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

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

Defined in: [distributions/normal.js:29](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/normal.js#L29)

Accepts either positional arguments or a single options object, matching the
dual-constructor convention of `@tangent.to/ds`.

###### Parameters

###### mu?

`number` \| `Object` \| `Tensor`\<`Rank`\>

Mean parameter $\mu$, or an options object
  `{ mu | mean, sigma | sd, name }`

###### sigma?

`number` \| `Tensor`\<`Rank`\>

Standard deviation parameter $\sigma > 0$

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
observed: Tensor<Rank> | null;
```

Defined in: [distributions/base.js:30](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/base.js#L30)

###### Inherited from

[`Distribution`](#distribution).[`observed`](#observed)

##### name

```ts
name: any;
```

Defined in: [distributions/normal.js:33](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/normal.js#L33)

###### Inherited from

[`Distribution`](#distribution).[`name`](#name)

##### mu

```ts
mu: Object | Tensor<Rank>;
```

Defined in: [distributions/normal.js:37](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/normal.js#L37)

##### sigma

```ts
sigma: Tensor<Rank> | undefined;
```

Defined in: [distributions/normal.js:38](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/normal.js#L38)

#### Methods

##### pdf()

```ts
pdf(value): Tensor<Rank>;
```

Defined in: [distributions/base.js:51](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/base.js#L51)

Probability density/mass function

Computed as `exp(logProb(value))`. Provided for parity with the
`@tangent.to/ds` distribution interface (`pdf`/`cdf`/`quantile`).

###### Parameters

###### value

`number` \| `Tensor`\<`Rank`\>

Value to evaluate

###### Returns

`Tensor`\<`Rank`\>

Probability density/mass

###### Inherited from

[`Distribution`](#distribution).[`pdf`](#pdf)

##### observe()

```ts
observe(data): Normal;
```

Defined in: [distributions/base.js:68](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/base.js#L68)

Set observed data for this distribution

###### Parameters

###### data

`number` \| `any`[] \| `Tensor`\<`Rank`\>

Observed data

###### Returns

[`Normal`](#normal)

###### Inherited from

[`Distribution`](#distribution).[`observe`](#observe)

##### logProb()

```ts
logProb(value): Tensor<Rank>;
```

Defined in: [distributions/normal.js:51](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/normal.js#L51)

Log probability density function

$$
\log p(x | \mu, \sigma) = -\frac{1}{2}\log(2\pi) - \log(\sigma) - \frac{(x-\mu)^2}{2\sigma^2}
$$

###### Parameters

###### value

`number` \| `Tensor`\<`Rank`\>

Value to evaluate

###### Returns

`Tensor`\<`Rank`\>

Log probability

###### Overrides

[`Distribution`](#distribution).[`logProb`](#logprob)

##### sample()

```ts
sample(shape?): Tensor<Rank>;
```

Defined in: [distributions/normal.js:71](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/normal.js#L71)

Sample from the normal distribution

###### Parameters

###### shape?

`number` \| `number`[]

Shape of samples to generate

###### Returns

`Tensor`\<`Rank`\>

Samples

###### Overrides

[`Distribution`](#distribution).[`sample`](#sample)

##### mean()

```ts
mean(): Object | Tensor<Rank>;
```

Defined in: [distributions/normal.js:82](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/normal.js#L82)

Get the mean of the distribution

###### Returns

`Object` \| `Tensor`\<`Rank`\>

##### variance()

```ts
variance(): Tensor<Rank>;
```

Defined in: [distributions/normal.js:89](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/normal.js#L89)

Get the variance of the distribution

###### Returns

`Tensor`\<`Rank`\>

##### getParams()

```ts
getParams(): object;
```

Defined in: [distributions/normal.js:97](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/normal.js#L97)

Get the distribution's parameters.

###### Returns

`object`

###### mu

```ts
mu: number;
```

###### sigma

```ts
sigma: number;
```

###### Overrides

[`Distribution`](#distribution).[`getParams`](#getparams)

***

### Uniform

Defined in: [distributions/uniform.js:7](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/uniform.js#L7)

Uniform distribution

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

Defined in: [distributions/uniform.js:21](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/uniform.js#L21)

Accepts either positional arguments or a single options object.

###### Parameters

###### lower?

`number` \| `Object` \| `Tensor`\<`Rank`\>

Lower bound, or an options object
  `{ lower | min, upper | max, name }`

###### upper?

`number` \| `Tensor`\<`Rank`\>

Upper bound

###### name?

`string` = `'Uniform'`

Name of the distribution

###### Returns

[`Uniform`](#uniform)

###### Examples

```ts
new Uniform(0, 1)
```

```ts
new Uniform({ min: 0, max: 1 })
```

###### Overrides

[`Distribution`](#distribution).[`constructor`](#constructor)

#### Properties

##### observed

```ts
observed: Tensor<Rank> | null;
```

Defined in: [distributions/base.js:30](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/base.js#L30)

###### Inherited from

[`Distribution`](#distribution).[`observed`](#observed)

##### name

```ts
name: any;
```

Defined in: [distributions/uniform.js:25](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/uniform.js#L25)

###### Inherited from

[`Distribution`](#distribution).[`name`](#name)

##### lower

```ts
lower: Object | Tensor<Rank>;
```

Defined in: [distributions/uniform.js:29](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/uniform.js#L29)

##### upper

```ts
upper: Tensor<Rank> | undefined;
```

Defined in: [distributions/uniform.js:30](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/uniform.js#L30)

#### Methods

##### pdf()

```ts
pdf(value): Tensor<Rank>;
```

Defined in: [distributions/base.js:51](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/base.js#L51)

Probability density/mass function

Computed as `exp(logProb(value))`. Provided for parity with the
`@tangent.to/ds` distribution interface (`pdf`/`cdf`/`quantile`).

###### Parameters

###### value

`number` \| `Tensor`\<`Rank`\>

Value to evaluate

###### Returns

`Tensor`\<`Rank`\>

Probability density/mass

###### Inherited from

[`Distribution`](#distribution).[`pdf`](#pdf)

##### observe()

```ts
observe(data): Uniform;
```

Defined in: [distributions/base.js:68](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/base.js#L68)

Set observed data for this distribution

###### Parameters

###### data

`number` \| `any`[] \| `Tensor`\<`Rank`\>

Observed data

###### Returns

[`Uniform`](#uniform)

###### Inherited from

[`Distribution`](#distribution).[`observe`](#observe)

##### logProb()

```ts
logProb(value): Tensor<Rank>;
```

Defined in: [distributions/uniform.js:38](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/uniform.js#L38)

Log probability density function

###### Parameters

###### value

`number` \| `Tensor`\<`Rank`\>

Value to evaluate

###### Returns

`Tensor`\<`Rank`\>

Log probability

###### Overrides

[`Distribution`](#distribution).[`logProb`](#logprob)

##### sample()

```ts
sample(shape?): Tensor<Rank>;
```

Defined in: [distributions/uniform.js:60](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/uniform.js#L60)

Sample from the uniform distribution

###### Parameters

###### shape?

`number` \| `number`[]

Shape of samples to generate

###### Returns

`Tensor`\<`Rank`\>

Samples

###### Overrides

[`Distribution`](#distribution).[`sample`](#sample)

##### mean()

```ts
mean(): Tensor<Rank>;
```

Defined in: [distributions/uniform.js:72](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/uniform.js#L72)

Get the mean of the distribution

###### Returns

`Tensor`\<`Rank`\>

##### variance()

```ts
variance(): Tensor<Rank>;
```

Defined in: [distributions/uniform.js:79](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/uniform.js#L79)

Get the variance of the distribution

###### Returns

`Tensor`\<`Rank`\>

##### getParams()

```ts
getParams(): object;
```

Defined in: [distributions/uniform.js:88](https://github.com/tangent-to/mc/blob/c32ffd3caf22b47cd6803332f7f477969f9f6e95/src/distributions/uniform.js#L88)

Get the distribution's parameters.

###### Returns

`object`

###### lower

```ts
lower: number;
```

###### upper

```ts
upper: number;
```

###### Overrides

[`Distribution`](#distribution).[`getParams`](#getparams)

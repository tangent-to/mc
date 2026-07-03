---
layout: default
title: samplers
parent: API Reference
nav_order: 3
permalink: /api/samplers
---
# samplers

## Classes

### HMC

Defined in: [samplers/hmc-vector.js:22](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/hmc-vector.js#L22)

Vector-aware Hamiltonian Monte Carlo.

Unlike the scalar `HamiltonianMC`/`NUTS` in this package, this sampler flattens
all free variables - scalars and 1-D vectors alike - into a single real vector
and runs leapfrog dynamics on it. That makes it suitable for hierarchical
models whose parameters are vectors (per-group effects, per-site plateaus, …)
and for likelihoods defined through Model#potential (a deterministic
mean computed from the latent variables and data).

Step size is tuned during warm-up by dual averaging (Hoffman & Gelman, 2014)
toward a target acceptance rate; a unit mass matrix is used.

#### Example

```ts
const hmc = new HMC({ stepSize: 0.05, nSteps: 20 });
const { trace } = hmc.sample(model, { slope: 0, intercept: 0, sigma: 1 },
                             { nSamples: 1000, nWarmup: 500 });
```

#### Constructors

##### Constructor

```ts
new HMC(opts?): HMC;
```

Defined in: [samplers/hmc-vector.js:31](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/hmc-vector.js#L31)

###### Parameters

###### opts?

###### stepSize?

`number` = `0.05`

Initial leapfrog step size (adapted in warm-up).

###### nSteps?

`number` = `20`

Leapfrog steps per proposal.

###### targetAccept?

`number` = `0.8`

Target acceptance for step-size adaptation.

###### adapt?

`boolean` = `true`

Adapt the step size during warm-up.

###### seed?

`number`

Optional RNG seed for reproducibility.

###### Returns

[`HMC`](#hmc)

#### Properties

##### stepSize

```ts
stepSize: number;
```

Defined in: [samplers/hmc-vector.js:32](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/hmc-vector.js#L32)

##### nSteps

```ts
nSteps: number;
```

Defined in: [samplers/hmc-vector.js:33](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/hmc-vector.js#L33)

##### targetAccept

```ts
targetAccept: number;
```

Defined in: [samplers/hmc-vector.js:34](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/hmc-vector.js#L34)

##### adapt

```ts
adapt: boolean;
```

Defined in: [samplers/hmc-vector.js:35](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/hmc-vector.js#L35)

##### seed

```ts
seed: number | undefined;
```

Defined in: [samplers/hmc-vector.js:36](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/hmc-vector.js#L36)

#### Methods

##### sample()

```ts
sample(
   model, 
   initialValues, 
   opts?): object;
```

Defined in: [samplers/hmc-vector.js:52](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/hmc-vector.js#L52)

Run a single chain.

###### Parameters

###### model

`Model`

###### initialValues

`Object`

{name: number | number[]} starting point.

###### opts?

###### nSamples?

`number` = `1000`

###### nWarmup?

`number` = `500`

###### thin?

`number` = `1`

###### progress?

`boolean` = `false`

###### Returns

`object`

###### trace

```ts
trace: Object;
```

###### acceptanceRate

```ts
acceptanceRate: number;
```

###### stepSize

```ts
stepSize: number;
```

###### divergences

```ts
divergences: number;
```

###### specs

```ts
specs: any[];
```

##### sampleChains()

```ts
sampleChains(
   model, 
   initial, 
   opts?): any[];
```

Defined in: [samplers/hmc-vector.js:203](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/hmc-vector.js#L203)

Run several independent chains (sequentially) from (optionally) jittered
starting points. Returns an array of single-chain results, ready for
[summary](#summary).

###### Parameters

###### model

`Model`

###### initial

`Object` \| ((`chain`) => `Object`)

Starting values, or a
  function returning starting values for each chain index.

###### opts?

`Object` = `{}`

As [HMC#sample](#sample), plus `chains` (default 4).

###### Returns

`any`[]

per-chain results

***

### HamiltonianMC

Defined in: [samplers/hmc.js:24](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/hmc.js#L24)

Hamiltonian Monte Carlo (HMC) sampler

Uses gradient information for efficient exploration of the posterior.
HMC simulates Hamiltonian dynamics to propose distant states with high acceptance probability.

**Hamiltonian**:
$$
H(\theta, p) = -\log p(\theta|y) + \frac{1}{2}p^T p
$$
where $\theta$ is position (parameters), $p$ is momentum.

**Leapfrog integrator** preserves volume and is reversible:
1. Half-step momentum: $p_{i+1/2} = p_i + \frac{\epsilon}{2}\nabla_\theta \log p(\theta_i|y)$
2. Full-step position: $\theta_{i+1} = \theta_i + \epsilon p_{i+1/2}$
3. Half-step momentum: $p_{i+1} = p_{i+1/2} + \frac{\epsilon}{2}\nabla_\theta \log p(\theta_{i+1}|y)$

#### See

[Conceptual Introduction to HMC](https://arxiv.org/abs/1701.02434|A)

#### Constructors

##### Constructor

```ts
new HamiltonianMC(stepSize?, nSteps?): HamiltonianMC;
```

Defined in: [samplers/hmc.js:37](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/hmc.js#L37)

Accepts either positional arguments or a single options object.

###### Parameters

###### stepSize?

`number` \| `Object`

Leapfrog step size (epsilon), or an options
  object `{ stepSize, nSteps }`

###### nSteps?

`number` = `10`

Number of leapfrog steps (L)

###### Returns

[`HamiltonianMC`](#hamiltonianmc)

###### Examples

```ts
new HamiltonianMC(0.01, 10)
```

```ts
new HamiltonianMC({ stepSize: 0.01, nSteps: 10 })
```

#### Properties

##### stepSize

```ts
stepSize: number | Object;
```

Defined in: [samplers/hmc.js:43](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/hmc.js#L43)

##### nSteps

```ts
nSteps: number | undefined;
```

Defined in: [samplers/hmc.js:44](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/hmc.js#L44)

#### Methods

##### getParams()

```ts
getParams(): object;
```

Defined in: [samplers/hmc.js:51](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/hmc.js#L51)

Get the sampler's configuration.

###### Returns

`object`

###### stepSize

```ts
stepSize: number;
```

###### nSteps

```ts
nSteps: number;
```

##### leapfrog()

```ts
leapfrog(
   position, 
   momentum, 
   model): Object;
```

Defined in: [samplers/hmc.js:62](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/hmc.js#L62)

Leapfrog integrator for Hamiltonian dynamics

###### Parameters

###### position

`Object`

Current position (parameters)

###### momentum

`Object`

Current momentum

###### model

`Model`

The probabilistic model

###### Returns

`Object`

New position and momentum

##### hamiltonian()

```ts
hamiltonian(
   position, 
   momentum, 
   model): number;
```

Defined in: [samplers/hmc.js:126](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/hmc.js#L126)

Compute Hamiltonian (total energy)

###### Parameters

###### position

`Object`

Current position

###### momentum

`Object`

Current momentum

###### model

`Model`

The probabilistic model

###### Returns

`number`

Hamiltonian value

##### sample()

```ts
sample(
   model, 
   initialValues, 
   nSamples?, 
   burnIn?, 
   thin?): Object;
```

Defined in: [samplers/hmc.js:153](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/hmc.js#L153)

Run HMC sampling.

The sampling controls may be passed positionally or as a single options
object. When an options object is supplied as the third argument, the
`burnIn` and `thin` positional arguments are ignored in favour of the
object's fields.

###### Parameters

###### model

`Model`

The probabilistic model

###### initialValues

`Object`

Initial parameter values

###### nSamples?

`number` \| `Object`

Number of samples, or an options object

`number`

***

`Object`

###### burnIn?

`number` = `500`

Number of burn-in samples to discard (positional form)

###### thin?

`number` = `1`

Thinning interval (positional form)

###### Returns

`Object`

Trace object with samples and diagnostics

###### Examples

```ts
hmc.sample(model, { mu: 0 }, 1000, 500, 1)
```

```ts
hmc.sample(model, { mu: 0 }, { nSamples: 1000, burnIn: 500, thin: 1 })
```

***

### MetropolisHastings

Defined in: [samplers/metropolis.js:21](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/metropolis.js#L21)

Metropolis-Hastings MCMC sampler

A simple but effective MCMC algorithm for sampling from posterior distributions.

**Algorithm**: At each iteration, a proposal $\theta'$ is generated from a symmetric
proposal distribution $q(\theta'|\theta) = \mathcal{N}(\theta, \sigma^2)$.
The proposal is accepted with probability:
$$
\alpha = \min\left(1, \frac{p(\theta'|y)}{p(\theta|y)}\right)
$$

**Optimal acceptance rate**: Target ~23.4% for high-dimensional problems, 44% for 1D.

#### See

[https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings\_algorithm\|Metropolis-Hastings](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm|Metropolis-Hastings)

#### Constructors

##### Constructor

```ts
new MetropolisHastings(proposalStd?): MetropolisHastings;
```

Defined in: [samplers/metropolis.js:33](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/metropolis.js#L33)

Accepts either a positional argument or a single options object.

###### Parameters

###### proposalStd?

`number` \| `Object`

Standard deviation for the Gaussian
  proposal distribution, or an options object `{ proposalStd }`

###### Returns

[`MetropolisHastings`](#metropolishastings)

###### Examples

```ts
new MetropolisHastings(0.5)
```

```ts
new MetropolisHastings({ proposalStd: 0.5 })
```

#### Properties

##### proposalStd

```ts
proposalStd: number | Object;
```

Defined in: [samplers/metropolis.js:37](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/metropolis.js#L37)

#### Methods

##### getParams()

```ts
getParams(): object;
```

Defined in: [samplers/metropolis.js:44](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/metropolis.js#L44)

Get the sampler's configuration.

###### Returns

`object`

###### proposalStd

```ts
proposalStd: number;
```

##### sample()

```ts
sample(
   model, 
   initialValues, 
   nSamples?, 
   burnIn?, 
   thin?): Object;
```

Defined in: [samplers/metropolis.js:71](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/metropolis.js#L71)

Run Metropolis-Hastings sampling.

The sampling controls may be passed positionally or as a single options
object. When an options object is supplied as the third argument, the
`burnIn` and `thin` positional arguments are ignored in favour of the
object's fields.

###### Parameters

###### model

`Model`

The probabilistic model

###### initialValues

`Object`

Initial parameter values

###### nSamples?

`number` \| `Object`

Number of samples, or an options object

`number`

***

`Object`

###### burnIn?

`number` = `500`

Number of burn-in samples to discard (positional form)

###### thin?

`number` = `1`

Thinning interval, keep every nth sample (positional form)

###### Returns

`Object`

Trace object with samples and diagnostics

###### Examples

```ts
mh.sample(model, { mu: 0 }, 1000, 500, 1)
```

```ts
mh.sample(model, { mu: 0 }, { nSamples: 1000, burnIn: 500, thin: 1 })
```

##### tuneProposal()

```ts
tuneProposal(currentAcceptanceRate): number;
```

Defined in: [samplers/metropolis.js:145](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/metropolis.js#L145)

Tune the proposal standard deviation to achieve target acceptance rate

###### Parameters

###### currentAcceptanceRate

`number`

Current acceptance rate

###### Returns

`number`

New proposal standard deviation

***

### NUTS

Defined in: [samplers/nuts.js:28](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/nuts.js#L28)

No-U-Turn Sampler (NUTS)

An extension of Hamiltonian Monte Carlo that automatically tunes the trajectory length.
NUTS eliminates the need to manually set the number of leapfrog steps by running
until the trajectory makes a "U-turn" (starts coming back).

**Algorithm**: Uses recursive tree doubling to adaptively determine path length.
The trajectory is stopped when:
$$
(p^+ - p^-) \cdot \theta^+ < 0 \quad \text{or} \quad (p^+ - p^-) \cdot \theta^- < 0
$$
where $\theta^+, p^+$ are the forward endpoint and $\theta^-, p^-$ are the backward endpoint.

**Advantages over HMC:**
- No manual tuning of trajectory length
- Better exploration of complex posteriors
- State-of-the-art MCMC performance

**Dual averaging** is used to automatically tune step size during warm-up.

#### See

[No-U-Turn Sampler (Hoffman & Gelman, 2014)](https://arxiv.org/abs/1111.4246|The)

#### Constructors

##### Constructor

```ts
new NUTS(
   stepSize?, 
   maxTreeDepth?, 
   targetAcceptance?): NUTS;
```

Defined in: [samplers/nuts.js:42](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/nuts.js#L42)

Accepts either positional arguments or a single options object.

###### Parameters

###### stepSize?

`number` \| `Object`

Initial leapfrog step size (adapted during
  warmup), or an options object `{ stepSize, maxTreeDepth, targetAcceptance }`

###### maxTreeDepth?

`number` = `10`

Maximum tree depth (default 10, up to 2^10 steps)

###### targetAcceptance?

`number` = `0.8`

Target acceptance rate for adaptation (default 0.8)

###### Returns

[`NUTS`](#nuts)

###### Examples

```ts
new NUTS(0.01, 10, 0.8)
```

```ts
new NUTS({ stepSize: 0.01, maxTreeDepth: 10, targetAcceptance: 0.8 })
```

#### Properties

##### stepSize

```ts
stepSize: number | Object;
```

Defined in: [samplers/nuts.js:49](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/nuts.js#L49)

##### maxTreeDepth

```ts
maxTreeDepth: number | undefined;
```

Defined in: [samplers/nuts.js:50](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/nuts.js#L50)

##### targetAcceptance

```ts
targetAcceptance: number | undefined;
```

Defined in: [samplers/nuts.js:51](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/nuts.js#L51)

##### mu

```ts
mu: number;
```

Defined in: [samplers/nuts.js:54](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/nuts.js#L54)

##### gamma

```ts
gamma: number;
```

Defined in: [samplers/nuts.js:55](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/nuts.js#L55)

##### t0

```ts
t0: number;
```

Defined in: [samplers/nuts.js:56](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/nuts.js#L56)

##### kappa

```ts
kappa: number;
```

Defined in: [samplers/nuts.js:57](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/nuts.js#L57)

#### Methods

##### getParams()

```ts
getParams(): object;
```

Defined in: [samplers/nuts.js:64](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/nuts.js#L64)

Get the sampler's configuration.

###### Returns

`object`

###### stepSize

```ts
stepSize: number;
```

###### maxTreeDepth

```ts
maxTreeDepth: number;
```

###### targetAcceptance

```ts
targetAcceptance: number;
```

##### leapfrog()

```ts
leapfrog(
   position, 
   momentum, 
   stepSize, 
   model): Object;
```

Defined in: [samplers/nuts.js:80](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/nuts.js#L80)

Single leapfrog step

###### Parameters

###### position

`Object`

Current position (parameters)

###### momentum

`Object`

Current momentum

###### stepSize

`number`

Step size for this step

###### model

`Model`

The probabilistic model

###### Returns

`Object`

New position and momentum

##### hamiltonian()

```ts
hamiltonian(
   position, 
   momentum, 
   model): number;
```

Defined in: [samplers/nuts.js:132](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/nuts.js#L132)

Compute Hamiltonian (total energy)

###### Parameters

###### position

`Object`

Current position

###### momentum

`Object`

Current momentum

###### model

`Model`

The probabilistic model

###### Returns

`number`

Hamiltonian value

##### isUTurn()

```ts
isUTurn(
   positionMinus, 
   positionPlus, 
   momentumMinus, 
   momentumPlus): boolean;
```

Defined in: [samplers/nuts.js:144](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/nuts.js#L144)

Check if trajectory is making a U-turn

###### Parameters

###### positionMinus

`Object`

Backward endpoint position

###### positionPlus

`Object`

Forward endpoint position

###### momentumMinus

`Object`

Backward endpoint momentum

###### momentumPlus

`Object`

Forward endpoint momentum

###### Returns

`boolean`

True if trajectory is making a U-turn

##### buildTree()

```ts
buildTree(
   position, 
   momentum, 
   slice, 
   direction, 
   depth, 
   stepSize, 
   model, 
   H0): Object;
```

Defined in: [samplers/nuts.js:177](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/nuts.js#L177)

Build tree recursively (doubling procedure)

###### Parameters

###### position

`Object`

Starting position

###### momentum

`Object`

Starting momentum

###### slice

`number`

Slice variable for acceptance

###### direction

`number`

Direction (+1 forward, -1 backward)

###### depth

`number`

Current tree depth

###### stepSize

`number`

Step size

###### model

`Model`

The probabilistic model

###### H0

`number`

Initial Hamiltonian

###### Returns

`Object`

Tree information

##### sample()

```ts
sample(
   model, 
   initialValues, 
   nSamples?, 
   nWarmup?, 
   thin?): Object;
```

Defined in: [samplers/nuts.js:278](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/nuts.js#L278)

Run NUTS sampling.

The sampling controls may be passed positionally or as a single options
object. When an options object is supplied as the third argument, the
`nWarmup` and `thin` positional arguments are ignored in favour of the
object's fields.

###### Parameters

###### model

`Model`

The probabilistic model

###### initialValues

`Object`

Initial parameter values

###### nSamples?

`number` \| `Object`

Number of samples, or an options object

`number`

***

`Object`

###### nWarmup?

`number` = `500`

Number of warmup samples for step-size adaptation (positional form)

###### thin?

`number` = `1`

Thinning interval (positional form)

###### Returns

`Object`

Trace object with samples and diagnostics

###### Examples

```ts
nuts.sample(model, { mu: 0 }, 1000, 500, 1)
```

```ts
nuts.sample(model, { mu: 0 }, { nSamples: 1000, nWarmup: 500, thin: 1 })
```

## Functions

### summary()

```ts
function summary(chainsOrResults, opts?): Object[];
```

Defined in: [samplers/hmc-vector.js:224](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/samplers/hmc-vector.js#L224)

ArviZ-style posterior summary across one or more chains.

#### Parameters

##### chainsOrResults

`Object` \| `any`[]

Array of chain results (`{trace}` from
  [HMC#sample](#sample)), an array of raw trace dicts, or a single trace dict.

##### opts?

###### hdi?

`number` = `0.94`

HDI mass (e.g. 0.94 → hdi_3%/hdi_97%).

#### Returns

`Object`[]

One row per scalar parameter component with
  `{ param, mean, sd, hdi_lo, hdi_hi, ess, rhat }`.

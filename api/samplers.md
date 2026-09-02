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

Defined in: [samplers/hmc-vector.js:24](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/hmc-vector.js#L24)

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

Defined in: [samplers/hmc-vector.js:34](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/hmc-vector.js#L34)

Create a vector-aware HMC sampler.

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

Defined in: [samplers/hmc-vector.js:35](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/hmc-vector.js#L35)

##### nSteps

```ts
nSteps: number;
```

Defined in: [samplers/hmc-vector.js:36](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/hmc-vector.js#L36)

##### targetAccept

```ts
targetAccept: number;
```

Defined in: [samplers/hmc-vector.js:37](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/hmc-vector.js#L37)

##### adapt

```ts
adapt: boolean;
```

Defined in: [samplers/hmc-vector.js:38](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/hmc-vector.js#L38)

##### seed

```ts
seed: number | undefined;
```

Defined in: [samplers/hmc-vector.js:39](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/hmc-vector.js#L39)

#### Methods

##### getParams()

```ts
getParams(): object;
```

Defined in: [samplers/hmc-vector.js:56](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/hmc-vector.js#L56)

The constructor options, so a worker can rebuild this sampler.

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

###### targetAccept

```ts
targetAccept: number;
```

###### adapt

```ts
adapt: boolean;
```

##### sample()

```ts
sample(
   userModel, 
   userInitialValues, 
   options?): 
  | Promise<Object>
  | {
  trace: {
  };
  acceptanceRate: number;
  stepSize: number;
  divergences: number;
  specs: any;
};
```

Defined in: [samplers/hmc-vector.js:60](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/hmc-vector.js#L60)

###### Parameters

###### userModel

`any`

###### userInitialValues

`any`

###### options?

###### Returns

  \| `Promise`\<`Object`\>
  \| \{
  `trace`: \{
  \};
  `acceptanceRate`: `number`;
  `stepSize`: `number`;
  `divergences`: `number`;
  `specs`: `any`;
\}

##### sampleChains()

```ts
sampleChains(
   model, 
   initial, 
   opts?): any[];
```

Defined in: [samplers/hmc-vector.js:217](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/hmc-vector.js#L217)

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

Defined in: [samplers/hmc.js:26](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/hmc.js#L26)

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

Defined in: [samplers/hmc.js:39](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/hmc.js#L39)

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

Defined in: [samplers/hmc.js:45](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/hmc.js#L45)

##### nSteps

```ts
nSteps: number | undefined;
```

Defined in: [samplers/hmc.js:46](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/hmc.js#L46)

#### Methods

##### getParams()

```ts
getParams(): object;
```

Defined in: [samplers/hmc.js:53](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/hmc.js#L53)

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

Defined in: [samplers/hmc.js:64](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/hmc.js#L64)

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

Defined in: [samplers/hmc.js:115](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/hmc.js#L115)

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
   userModel, 
   userInitialValues, 
   nSamples?, 
   burnIn?, 
   thin?): Object;
```

Defined in: [samplers/hmc.js:143](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/hmc.js#L143)

Run HMC sampling.

The sampling controls may be passed positionally or as a single options
object. When an options object is supplied as the third argument, the
`burnIn` and `thin` positional arguments are ignored in favour of the
object's fields.

###### Parameters

###### userModel

`any`

###### userInitialValues

`any`

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

Defined in: [samplers/metropolis.js:22](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/metropolis.js#L22)

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

Defined in: [samplers/metropolis.js:34](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/metropolis.js#L34)

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

Defined in: [samplers/metropolis.js:38](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/metropolis.js#L38)

#### Methods

##### getParams()

```ts
getParams(): object;
```

Defined in: [samplers/metropolis.js:45](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/metropolis.js#L45)

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

Defined in: [samplers/metropolis.js:73](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/metropolis.js#L73)

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

Defined in: [samplers/metropolis.js:157](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/metropolis.js#L157)

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

Defined in: [samplers/nuts.js:30](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/nuts.js#L30)

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

Defined in: [samplers/nuts.js:44](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/nuts.js#L44)

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

Defined in: [samplers/nuts.js:51](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/nuts.js#L51)

##### maxTreeDepth

```ts
maxTreeDepth: number | undefined;
```

Defined in: [samplers/nuts.js:52](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/nuts.js#L52)

##### targetAcceptance

```ts
targetAcceptance: number | undefined;
```

Defined in: [samplers/nuts.js:53](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/nuts.js#L53)

##### mu

```ts
mu: number;
```

Defined in: [samplers/nuts.js:56](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/nuts.js#L56)

##### gamma

```ts
gamma: number;
```

Defined in: [samplers/nuts.js:57](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/nuts.js#L57)

##### t0

```ts
t0: number;
```

Defined in: [samplers/nuts.js:58](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/nuts.js#L58)

##### kappa

```ts
kappa: number;
```

Defined in: [samplers/nuts.js:59](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/nuts.js#L59)

#### Methods

##### getParams()

```ts
getParams(): object;
```

Defined in: [samplers/nuts.js:66](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/nuts.js#L66)

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

Defined in: [samplers/nuts.js:82](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/nuts.js#L82)

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

##### leapfrogStep()

```ts
leapfrogStep(
   position, 
   momentum, 
   startGrad, 
   stepSize, 
   model): object;
```

Defined in: [samplers/nuts.js:135](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/nuts.js#L135)

Single leapfrog step that REUSES the start gradient and computes the
endpoint gradient and log-probability in one combined pass.

The start-of-step gradient is the previous step's endpoint gradient, so
threading it along the trajectory avoids recomputing `gradOf(position)`
that the previous step already produced. The endpoint's potential value is
needed for the Hamiltonian anyway, so `logProbAndGradient` fetches value and
gradient together instead of a separate gradient pass plus a `logProb` pass.

###### Parameters

###### position

`Object`

Current position (parameters)

###### momentum

`Object`

Current momentum

###### startGrad

`Object`

Gradient of the log-posterior at `position`

###### stepSize

`number`

Signed step size for this step

###### model

`Model`

The probabilistic model

###### Returns

`object`

New position/momentum, the endpoint gradient (to thread onward), and the
  endpoint log-probability.

###### position

```ts
position: Object;
```

###### momentum

```ts
momentum: Object;
```

###### grad

```ts
grad: Object;
```

###### logProb

```ts
logProb: number;
```

##### hamiltonian()

```ts
hamiltonian(
   position, 
   momentum, 
   model): number;
```

Defined in: [samplers/nuts.js:170](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/nuts.js#L170)

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

Defined in: [samplers/nuts.js:182](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/nuts.js#L182)

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
   logSlice, 
   direction, 
   depth, 
   stepSize, 
   model, 
   H0, 
   startGrad?): Object;
```

Defined in: [samplers/nuts.js:214](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/nuts.js#L214)

Build tree recursively (doubling procedure)

###### Parameters

###### position

`Object`

Starting position

###### momentum

`Object`

Starting momentum

###### logSlice

`number`

LOG slice variable log(u) for the membership test
  (see [NUTS#sample](#sample-3)); a state is in the slice iff `logSlice ≤ -H`

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

###### startGrad?

`Object`

Gradient of the log-posterior at `position`
  (the previous step's endpoint gradient). Computed on demand when omitted.

###### Returns

`Object`

Tree information (also carries `gradMinus`/`gradPlus`, the
  endpoint gradients, so the caller can thread them onward)

##### sample()

```ts
sample(
   userModel, 
   userInitialValues, 
   nSamples?, 
   nWarmup?, 
   thin?): 
  | Promise<Object>
  | {
  trace: Object;
  acceptanceRate: number;
  nSamples: number;
  stepSize: number | Object;
};
```

Defined in: [samplers/nuts.js:344](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/nuts.js#L344)

`sample(model, init, { chains: 4, ... })` runs four chains and returns a
Promise of `{ trace, byChain, chains, acceptanceRates, seeds, parallel,
parallelReason }`, pooled and per chain. With one chain, or the positional
form, it returns the trace synchronously as before.

###### Parameters

###### userModel

`any`

###### userInitialValues

`any`

###### nSamples?

`number` = `1000`

###### nWarmup?

`number` = `500`

###### thin?

`number` = `1`

###### Returns

  \| `Promise`\<`Object`\>
  \| \{
  `trace`: `Object`;
  `acceptanceRate`: `number`;
  `nSamples`: `number`;
  `stepSize`: `number` \| `Object`;
\}

## Functions

### summary()

```ts
function summary(chainsOrResults, opts?): Object[];
```

Defined in: [samplers/hmc-vector.js:238](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/samplers/hmc-vector.js#L238)

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

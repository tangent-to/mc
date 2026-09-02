# Design note: mc 0.10

*Observed random variables, a serializable tape, and a `sample()` that chooses its own thread.*

Status: proposal. Nothing here is implemented. The note records what a user
writes today, what they should write, the three JavaScript constraints that
shape the answer, and the order in which to build it.

## 1. The gap

A PyMC user writes the mean, names the observation distribution, hands over
the data, and samples:

```python
mu = mu0 + tau * z[site] + gamma[cycle] + qp(N, gN, nsN) + qp(K, gK, nsK)
pm.Normal("y", mu=mu, sigma=sigma, observed=Y)
trace = pm.sample(chains=4)
```

The same model in mc 0.9, taken from a real notebook:

```js
quadModel.addVariable("logTau", new mc.distributions.Normal(Math.log(0.4), 1));
// ... eight more priors ...
quadModel.autoPotential("likelihood", (v) => {
  const tau = exp(v.logTau), sig = exp(v.qLogSig);
  const mu = add(v.mu0, mul(tau, matmul(siteOneHot, v.z)), matmul(cycOneHot, v.cyc),
                 qpExpr(Narr, v.logGN, v.logNstarN), qpExpr(Karr, v.logGK, v.logNstarK));
  const r = div(sub(Yarr, mu), sig);
  return add(
    mul(-0.5, sum(square(r))),                          // Gaussian kernel
    mul(-Yarr.length, log(sig)),                        // -n log sigma
    -0.5 * Yarr.length * LOG2PI,                        // -(n/2) log 2pi
    mul(-0.5 / (TAU_SCALE * TAU_SCALE), square(tau)),   // half-Normal prior on tau
    v.logTau,                                           // Jacobian of tau = exp(logTau)
  );
});
for (const init of inits) nuts.sample(quadModel, init, opts);   // one after another
```

Three things are being written by hand that PyMC derives:

- **The log-density of the observation model.** The kernel, `-n log sigma`,
  the normalizing constant. mc's distributions compute `logProb` on plain
  numbers only, so an observation whose mean depends on the parameters has
  nowhere to go except a hand-written expression.
- **The prior on a constrained parameter and its Jacobian.** `tau` is
  half-Normal, but the notebook declares `logTau ~ Normal` and adds the
  half-Normal density and the `+logTau` Jacobian to the likelihood term
  itself. mc already samples in an unconstrained parameterization with the
  Jacobian applied (`unconstrainedView`); the notebook predates trusting it.
- **The loop over chains.** Four independent chains, run in series.

None of these are JavaScript's fault. They are gaps in mc.

## 2. What the user should write

```js
const m = new mc.Model("quadratic-plateau");
m.addVariable("mu0",   new Normal(5, 3));
m.addVariable("tau",   new HalfNormal(0.5));          // natural scale; mc transforms
m.addVariable("z",     new Normal(zeros(nSite), 1));       // vectorized, as today
m.addVariable("gN",    new Lognormal(0, 1));
m.addVariable("nsN",   new Lognormal(Math.log(2), 0.8));
m.addVariable("gK",    new Lognormal(0, 1));
m.addVariable("nsK",   new Lognormal(Math.log(2), 0.8));
m.addVariable("cyc",   new Normal(zeros(nCyc), 2));
m.addVariable("sigma", new Lognormal(0, 1));         // the notebook's qLogSig ~ Normal(0, 1)

const qp = (x, g, ns) => mul(g, sub(1, square(relu(sub(1, div(x, ns))))));

m.observe("y", (v) => new Normal(
  add(v.mu0, mul(v.tau, matmul(siteOneHot, v.z)), matmul(cycOneHot, v.cyc),
      qp(Narr, v.gN, v.nsN), qp(Karr, v.gK, v.nsK)),
  v.sigma,
), Yarr);

const fit = await new NUTS().sample(m, init, { chains: 4, nSamples: 400, nWarmup: 400 });
```

No log-density. No Jacobian. No loop. The only trace of JavaScript is that
`add(a, b, c)` is a call rather than `a + b + c`.

## 3. The three constraints, and which ones bind

**No operator overloading.** There is no way to define `+` on an object. A
TC39 proposal existed and is inactive. This is the one constraint that cannot
be designed around; `add(a, b, c, d, e)` taking every term at once is the
closest the language allows, and it is already in grad 0.1.4.

**No fork.** A Python child process inherits the parent's memory, so PyMC can
hand a worker the model as is. A JavaScript worker starts with an empty module
graph. This is what forced mc 0.9's `sampleChains` to take a *factory* whose
source is re-evaluated inside the worker, with every array the likelihood
touches threaded through a `data` object. That API is correct and it is
unreadable. Section 4.2 removes the need for it.

**The browser's single thread.** Fourteen seconds of synchronous sampling
freeze a notebook's tab. The cache in the guava notebook hides this after the
first run; nothing hides it on the first run. Workers are the only fix, which
makes 4.2 matter more than its speedup suggests.

There is a fourth constraint that is ours rather than the language's, and it
has bitten twice now. Two copies of the same numeric library in one page do not
interoperate: `CONSIDERATIONS.md` records this as a reason TensorFlow.js was
removed, and it recurred this year with grad. A notebook pinning
`@tangent.to/grad@0.1.2` beside `@tangent.to/mc@0.9.2` loaded grad twice as
soon as mc's own range resolved to 0.1.3; the two copies had different `Var`
classes and `autoPotential` rejected the expression. `mc.ops` (0.9.5) closes
the immediate hole. The design below closes it structurally, because the user
no longer imports grad at all.

## 4. Design

### 4.1 Distributions accept expressions

Every distribution's log-density is written once, in grad's ops, and accepts a
number, an array, or a `Var` for each parameter. grad's `toVar` already
coerces numbers and arrays, so one implementation serves both the prior path
(numeric parameters, numeric value) and the observation path (expression
parameters, numeric value).

```js
// normal.js, sketch
logDensity(x) {
  const z = div(sub(x, this.mu), this.sigma);
  return add(mul(-0.5, sum(square(z))), mul(-size(x), log(this.sigma)), -0.5 * size(x) * LOG2PI);
}
```

Consequences:

- **Priors lose their `dlogpdf` special case.** Their gradient comes from the
  tape like everything else, exact to the same precision. The analytic
  gradients from proba stay available for `.logProb()` on plain numbers, which
  is still the fast path for Metropolis and for anything that never needs a
  gradient.
- **The seven built-in distributions are the scope.** A user-defined
  `Distribution` subclass keeps working on the numeric path and is simply not
  differentiable, exactly as today.

### 4.2 `observe(name, factory, data)`

```js
model.observe(name, (v) => Distribution, observed)
```

`factory` receives the free variables as `Var`s and returns a distribution
whose parameters are expressions. mc evaluates `dist.logDensity(observed)`,
which is a grad expression, and registers it as a compiled potential. That is
all `autoPotential` does today; `observe` is `autoPotential` with the density
supplied by the distribution instead of by the user.

`autoPotential` and `potential` stay. `observe` is the common case, not the
only one: a custom likelihood, a censored term, a Jacobian for a transform mc
does not know about, all still go through `autoPotential`.

### 4.3 The tape is the serialization format

This is the part that makes worker chains free.

`compile(f)` in grad 0.1.3 builds the graph once and replays it. Once built,
the graph is a static structure: an ordered list of nodes, each with an op, its
parents, and any static arguments (a `reshape` shape, a `slice` start and size,
a `pow` exponent). Its leaves are either parameters, identified by name, or
constants, which hold the data the closure captured. **Nothing in that
structure is a closure.** It is plain data, and structured clone can carry it
to a worker.

grad 0.2 adds one field per node, recorded at construction: the op name and
its static arguments. Then:

```js
const plan = compile(f);       // as today
plan.toJSON()                  // { nodes: [{op, parents, args}], leaves: {...}, constants: [...] }
Plan.fromJSON(json)            // rebuilds by calling ops in order, then compiles
```

Every op already knows its name (`binary('add', ...)`); the static arguments
are the only new bookkeeping, and only `reshape`, `slice`, `pow` and
`triangularSolve` (its `lower` flag) have any.

A worker then receives the *model*, not a factory: variable names, each prior
as `{ kind, params }`, each observation term as a serialized plan. It never
sees the user's closures because the closures have already done their work on
the calling thread. The user writes the model exactly as in section 2, with
`siteOneHot` and `Narr` captured from the enclosing scope as they always were.

The data is copied to each worker as constant leaves, which is the same cost
the current `data` object pays. At the sizes MCMC runs on this is not a
concern; at the sizes it would be, the chain itself would take hours.

### 4.4 `sample()` chooses the thread

```js
sampler.sample(model, init, { chains: 4 })
```

The decision procedure, in order:

1. `chains` is absent or `1`: the calling thread, as today. Nothing changes
   for existing code.
2. `parallel: false` was passed: the chains run in series on the calling
   thread, from the same per-chain seeds a worker run would use, so the draws
   are identical to what workers would produce. This is the existing
   `sampleChains` fallback and its byte-for-byte guarantee carries over.
3. Otherwise, mc asks two questions of the runtime and one of the model:
   - Does a `Worker` constructor exist (browser, Deno), or can
     `node:worker_threads` be imported? This is the check `sampleChains`
     makes today.
   - Is every term serializable? Every prior is a built-in distribution and
     every likelihood term is a compiled plan. A `potential` written over plain
     numbers is not, and a user-defined distribution is not.
   - If both hold, the chains go to workers. If either fails, they run in
     series, and mc says so **once**, naming the term that prevented it:
     `sample: running 4 chains in series; potential "extra" is a plain
     function and cannot be sent to a worker`.

The user therefore never chooses. They ask for four chains and get them on
whatever the runtime and the model allow, with the same draws either way.

`sampleChains` with an explicit factory stays exported, for the case where a
model has a numeric `potential` and the author would still rather write the
factory than lose the workers.

### 4.5 Constrained parameters on their natural scale

`unconstrainedView` already transforms a `HalfNormal`, `Gamma`, `Lognormal`,
`Beta` or `Uniform` variable to an unconstrained parameterization and applies
the log-Jacobian. Only NUTS goes through it today; `HMC` and the vector
sampler do not. In 0.10 every gradient-based sampler does, so that declaring
`tau ~ HalfNormal(0.5)` is safe everywhere and the notebook's hand-written
`+logTau` disappears.

## 5. Robustness to JavaScript's evolution

The design keeps the user's API at PyMC's level and keeps every
JavaScript-specific mechanism internal. That is what insulates it: if the
language gains a capability, the mechanism changes and the API does not.

**Operator overloading.** If a proposal ever lands, `Var` gains the operator
methods and `a + b` becomes sugar for `add(a, b)`. Nothing existing breaks;
the variadic form and the operator would coexist as they do in numpy, where
`np.add(a, b, c)` is not how anyone writes it but still works.

**Shared memory or fork.** There is no proposal for fork. `SharedArrayBuffer`
exists now, behind cross-origin isolation headers. If a runtime let workers
share the parent's memory, the serialized plan would be transferred without
copying instead of cloned, and the decision procedure in 4.4 would not change.
The plan is a data structure; how it reaches the worker is a detail below it.

**`f64` on the GPU.** WebGPU has no double precision today, which is why grad
has no GPU backend. If it gained one, the compiled plan is already the shape a
kernel compiler consumes: a static list of ops over fixed-shape buffers. That
would be a new backend for `Plan`, not a change to what a model looks like.

The one thing this does *not* insulate against is the notebook platform.
`note.tangent.to` resolves modules and creates workers its own way; a change
there is a change to that platform, not to JavaScript, and mc can only detect
it (4.4, step 3) and fall back.

## 6. Breaking points

- `Distribution.logProb` stays numeric and unchanged. `logDensity` is new.
  Existing code is unaffected.
- `addVariable(name, dist, observed)` keeps its meaning for a distribution
  with fixed parameters. `observe` is for computed parameters. The two could
  be unified later; not now.
- Prior gradients switch from proba's `dlogpdf` to the tape. Measured cost is
  negligible at prior sizes, but it is a change in the code path and the
  PyMC parity tests must pass unchanged.
- `sampleChains` keeps its signature. `sample` gains `chains` and `parallel`.
- grad 0.2's `node()` gains a fifth argument. Every op site changes by one
  line. Hand-built `Var`s without a spec are not serializable and fall back,
  as they already fall back from `compile`.

## 7. Out of scope

- Automatic discovery of what a closure captures. Not possible in JavaScript
  and not needed: the tape captures it by construction.
- Serializing a numeric `potential`. Its function is opaque. The user either
  writes it as an expression or gives up workers for that model.
- Operator overloading. See section 3.
- Vectorized shapes beyond rank 2. Unchanged from grad's scope.

## 8. Order of work

Each step is shippable on its own and measured before the next.

1. **grad 0.2, serializable plan.** Add the op spec to `node()`, `toJSON` /
   `fromJSON`, and a test that a plan round-trips and replays to bit-identical
   gradients across every op. Small; a day.
2. **mc: `logDensity` on the seven distributions.** Written in grad's ops,
   checked against `logProb` to 1e-12 on numeric inputs and against finite
   differences on `Var` inputs. Priors switch to the tape; PyMC parity tests
   must not move.
3. **mc: `observe`.** Thin over `autoPotential`. The guava §2.4 model
   rewritten as in section 2 becomes the acceptance test: same posterior as
   the hand-written density, to Monte Carlo error, and it is the version that
   goes into the notebook.
4. **mc: `sample(model, init, { chains })`.** Model serialization, the
   decision procedure, the one-line fallback message. Measured on §2.4: the
   target is the 5.7 s the factory path already reaches, with no factory.
5. **mc: every gradient sampler through `unconstrainedView`.**

Steps 1 and 2 are independent and can proceed in parallel. Step 4 needs 1.
One release, 0.10.0, when all five are in, rather than a patch per step.

## 9. Open questions

- Should `observe` accept a distribution *instance* whose parameters are
  `Var`s, `new Normal(muExpr, sigma)`, rather than a factory? It reads better
  and is what section 2 shows. It requires the distribution constructor to
  tolerate `Var` parameters without evaluating anything, which 4.1 gives it.
  Leaning yes.
- Should the fallback message be a warning or a returned field? A notebook
  reader sees `console.warn`; a script may prefer `fit.parallel === false`
  plus `fit.parallelReason`. Probably both.
- A `{ shape }` option on `addVariable` would read better than a vector of
  zeros for the location parameter, and is what PyMC does. It collides with
  the existing third positional argument, `observed`. Worth doing in the same
  release, as an options-object form; not required for any of the above.
- Does `deterministic` need to travel to workers? It runs post hoc on the
  trace, so no, but it should be evaluated on the calling thread after the
  chains return, which is a small change from today.

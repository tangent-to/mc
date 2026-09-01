/**
 * Constrained-parameter transforms.
 *
 * A gradient sampler moves through ℝⁿ. A parameter with a bounded support —
 * a scale in (0, ∞), a probability in (0, 1) — has no business being stepped
 * through directly: leapfrog will walk it past its boundary, where the density
 * is -Infinity and the gradient is meaningless. Rejecting those proposals
 * "works" but wastes the trajectory and biases the step-size adaptation toward
 * tiny steps near the boundary; it is also how a single non-finite value used
 * to poison an entire run (see tests/nuts-divergence.test.js).
 *
 * Stan and PyMC both solve this the same way, and so does this module: sample
 * an UNCONSTRAINED parameter u, map it to the constrained x the model is
 * written in, and add log|dx/du| to the log-density so the change of variables
 * leaves the posterior invariant. The sampler then cannot propose an invalid
 * value at all, and every proposal is a live one.
 *
 *   (a, ∞)   x = a + eᵘ                  log|dx/du| = u
 *   (-∞, b)  x = b − eᵘ                  log|dx/du| = u
 *   (a, b)   x = a + (b−a)·σ(u)          log|dx/du| = log(b−a) + log σ(u) + log(1−σ(u))
 *   ℝ        x = u                       log|dx/du| = 0
 */

/** @typedef {{lower: number|null, upper: number|null}} Support */

const sigmoid = (u) => 1 / (1 + Math.exp(-u));

/** Numerically safe log σ(u): avoids underflow for very negative u. @private */
function logSigmoid(u) {
  return u < 0 ? u - Math.log1p(Math.exp(u)) : -Math.log1p(Math.exp(-u));
}

/**
 * Build the transform for one support.
 *
 * @param {Support} support - `{lower, upper}`; null means unbounded that side
 * @returns {{
 *   toUnconstrained: (x: number) => number,
 *   toConstrained: (u: number) => number,
 *   logDetJacobian: (u: number) => number,
 *   dLogDetJacobian: (u: number) => number,
 *   dxdu: (u: number) => number,
 *   isIdentity: boolean,
 * }}
 */
export function makeTransform(support) {
  const lower = support?.lower ?? null;
  const upper = support?.upper ?? null;
  const hasLo = lower !== null && Number.isFinite(lower);
  const hasHi = upper !== null && Number.isFinite(upper);

  if (!hasLo && !hasHi) {
    return {
      toUnconstrained: (x) => x,
      toConstrained: (u) => u,
      logDetJacobian: () => 0,
      dLogDetJacobian: () => 0,
      dxdu: () => 1,
      isIdentity: true,
    };
  }

  if (hasLo && !hasHi) {
    return {
      toUnconstrained: (x) => Math.log(x - lower),
      toConstrained: (u) => lower + Math.exp(u),
      logDetJacobian: (u) => u,
      dLogDetJacobian: () => 1, // d/du of u
      dxdu: (u) => Math.exp(u),
      isIdentity: false,
    };
  }

  if (!hasLo && hasHi) {
    return {
      toUnconstrained: (x) => Math.log(upper - x),
      toConstrained: (u) => upper - Math.exp(u),
      logDetJacobian: (u) => u,
      dLogDetJacobian: () => 1, // d/du of u
      dxdu: (u) => -Math.exp(u),
      isIdentity: false,
    };
  }

  const width = upper - lower;
  return {
    toUnconstrained: (x) => {
      const p = (x - lower) / width;
      return Math.log(p) - Math.log1p(-p);
    },
    toConstrained: (u) => lower + width * sigmoid(u),
    logDetJacobian: (u) => Math.log(width) + logSigmoid(u) + logSigmoid(-u),
    // d/du [log σ(u) + log σ(-u)] = (1 - σ(u)) - σ(u) = 1 - 2σ(u)
    dLogDetJacobian: (u) => 1 - 2 * sigmoid(u),
    dxdu: (u) => {
      const s = sigmoid(u);
      return width * s * (1 - s);
    },
    isIdentity: false,
  };
}

/**
 * The support of an mc distribution, read from its type and parameters.
 *
 * Kept here rather than on each class so the mapping is visible in one place;
 * a distribution that does not declare one is treated as unbounded, which is
 * the safe default (the sampler then behaves exactly as it did before).
 *
 * @param {Object} distribution
 * @returns {Support}
 */
export function supportOf(distribution) {
  if (!distribution) return { lower: null, upper: null };
  if (typeof distribution.support === 'function') return distribution.support();
  switch (distribution.constructor?.name) {
    case 'HalfNormal':
    case 'Gamma':
    case 'Lognormal':
      return { lower: 0, upper: null };
    case 'Beta':
      return { lower: 0, upper: 1 };
    case 'Uniform': {
      // Array-valued bounds would need a per-element support; those stay
      // untransformed rather than silently using one element's bounds.
      const { lower, upper } = distribution;
      if (typeof lower === 'number' && typeof upper === 'number') return { lower, upper };
      return { lower: null, upper: null };
    }
    default:
      // Normal, Bernoulli (discrete, never stepped by a gradient sampler),
      // and anything unrecognised.
      return { lower: null, upper: null };
  }
}

/**
 * Present a model through its unconstrained parameterization, under the same
 * method names a sampler already calls.
 *
 * Wrapping rather than editing every call site keeps the samplers unaware that
 * a transform exists: they move through ℝⁿ, and the view maps back on the way
 * in. Only two things outside this need to know — the initial values, which
 * must be transformed once, and the recorded draws, which must be transformed
 * back.
 *
 * A model with no bounded variable is returned untouched, so nothing changes
 * for the unconstrained case, not even an extra object hop.
 *
 * @param {Object} model
 * @returns {Object} the model, or a view of it in unconstrained space
 */
export function unconstrainedView(model) {
  if (typeof model.hasConstrainedVariables !== 'function' || !model.hasConstrainedVariables()) {
    return model;
  }
  // Inherit from the model itself, so every other thing a sampler reaches for
  // — getFreeVariableNames, variables, deterministics — resolves through the
  // prototype chain untouched. Only the three evaluation entry points are
  // overridden; listing the API by hand would break the next time a sampler
  // calls something new.
  return Object.create(model, {
    __unconstrained: { value: true },
    logProb: {
      value: (u) => model.logProbAndGradientUnconstrained(u).logProb,
    },
    logProbAndGradient: {
      value: (u) => model.logProbAndGradientUnconstrained(u),
    },
    gradientsOnly: {
      value: (u) => model.gradientsOnlyUnconstrained(u),
    },
  });
}

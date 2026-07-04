/**
 * Package-level random number generation, backed by @tangent.to/proba's
 * seedable xoshiro128** generator. All samplers and distribution sampling
 * draw from this single stream so that `setRandomSeed(seed)` makes an
 * entire mc run reproducible.
 */

import { createRng } from '@tangent.to/proba';

let rng = createRng();

/**
 * Seed the package RNG for reproducible sampling.
 * @param {number} seed - Any finite number
 */
export function setRandomSeed(seed) {
  rng = createRng(seed);
}

/**
 * Get the current package RNG ({float, int, normal}).
 * @returns {Object}
 */
export function getRng() {
  return rng;
}

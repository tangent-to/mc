import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { rhat, ess } from '../src/utils/trace.js';
import * as mc from '../src/index.js';

// Reference values computed by ArviZ 0.20 (tests/generate_arviz_fixtures.py) on the same chains.
const fixtures = JSON.parse(readFileSync(new URL('./fixtures/arviz-diagnostics.json', import.meta.url)));

describe('rank-normalized diagnostics match ArviZ', () => {
  for (const [name, f] of Object.entries(fixtures)) {
    it(`${name}: R-hat, bulk ESS, tail ESS`, () => {
      expect(rhat(f.chains)).toBeCloseTo(f.rhat, 6);
      expect(ess(f.chains)).toBeCloseTo(f.ess_bulk, 3);
      expect(ess(f.chains, { kind: 'tail' })).toBeCloseTo(f.ess_tail, 3);
    });
  }
});

describe('rank-normalized diagnostics behave as diagnostics', () => {
  it('flags a shifted chain and a chain with a different scale', () => {
    expect(rhat(fixtures.iid.chains)).toBeLessThan(1.01);
    expect(rhat(fixtures.shifted.chains)).toBeGreaterThan(1.1);
    // The folded R-hat is what catches a scale mismatch at equal location.
    expect(rhat(fixtures.scale.chains)).toBeGreaterThan(1.1);
    expect(ess(fixtures.scale.chains, { kind: 'tail' })).toBeLessThan(100);
  });

  it('returns NaN when the input is too short or not finite', () => {
    expect(rhat([[1, 2, 3, 4, 5]])).toBeNaN();           // one chain
    expect(rhat([[1, 2, 3], [4, 5, 6]])).toBeNaN();       // fewer than 4 draws
    expect(ess([[1, 2, NaN, 4, 5]])).toBeNaN();
  });

  it('rejects an unknown ESS kind', () => {
    expect(() => ess(fixtures.iid.chains, { kind: 'mean' })).toThrow(/unknown kind/);
  });

  it('is exported flat and in the diagnostics namespace', () => {
    expect(mc.rhat).toBe(rhat);
    expect(mc.diagnostics.ess).toBe(ess);
    expect(mc.default.diagnostics.rhat).toBe(rhat);
  });
});

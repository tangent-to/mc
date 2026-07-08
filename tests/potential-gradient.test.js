/**
 * Analytic gradients for potentials: `model.potential(name, fn, gradFn)`.
 * The supplied gradient must match the finite-difference fallback, compose with
 * analytic prior gradients, support vector-valued variables, and mix with
 * potentials that have no analytic gradient.
 */

import { describe, it, expect } from 'vitest';
import { Model } from '../src/model.js';
import { Normal } from '../src/distributions/normal.js';

const maxAbsDiff = (a, b) => {
  let m = 0;
  for (const k of Object.keys(a)) {
    const va = a[k], vb = b[k];
    if (Array.isArray(va)) for (let i = 0; i < va.length; i++) m = Math.max(m, Math.abs(va[i] - vb[i]));
    else m = Math.max(m, Math.abs(va - vb));
  }
  return m;
};

describe('potential(name, fn, gradFn) — scalar params', () => {
  // Quadratic mean model with a per-site intercept, exactly like the guava notebook.
  const n = 60;
  const N = [], K = [], S = [], Y = [];
  for (let i = 0; i < n; i++) {
    const s = i % 2, nn = (i % 7) / 3, kk = (i % 5) / 3;
    S.push(s); N.push(nn); K.push(kk);
    Y.push((s ? 5.5 : 5) + 1.2 * nn - 0.25 * nn * nn + 0.4 * kk - 0.08 * kk * kk);
  }
  const like = (p) => {
    const sig = Math.exp(p.qLogSig);
    const mu = S.map((s, i) => (s ? p.b0_1 : p.b0_0) + p.bN * N[i] + p.bNN * N[i] * N[i] + p.bK * K[i] + p.bKK * K[i] * K[i]);
    return new Normal(mu, sig).logProb(Y);
  };
  const grad = (p) => {
    const sig = Math.exp(p.qLogSig), inv2 = 1 / (sig * sig);
    let g00 = 0, g01 = 0, gN = 0, gNN = 0, gK = 0, gKK = 0, gs = 0;
    for (let i = 0; i < n; i++) {
      const s = S[i];
      const mu = (s ? p.b0_1 : p.b0_0) + p.bN * N[i] + p.bNN * N[i] * N[i] + p.bK * K[i] + p.bKK * K[i] * K[i];
      const z = Y[i] - mu, r = z * inv2;
      if (s) g01 += r; else g00 += r;
      gN += r * N[i]; gNN += r * N[i] * N[i]; gK += r * K[i]; gKK += r * K[i] * K[i];
      gs += -1 + z * z * inv2;
    }
    return { b0_0: g00, b0_1: g01, bN: gN, bNN: gNN, bK: gK, bKK: gKK, qLogSig: gs };
  };
  const addVars = (m) => {
    m.addVariable('b0_0', new Normal(5, 3)); m.addVariable('b0_1', new Normal(5, 3));
    m.addVariable('bN', new Normal(0, 3)); m.addVariable('bNN', new Normal(0, 3));
    m.addVariable('bK', new Normal(0, 3)); m.addVariable('bKK', new Normal(0, 3));
    m.addVariable('qLogSig', new Normal(0, 1));
  };
  const p = { b0_0: 5, b0_1: 5.5, bN: 1, bNN: -0.2, bK: 0.3, bKK: -0.05, qLogSig: Math.log(0.3) };

  it('matches the finite-difference gradient (incl. prior contribution)', () => {
    const mFD = new Model('fd'); addVars(mFD); mFD.potential('like', like);
    const mAN = new Model('an'); addVars(mAN); mAN.potential('like', like, grad);
    const rFD = mFD.logProbAndGradient(p);
    const rAN = mAN.logProbAndGradient(p);
    expect(rAN.logProb).toBeCloseTo(rFD.logProb, 10);
    expect(maxAbsDiff(rAN.gradients, rFD.gradients)).toBeLessThan(1e-5);
  });

  it('produces a finite, correct gradient away from the optimum too', () => {
    const m = new Model('an'); addVars(m); m.potential('like', like, grad);
    const q = { ...p, bN: -2, bNN: 0.5, qLogSig: 0.2 };
    const g = m.logProbAndGradient(q).gradients;
    for (const k of Object.keys(g)) expect(Number.isFinite(g[k])).toBe(true);
    // finite-difference cross-check at this point
    const mFD = new Model('fd'); addVars(mFD); mFD.potential('like', like);
    expect(maxAbsDiff(g, mFD.logProbAndGradient(q).gradients)).toBeLessThan(1e-4);
  });
});

describe('potential gradient — vector-valued variable', () => {
  const n = 40, P = 4;
  const X = [], y = [];
  for (let i = 0; i < n; i++) {
    const row = Array.from({ length: P }, (_, j) => ((i * (j + 1)) % 5) / 2 - 1);
    X.push(row);
    y.push(1 + 0.5 * row[0] - 0.3 * row[1] + 0.2 * row[2]);
  }
  const like = (p) => {
    const sig = Math.exp(p.logSig);
    const mu = X.map((xi) => p.alpha + xi.reduce((s, x, j) => s + p.beta[j] * x, 0));
    return new Normal(mu, sig).logProb(y);
  };
  const grad = (p) => {
    const sig = Math.exp(p.logSig), inv2 = 1 / (sig * sig);
    let ga = 0, gl = 0; const gb = new Array(P).fill(0);
    for (let i = 0; i < n; i++) {
      const mu = p.alpha + X[i].reduce((s, x, j) => s + p.beta[j] * x, 0);
      const z = y[i] - mu, r = z * inv2;
      ga += r; for (let j = 0; j < P; j++) gb[j] += r * X[i][j];
      gl += -1 + z * z * inv2;
    }
    return { alpha: ga, beta: gb, logSig: gl };
  };
  const addVars = (m) => {
    m.addVariable('alpha', new Normal(0, 2));
    m.addVariable('beta', new Normal(0, 1)); // vector
    m.addVariable('logSig', new Normal(0, 1));
  };
  const p = { alpha: 1, beta: [0.5, -0.3, 0.2, 0], logSig: Math.log(0.4) };

  it('matches finite differences for the array parameter', () => {
    const mFD = new Model('fd'); addVars(mFD); mFD.potential('lik', like);
    const mAN = new Model('an'); addVars(mAN); mAN.potential('lik', like, grad);
    const gFD = mFD.logProbAndGradient(p).gradients;
    const gAN = mAN.logProbAndGradient(p).gradients;
    expect(Array.isArray(gAN.beta)).toBe(true);
    expect(maxAbsDiff(gAN, gFD)).toBeLessThan(1e-5);
  });
});

describe('potential gradient — mixing analytic and finite-difference terms', () => {
  it('adds an analytic term and a finite-difference term correctly', () => {
    const build = (withGrad) => {
      const m = new Model('m');
      m.addVariable('a', new Normal(0, 5));
      m.addVariable('b', new Normal(0, 5));
      // term1 depends on a (analytic gradient supplied when withGrad)
      const t1 = (p) => new Normal(p.a, 1).logProb([1.5, 2.0]);
      const g1 = (p) => ({ a: (1.5 - p.a) + (2.0 - p.a) });
      // term2 depends on b (never has an analytic gradient -> FD path)
      const t2 = (p) => new Normal(p.b, 1).logProb([-1.0]);
      m.potential('t1', t1, withGrad ? g1 : undefined);
      m.potential('t2', t2);
      return m;
    };
    const p = { a: 0.3, b: -0.4 };
    const gMixed = build(true).logProbAndGradient(p).gradients;
    const gAllFD = build(false).logProbAndGradient(p).gradients;
    expect(maxAbsDiff(gMixed, gAllFD)).toBeLessThan(1e-5);
    // b's gradient still comes through the FD branch
    expect(Number.isFinite(gMixed.b)).toBe(true);
  });

  it('potential() without gradFn keeps the finite-difference behaviour', () => {
    const m = new Model('m');
    m.addVariable('a', new Normal(0, 5));
    m.potential('t', (p) => new Normal(p.a, 1).logProb([2.0]));
    const g = m.logProbAndGradient({ a: 0.5 }).gradients;
    // d/da [ logN(2;a,1) + logN(a;0,5) ] = (2 - a) - a/25 = 1.5 - 0.02
    expect(g.a).toBeCloseTo(1.48, 6);
  });
});

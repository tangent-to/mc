// Port of _real2.mjs to the proba-backed mc (no TensorFlow.js).
// Same model, same data, plain JS math in the potential.
import { Model, distributions, samplers, setRandomSeed } from './src/index.js';
import { readFileSync } from 'node:fs';
const { Normal } = distributions; const { NUTS } = samplers;
setRandomSeed(7);
let txt = readFileSync('/home/essi/Documents/git/guava/data/data_2.0.csv', 'utf8').replace(/^﻿/, '');
const L = txt.split(/\r?\n/).filter((x) => x.length); const sep = L[0].includes(',') ? ',' : ';';
const H = L[0].split(sep); const rows = L.slice(1).map((l) => { const c = l.split(sep); const o = {}; H.forEach((h, i) => o[h] = c[i]); return o; });
const num = (v) => (v == null || v === '' ? NaN : +v);
const mits = rows.filter((r) => r.project === 'Souza H.' || r.project === 'Montes RM.')
  .map((r) => ({ N: num(r['N kg/ha']), K: num(r['K2O kg/ha']), y: num(r['yield']), site: r.project === 'Montes RM.' ? 1 : 0 }))
  .filter((d) => Number.isFinite(d.N) && Number.isFinite(d.K) && Number.isFinite(d.y));
const RS = 100, YS = 10;
const Nv = mits.map((d) => d.N / RS), Kv = mits.map((d) => d.K / RS);
const Yv = mits.map((d) => d.y / YS), Sv = mits.map((d) => d.site);
const sigmoid = (z) => 1 / (1 + Math.exp(-z));
function build() {
  const m = new Model();
  m.addVariable('logA0', new Normal(Math.log(5), 0.5)); m.addVariable('logA1', new Normal(Math.log(5), 0.5));
  m.addVariable('logcN', new Normal(0, 1)); m.addVariable('logcK', new Normal(0, 1));
  m.addVariable('logitLN0', new Normal(0, 1.5)); m.addVariable('logitLN1', new Normal(0, 1.5));
  m.addVariable('logitLK', new Normal(0, 1.5)); m.addVariable('logSig', new Normal(0, 1));
  m.potential('lik', (p) => {
    const A0 = Math.exp(p.logA0), A1 = Math.exp(p.logA1), cN = Math.exp(p.logcN), cK = Math.exp(p.logcK);
    const LN0 = sigmoid(p.logitLN0), LN1 = sigmoid(p.logitLN1), LK = sigmoid(p.logitLK), sig = Math.exp(p.logSig);
    const mu = Nv.map((n, i) => {
      const As = Sv[i] ? A1 : A0, LNs = Sv[i] ? LN1 : LN0;
      return As * (1 - LNs * Math.exp(-cN * n)) * (1 - LK * Math.exp(-cK * Kv[i]));
    });
    return new Normal(mu, sig).logProb(Yv);
  });
  return m;
}
const init = { logA0: Math.log(5), logA1: Math.log(5), logcN: 0, logcK: 0, logitLN0: 0, logitLN1: 0, logitLK: 0, logSig: 0 };
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
for (const ss of [0.05, 0.02, 0.01]) {
  const m = build(); const t0 = Date.now();
  const r = new NUTS({ stepSize: ss, maxTreeDepth: 8, targetAcceptance: 0.8 }).sample(m, init, { nSamples: 200, nWarmup: 300 });
  const tr = r.trace;
  console.log(`ss=${ss} accept=${(r.acceptanceRate * 100).toFixed(0)}% step=${(+r.stepSize).toExponential(2)} NaN=${tr.logA0.some(Number.isNaN)} ` +
    `A0=${(Math.exp(mean(tr.logA0)) * YS).toFixed(1)} A1=${(Math.exp(mean(tr.logA1)) * YS).toFixed(1)} ` +
    `cN=${(Math.exp(mean(tr.logcN)) / RS).toExponential(2)} LK=${sigmoid(mean(tr.logitLK)).toFixed(2)} sig=${(Math.exp(mean(tr.logSig)) * YS).toFixed(1)} t=${((Date.now() - t0) / 1000).toFixed(1)}s`);
}

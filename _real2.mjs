import { Model, distributions, samplers, tf } from './src/index.js';
import { readFileSync } from 'node:fs';
const { Normal } = distributions; const { NUTS } = samplers;
await tf.setBackend('cpu'); await tf.ready();
let txt=readFileSync('/home/essi/Documents/git/guava/data/data_2.0.csv','utf8').replace(/^﻿/,'');
const L=txt.split(/\r?\n/).filter(x=>x.length); const sep=L[0].includes(',')?',':';';
const H=L[0].split(sep); const rows=L.slice(1).map(l=>{const c=l.split(sep);const o={};H.forEach((h,i)=>o[h]=c[i]);return o;});
const num=v=>(v==null||v===''?NaN:+v);
let mits=rows.filter(r=>r.project==='Souza H.'||r.project==='Montes RM.')
 .map(r=>({N:num(r['N kg/ha']),K:num(r['K2O kg/ha']),y:num(r['yield']),site:r.project==='Montes RM.'?1:0}))
 .filter(d=>Number.isFinite(d.N)&&Number.isFinite(d.K)&&Number.isFinite(d.y));
const RS=100, YS=10;
const Nt=tf.tensor1d(mits.map(d=>d.N/RS)),Kt=tf.tensor1d(mits.map(d=>d.K/RS));
const Yt=tf.tensor1d(mits.map(d=>d.y/YS)),St=tf.tensor1d(mits.map(d=>d.site));
function build(){const m=new Model();
 m.addVariable('logA0',new Normal(Math.log(5),0.5));m.addVariable('logA1',new Normal(Math.log(5),0.5));
 m.addVariable('logcN',new Normal(0,1));m.addVariable('logcK',new Normal(0,1));
 m.addVariable('logitLN0',new Normal(0,1.5));m.addVariable('logitLN1',new Normal(0,1.5));
 m.addVariable('logitLK',new Normal(0,1.5));m.addVariable('logSig',new Normal(0,1));
 m.potential('lik',p=>{const A0=tf.exp(p.logA0),A1=tf.exp(p.logA1),cN=tf.exp(p.logcN),cK=tf.exp(p.logcK);
  const LN0=tf.sigmoid(p.logitLN0),LN1=tf.sigmoid(p.logitLN1),LK=tf.sigmoid(p.logitLK),sig=tf.exp(p.logSig);
  const oS=tf.sub(1,St),As=tf.add(tf.mul(St,A1),tf.mul(oS,A0)),LNs=tf.add(tf.mul(St,LN1),tf.mul(oS,LN0));
  const nf=tf.sub(1,tf.mul(LNs,tf.exp(tf.mul(tf.neg(cN),Nt)))),kf=tf.sub(1,tf.mul(LK,tf.exp(tf.mul(tf.neg(cK),Kt))));
  return new Normal(tf.mul(tf.mul(As,nf),kf),sig).logProb(Yt);});
 return m;}
const init={logA0:Math.log(5),logA1:Math.log(5),logcN:0,logcK:0,logitLN0:0,logitLN1:0,logitLK:0,logSig:0};
const mean=a=>a.reduce((x,y)=>x+y,0)/a.length;
for(const ss of [0.05,0.02,0.01]){const m=build();const t0=Date.now();
 const r=new NUTS({stepSize:ss,maxTreeDepth:8,targetAcceptance:0.8}).sample(m,init,{nSamples:200,nWarmup:300});
 const tr=r.trace;
 console.log(`ss=${ss} accept=${(r.acceptanceRate*100).toFixed(0)}% step=${(+r.stepSize).toExponential(2)} NaN=${tr.logA0.some(Number.isNaN)} `+
  `A0=${(Math.exp(mean(tr.logA0))*YS).toFixed(1)} A1=${(Math.exp(mean(tr.logA1))*YS).toFixed(1)} `+
  `cN=${(Math.exp(mean(tr.logcN))/RS).toExponential(2)} LK=${(1/(1+Math.exp(-mean(tr.logitLK)))).toFixed(2)} sig=${(Math.exp(mean(tr.logSig))*YS).toFixed(1)} t=${((Date.now()-t0)/1000).toFixed(0)}s`);}

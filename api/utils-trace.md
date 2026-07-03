---
layout: default
title: utils/trace
parent: API Reference
nav_order: 5
permalink: /api/utils-trace
---
# utils/trace

## Functions

### summarize()

```ts
function summarize(samples): Object;
```

Defined in: [utils/trace.js:10](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/utils/trace.js#L10)

Compute summary statistics for a trace

#### Parameters

##### samples

`number`[]

Array of samples

#### Returns

`Object`

Summary statistics

***

### effectiveSampleSize()

```ts
function effectiveSampleSize(samples): number;
```

Defined in: [utils/trace.js:39](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/utils/trace.js#L39)

Compute effective sample size (ESS) using autocorrelation

#### Parameters

##### samples

`number`[]

Array of samples

#### Returns

`number`

Effective sample size

***

### gelmanRubin()

```ts
function gelmanRubin(chains): number;
```

Defined in: [utils/trace.js:76](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/utils/trace.js#L76)

Compute the Gelman-Rubin diagnostic (R-hat) for convergence
Requires multiple chains

#### Parameters

##### chains

`number`[][]

Array of chains (each chain is an array of samples)

#### Returns

`number`

R-hat statistic

***

### printSummary()

```ts
function printSummary(trace): void;
```

Defined in: [utils/trace.js:113](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/utils/trace.js#L113)

Print trace summary for all variables

#### Parameters

##### trace

`Object`

Trace object from sampling

#### Returns

`void`

***

### traceToJSON()

```ts
function traceToJSON(trace): string;
```

Defined in: [utils/trace.js:138](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/utils/trace.js#L138)

Export trace to JSON format

#### Parameters

##### trace

`Object`

Trace object

#### Returns

`string`

JSON string

***

### traceToCSV()

```ts
function traceToCSV(samples): string;
```

Defined in: [utils/trace.js:147](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/utils/trace.js#L147)

Save trace to CSV format (for a single variable)

#### Parameters

##### samples

`number`[]

Array of samples

#### Returns

`string`

CSV string

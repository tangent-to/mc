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
function summarize(samples): object;
```

Defined in: [utils/trace.js:13](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/utils/trace.js#L13)

Compute summary statistics for a trace

#### Parameters

##### samples

`number`[]

Array of samples

#### Returns

`object`

Summary statistics: the
  mean, median, standard deviation, variance, 2.5%/97.5% interval bounds,
  and the sample count

##### mean

```ts
mean: number;
```

##### median

```ts
median: number;
```

##### std

```ts
std: number;
```

##### variance

```ts
variance: number;
```

##### hdi\_2\_5

```ts
hdi_2_5: number;
```

##### hdi\_97\_5

```ts
hdi_97_5: number;
```

##### n

```ts
n: number;
```

***

### effectiveSampleSize()

```ts
function effectiveSampleSize(samples): number;
```

Defined in: [utils/trace.js:42](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/utils/trace.js#L42)

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

Defined in: [utils/trace.js:79](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/utils/trace.js#L79)

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

Defined in: [utils/trace.js:116](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/utils/trace.js#L116)

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

Defined in: [utils/trace.js:141](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/utils/trace.js#L141)

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

Defined in: [utils/trace.js:150](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/utils/trace.js#L150)

Save trace to CSV format (for a single variable)

#### Parameters

##### samples

`number`[]

Array of samples

#### Returns

`string`

CSV string

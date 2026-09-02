---
layout: default
title: utils/visualize
parent: API Reference
nav_order: 6
permalink: /api/utils-visualize
---
# utils/visualize

## Functions

### tracePlot()

```ts
function tracePlot(
   trace, 
   variables?, 
   options?): Object;
```

Defined in: [utils/visualize.js:23](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/utils/visualize.js#L23)

Generate trace plot specification
Shows the sampled values over iterations to assess convergence

#### Parameters

##### trace

`Object`

MCMC trace object

##### variables?

`string`[] = `null`

Variable names to plot (null = all)

##### options?

`Object` = `{}`

Plot options

#### Returns

`Object`

Plot specification with .show() method

***

### posteriorPlot()

```ts
function posteriorPlot(
   trace, 
   variables?, 
   options?): Object;
```

Defined in: [utils/visualize.js:106](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/utils/visualize.js#L106)

Generate posterior distribution plot specification
Shows histograms and KDE of posterior samples

#### Parameters

##### trace

`Object`

MCMC trace object

##### variables?

`string`[] = `null`

Variable names to plot

##### options?

`Object` = `{}`

Plot options

#### Returns

`Object`

Plot specification with .show() method

***

### autocorrPlot()

```ts
function autocorrPlot(
   trace, 
   variables?, 
   maxLag?, 
   options?): Object;
```

Defined in: [utils/visualize.js:213](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/utils/visualize.js#L213)

Generate autocorrelation plot specification
Shows autocorrelation to assess mixing

#### Parameters

##### trace

`Object`

MCMC trace object

##### variables?

`string`[] = `null`

Variable names to plot

##### maxLag?

`number` = `50`

Maximum lag to compute

##### options?

`Object` = `{}`

Plot options

#### Returns

`Object`

Plot specification with .show() method

***

### pairPlot()

```ts
function pairPlot(
   trace, 
   variables?, 
   options?): Object;
```

Defined in: [utils/visualize.js:315](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/utils/visualize.js#L315)

Generate pair plot specification (scatter plot matrix)
Shows relationships between parameters

#### Parameters

##### trace

`Object`

MCMC trace object

##### variables?

`string`[] = `null`

Variable names to plot

##### options?

`Object` = `{}`

Plot options

#### Returns

`Object`

Plot specification with .show() method

***

### forestPlot()

```ts
function forestPlot(
   trace, 
   variables?, 
   hdi?, 
   options?): Object;
```

Defined in: [utils/visualize.js:395](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/utils/visualize.js#L395)

Generate forest plot specification
Shows posterior summaries with credible intervals

#### Parameters

##### trace

`Object`

MCMC trace object

##### variables?

`string`[] = `null`

Variable names to plot

##### hdi?

`number` = `0.95`

Highest Density Interval (default 0.95)

##### options?

`Object` = `{}`

Plot options

#### Returns

`Object`

Plot specification with .show() method

***

### rankPlot()

```ts
function rankPlot(
   trace, 
   variables?, 
   options?): Object;
```

Defined in: [utils/visualize.js:489](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/utils/visualize.js#L489)

Generate rank plot specification (for convergence diagnostics)
Useful for detecting non-stationarity and comparing chains

#### Parameters

##### trace

`Object`

MCMC trace object

##### variables?

`string`[] = `null`

Variable names to plot

##### options?

`Object` = `{}`

Plot options

#### Returns

`Object`

Plot specification with .show() method

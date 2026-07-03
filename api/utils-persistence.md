---
layout: default
title: utils/persistence
parent: API Reference
nav_order: 4
permalink: /api/utils-persistence
---
# utils/persistence

## Functions

### saveTrace()

```ts
function saveTrace(trace, filepath): void;
```

Defined in: [utils/persistence.js:8](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/utils/persistence.js#L8)

Save trace data to JSON file

#### Parameters

##### trace

`Object`

Trace object from MCMC sampling

##### filepath

`string`

Path to save the file

#### Returns

`void`

***

### loadTrace()

```ts
function loadTrace(filepath): Object;
```

Defined in: [utils/persistence.js:27](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/utils/persistence.js#L27)

Load trace data from JSON file

#### Parameters

##### filepath

`string`

Path to the file

#### Returns

`Object`

Trace object

***

### saveModelConfig()

```ts
function saveModelConfig(model, filepath): void;
```

Defined in: [utils/persistence.js:39](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/utils/persistence.js#L39)

Save model configuration to JSON
Note: This saves the model structure, not the trained parameters

#### Parameters

##### model

`Model`

The model to save

##### filepath

`string`

Path to save the file

#### Returns

`void`

***

### saveModelState()

```ts
function saveModelState(
   model, 
   trace, 
   filepath): void;
```

Defined in: [utils/persistence.js:65](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/utils/persistence.js#L65)

Save complete model state (config + trace)

#### Parameters

##### model

`Model`

The model

##### trace

`Object`

The MCMC trace

##### filepath

`string`

Path to save the file

#### Returns

`void`

***

### loadModelState()

```ts
function loadModelState(filepath): Object;
```

Defined in: [utils/persistence.js:97](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/utils/persistence.js#L97)

Load model state from file

#### Parameters

##### filepath

`string`

Path to the file

#### Returns

`Object`

{modelConfig, trace, metadata}

***

### saveTraceCSV()

```ts
function saveTraceCSV(trace, filepath): void;
```

Defined in: [utils/persistence.js:108](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/utils/persistence.js#L108)

Save trace to CSV format (for external analysis tools)

#### Parameters

##### trace

`Object`

Trace object

##### filepath

`string`

Path to save the file

#### Returns

`void`

***

### exportTraceForBrowser()

```ts
function exportTraceForBrowser(trace): string;
```

Defined in: [utils/persistence.js:135](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/utils/persistence.js#L135)

Export trace for browser environments (no fs dependency)
Returns a downloadable blob URL

#### Parameters

##### trace

`Object`

Trace object

#### Returns

`string`

JSON string for download

***

### importTraceFromJSON()

```ts
function importTraceFromJSON(jsonString): Object;
```

Defined in: [utils/persistence.js:153](https://github.com/tangent-to/mc/blob/6c1d7b7484a03ef091fe7b86e6f9a2bdd1631dab/src/utils/persistence.js#L153)

Import trace from JSON string (browser-compatible)

#### Parameters

##### jsonString

`string`

JSON string

#### Returns

`Object`

Trace object

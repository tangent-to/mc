---
layout: default
title: utils/persistence
parent: API Reference
nav_order: 4
permalink: /api/utils-persistence
---
# utils/persistence

Node-only file persistence for MCMC traces and model configuration.

Reached via the `@tangent.to/mc/persistence` subpath export, this module
uses `node:fs` to save and load traces (JSON/CSV) and model structure to
disk, plus browser-safe string helpers ([exportTraceForBrowser](#exporttraceforbrowser),
[importTraceFromJSON](#importtracefromjson)) that avoid the filesystem. It is intentionally
kept out of the main entry so the browser build stays `fs`-free.

## Functions

### saveTrace()

```ts
function saveTrace(trace, filepath): void;
```

Defined in: [utils/persistence.js:20](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/utils/persistence.js#L20)

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

Defined in: [utils/persistence.js:39](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/utils/persistence.js#L39)

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

Defined in: [utils/persistence.js:51](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/utils/persistence.js#L51)

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

Defined in: [utils/persistence.js:77](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/utils/persistence.js#L77)

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

Defined in: [utils/persistence.js:109](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/utils/persistence.js#L109)

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

Defined in: [utils/persistence.js:120](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/utils/persistence.js#L120)

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

Defined in: [utils/persistence.js:148](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/utils/persistence.js#L148)

Export trace for browser environments (no fs dependency)
Returns a JSON string suitable for download or serialization (e.g. wrap it
in a Blob to create a download link).

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

Defined in: [utils/persistence.js:166](https://github.com/tangent-to/mc/blob/63fcea62eb3faf619a906a8594f421e2a1c9c93b/src/utils/persistence.js#L166)

Import trace from JSON string (browser-compatible)

#### Parameters

##### jsonString

`string`

JSON string

#### Returns

`Object`

Trace object

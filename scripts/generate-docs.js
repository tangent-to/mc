#!/usr/bin/env node
/**
 * Documentation Generator for JSMC
 * Parses JSDoc comments and generates HTML documentation
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const SRC_DIR = path.join(__dirname, '..', 'src');
const DOCS_OUTPUT = path.join(__dirname, '..', 'docs', 'api');

// Parse JSDoc-style comments from source code
function parseJSDoc(content) {
  const jsdocRegex = /\/\*\*\s*([\s\S]*?)\*\//g;
  const docs = [];
  let match;

  while ((match = jsdocRegex.exec(content)) !== null) {
    const comment = match[1];
    const nextCode = content.slice(match.index + match[0].length, match.index + match[0].length + 200);

    // Parse comment tags
    const lines = comment.split('\n').map(line => line.trim().replace(/^\*\s?/, ''));
    const description = [];
    const params = [];
    const returns = {};
    let className = null;
    let methodName = null;

    // Detect class or function name from following code
    const classMatch = nextCode.match(/export\s+class\s+(\w+)/);
    const functionMatch = nextCode.match(/(?:export\s+)?(?:function|const)\s+(\w+)/);
    const methodMatch = nextCode.match(/(\w+)\s*\([^)]*\)\s*{/);

    if (classMatch) className = classMatch[1];
    if (functionMatch) methodName = functionMatch[1];
    else if (methodMatch) methodName = methodMatch[1];

    let currentTag = null;
    for (const line of lines) {
      if (line.startsWith('@param')) {
        const paramMatch = line.match(/@param\s+\{([^}]+)\}\s+(\w+)\s*-?\s*(.*)/);
        if (paramMatch) {
          params.push({
            type: paramMatch[1],
            name: paramMatch[2],
            description: paramMatch[3]
          });
        }
        currentTag = 'param';
      } else if (line.startsWith('@returns')) {
        const returnMatch = line.match(/@returns\s+\{([^}]+)\}\s*(.*)/);
        if (returnMatch) {
          returns.type = returnMatch[1];
          returns.description = returnMatch[2];
        }
        currentTag = 'returns';
      } else if (!line.startsWith('@')) {
        if (currentTag === 'param' && params.length > 0) {
          params[params.length - 1].description += ' ' + line;
        } else if (currentTag === 'returns') {
          returns.description = (returns.description || '') + ' ' + line;
        } else {
          description.push(line);
        }
      }
    }

    docs.push({
      description: description.join(' ').trim(),
      params,
      returns: returns.type ? returns : null,
      className,
      methodName
    });
  }

  return docs;
}

// Process a single file
function processFile(filepath, relativePath) {
  const content = fs.readFileSync(filepath, 'utf8');
  const docs = parseJSDoc(content);

  return {
    path: relativePath,
    docs
  };
}

// Recursively process directory
function processDirectory(dir, baseDir = dir) {
  const results = [];
  const entries = fs.readdirSync(dir, { withFileTypes: true });

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    const relativePath = path.relative(baseDir, fullPath);

    if (entry.isDirectory()) {
      results.push(...processDirectory(fullPath, baseDir));
    } else if (entry.isFile() && entry.name.endsWith('.js')) {
      results.push(processFile(fullPath, relativePath));
    }
  }

  return results;
}

// Generate HTML documentation
function generateHTML(fileData) {
  const header = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>JSMC API Documentation</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      line-height: 1.6;
      color: #333;
      max-width: 1200px;
      margin: 0 auto;
      padding: 20px;
    }
    header {
      background: #2c3e50;
      color: white;
      padding: 30px;
      margin: -20px -20px 40px -20px;
      border-radius: 8px;
    }
    h1 { font-size: 2.5em; margin-bottom: 10px; }
    h2 { color: #2c3e50; margin-top: 40px; margin-bottom: 20px; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
    h3 { color: #34495e; margin-top: 30px; margin-bottom: 15px; }
    nav {
      background: #ecf0f1;
      padding: 20px;
      margin-bottom: 30px;
      border-radius: 8px;
    }
    nav ul { list-style: none; }
    nav li { display: inline-block; margin-right: 20px; }
    nav a { color: #2980b9; text-decoration: none; font-weight: 500; }
    nav a:hover { text-decoration: underline; }
    .file-section {
      background: #f9f9f9;
      padding: 30px;
      margin-bottom: 30px;
      border-radius: 8px;
      border-left: 4px solid #3498db;
    }
    .doc-block {
      background: white;
      padding: 20px;
      margin: 20px 0;
      border-radius: 5px;
      border: 1px solid #ddd;
    }
    .class-name {
      font-size: 1.5em;
      color: #e74c3c;
      font-weight: bold;
    }
    .method-name {
      font-size: 1.2em;
      color: #3498db;
      font-weight: bold;
      font-family: 'Courier New', monospace;
    }
    .description {
      margin: 15px 0;
      color: #555;
    }
    .params, .returns {
      margin-top: 15px;
    }
    .params h4, .returns h4 {
      color: #7f8c8d;
      font-size: 0.9em;
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-bottom: 10px;
    }
    .param {
      margin: 10px 0;
      padding: 10px;
      background: #ecf0f1;
      border-radius: 4px;
    }
    .param-name {
      font-family: 'Courier New', monospace;
      color: #e74c3c;
      font-weight: bold;
    }
    .param-type {
      font-family: 'Courier New', monospace;
      color: #27ae60;
      font-style: italic;
    }
    .param-desc {
      margin-top: 5px;
      color: #555;
    }
    code {
      background: #ecf0f1;
      padding: 2px 6px;
      border-radius: 3px;
      font-family: 'Courier New', monospace;
      color: #e74c3c;
    }
    footer {
      margin-top: 60px;
      padding-top: 20px;
      border-top: 2px solid #ecf0f1;
      text-align: center;
      color: #7f8c8d;
    }
  </style>
</head>
<body>
  <header>
    <h1>JSMC API Documentation</h1>
    <p>Automatically generated from source code JSDoc comments</p>
  </header>

  <nav>
    <ul>`;

  const footer = `
  </ul>
  </nav>

  <main>`;

  const end = `
  </main>

  <footer>
    <p>Generated from JSMC source code | <a href="https://github.com/essicolo/jsmc">GitHub</a></p>
  </footer>
</body>
</html>`;

  let nav = '';
  let content = '';

  for (const file of fileData) {
    if (file.docs.length === 0) continue;

    const sectionId = file.path.replace(/\//g, '-').replace('.js', '');
    nav += `      <li><a href="#${sectionId}">${file.path}</a></li>\n`;

    content += `
    <section id="${sectionId}" class="file-section">
      <h2>${file.path}</h2>
`;

    for (const doc of file.docs) {
      if (!doc.description && doc.params.length === 0 && !doc.returns) continue;

      content += `
      <div class="doc-block">`;

      if (doc.className) {
        content += `
        <div class="class-name">class ${doc.className}</div>`;
      }

      if (doc.methodName) {
        const paramNames = doc.params.map(p => p.name).join(', ');
        content += `
        <div class="method-name">${doc.methodName}(${paramNames})</div>`;
      }

      if (doc.description) {
        content += `
        <div class="description">${doc.description}</div>`;
      }

      if (doc.params.length > 0) {
        content += `
        <div class="params">
          <h4>Parameters</h4>`;
        for (const param of doc.params) {
          content += `
          <div class="param">
            <span class="param-name">${param.name}</span>
            <span class="param-type">{${param.type}}</span>
            ${param.description ? `<div class="param-desc">${param.description}</div>` : ''}
          </div>`;
        }
        content += `
        </div>`;
      }

      if (doc.returns) {
        content += `
        <div class="returns">
          <h4>Returns</h4>
          <div class="param">
            <span class="param-type">{${doc.returns.type}}</span>
            ${doc.returns.description ? `<div class="param-desc">${doc.returns.description}</div>` : ''}
          </div>
        </div>`;
      }

      content += `
      </div>`;
    }

    content += `
    </section>`;
  }

  return header + nav + footer + content + end;
}

// Main function
function main() {
  console.log('Generating API documentation...\n');

  // Process all source files
  const fileData = processDirectory(SRC_DIR);

  // Generate HTML
  const html = generateHTML(fileData);

  // Ensure output directory exists
  if (!fs.existsSync(DOCS_OUTPUT)) {
    fs.mkdirSync(DOCS_OUTPUT, { recursive: true });
  }

  // Write HTML file
  const outputPath = path.join(DOCS_OUTPUT, 'index.html');
  fs.writeFileSync(outputPath, html);

  console.log(`Documentation generated: ${outputPath}`);
  console.log(`\nProcessed ${fileData.length} files`);
  console.log(`Total documentation blocks: ${fileData.reduce((sum, f) => sum + f.docs.length, 0)}`);
}

main();

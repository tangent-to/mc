#!/usr/bin/env node
/**
 * Build the API reference for the just-the-docs site from JSDoc.
 *
 * Pipeline:
 *   1. Run TypeDoc (config in typedoc.json) -> raw markdown in docs/api-generated/
 *   2. Post-process each page for just-the-docs:
 *        - strip any TypeDoc front matter
 *        - add Jekyll front matter (layout/title/parent/nav_order/permalink)
 *        - rewrite relative *.md links to the target page's permalink
 *   3. Write the result into docs/api/ (child pages of the "API Reference" page)
 *
 * Deliberately structure-agnostic: it flattens whatever files TypeDoc emits into
 * one level of children, so it does not depend on a particular module/namespace
 * layout. Usage: node scripts/build-api-docs.mjs   (or: npm run docs:api)
 */
import { execSync } from 'node:child_process';
import {
  rmSync, mkdirSync, readdirSync, readFileSync, writeFileSync, statSync, existsSync,
} from 'node:fs';
import { join, dirname, relative, posix, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const RAW = join(ROOT, 'docs', 'api-generated');
const OUT = join(ROOT, 'docs', 'api');

function walk(dir) {
  const out = [];
  if (!existsSync(dir)) return out;
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) out.push(...walk(p));
    else if (name.endsWith('.md')) out.push(p);
  }
  return out;
}

const titleCase = (s) => s.replace(/[-_/]/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()).trim();
const yaml = (s) => (/[:#"]/.test(s) ? JSON.stringify(s) : s);
// A source path (posix, no extension) -> a flat permalink slug.
const slugOf = (relNoExt) => relNoExt.replace(/\/index$/, '').replace(/\//g, '-') || 'index';

function main() {
  console.log('Running TypeDoc...');
  execSync('npx typedoc', { cwd: ROOT, stdio: 'inherit' });

  const files = walk(RAW).sort();
  const permalinkOf = new Map(); // source rel path -> permalink, for link rewriting
  const pages = [];
  for (const abs of files) {
    const rel = posix.normalize(relative(RAW, abs).split(/[/\\]/).join('/'));
    const relNoExt = rel.replace(/\.md$/, '');
    if (relNoExt === 'index' || relNoExt === 'README' || relNoExt === 'modules') continue; // drop TypeDoc root
    const slug = slugOf(relNoExt);
    const permalink = `/api/${slug}`;
    permalinkOf.set(rel, permalink);
    pages.push({ abs, rel, slug, permalink });
  }

  rmSync(OUT, { recursive: true, force: true });
  mkdirSync(OUT, { recursive: true });

  let written = 0, order = 1;
  for (const { abs, rel, slug, permalink } of pages) {
    let body = readFileSync(abs, 'utf8');
    body = body.replace(/^---\n[\s\S]*?\n---\n/, ''); // strip any TypeDoc front matter
    const h = body.match(/^#{1,2}\s+(.+)$/m);
    const title = (h ? h[1] : titleCase(basename(slug)));
    // Rewrite relative *.md links to permalinks (keep any #anchor).
    body = body.replace(/\]\(([^)]+?\.md)(#[^)]*)?\)/g, (m, target, anchor = '') => {
      if (/^https?:/.test(target)) return m;
      const resolved = posix.normalize(posix.join(posix.dirname(rel), target));
      const perma = permalinkOf.get(resolved);
      return perma ? `](${perma}${anchor})` : m;
    });
    const fm = [
      '---', 'layout: default', `title: ${yaml(title)}`,
      'parent: API Reference', `nav_order: ${order++}`, `permalink: ${permalink}`, '---', '',
    ].join('\n');
    writeFileSync(join(OUT, `${slug}.md`), fm + body);
    written++;
  }

  rmSync(RAW, { recursive: true, force: true });
  console.log(`Wrote ${written} API pages to docs/api/`);
}

main();

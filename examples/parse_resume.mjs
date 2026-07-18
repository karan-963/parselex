#!/usr/bin/env node
// Client example: call the Parselex /inference-v2/parse API and print structured JSON.
//
// With no args, parses the bundled demo resume (full-database/Karan.pdf) and
// saves the result to examples/output/Karan.json.
//
// Usage:
//   node parse_resume.mjs
//   node parse_resume.mjs resume.pdf
//   node parse_resume.mjs resume.pdf --out result.json
//   node parse_resume.mjs resume.pdf --url http://localhost:8000 --precision int8

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { basename, dirname, extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const DEFAULT_PDF = join(SCRIPT_DIR, '..', 'full-database', 'Karan.pdf');

function parseArgs(argv) {
  const args = { pdf: null, url: 'http://localhost:8000', precision: 'fp32', out: null };
  const rest = [];
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--url') args.url = argv[++i];
    else if (argv[i] === '--precision') args.precision = argv[++i];
    else if (argv[i] === '--out') args.out = argv[++i];
    else rest.push(argv[i]);
  }
  args.pdf = rest[0] ?? DEFAULT_PDF;
  if (!args.out) {
    const stem = basename(args.pdf, extname(args.pdf));
    args.out = join(SCRIPT_DIR, 'output', `${stem}.json`);
  }
  return args;
}

async function parseResume(pdfPath, baseUrl, precision) {
  const bytes = await readFile(pdfPath);
  const form = new FormData();
  form.append('file', new Blob([bytes], { type: 'application/pdf' }), basename(pdfPath));

  const url = `${baseUrl.replace(/\/$/, '')}/inference-v2/parse?precision=${precision}`;
  const res = await fetch(url, { method: 'POST', body: form });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}: ${JSON.stringify(data)}`);
  }
  return data;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  let result;
  try {
    result = await parseResume(args.pdf, args.url, args.precision);
  } catch (err) {
    console.error(`Failed to reach ${args.url} — is the engine running?\n${err.message}`);
    process.exit(1);
  }

  const output = JSON.stringify(result.structured, null, 2);
  await mkdir(dirname(args.out), { recursive: true });
  await writeFile(args.out, output);
  console.log(`Wrote structured JSON to ${args.out}`);
}

main();

'use strict';

const fs = require('node:fs');
const path = require('node:path');

const ROOT = path.resolve(__dirname, '..');
const DESTINATION = path.join(ROOT, '_site');
const PUBLIC_ENTRIES = ['index.html', 'styles.css', 'blog', 'assets', 'CNAME'];

fs.rmSync(DESTINATION, { recursive: true, force: true });
fs.mkdirSync(DESTINATION, { recursive: true });

for (const entry of PUBLIC_ENTRIES) {
  const source = path.join(ROOT, entry);
  if (!fs.existsSync(source)) continue;
  fs.cpSync(source, path.join(DESTINATION, entry), { recursive: true });
}

fs.writeFileSync(path.join(DESTINATION, '.nojekyll'), '');
process.stdout.write('Prepared _site for GitHub Pages.\n');

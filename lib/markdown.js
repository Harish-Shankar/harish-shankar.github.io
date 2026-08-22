'use strict';

const katex = require('katex');

const PYTHON_KEYWORDS = new Set([
  'and', 'as', 'assert', 'async', 'await', 'break', 'case', 'class', 'continue',
  'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global',
  'if', 'import', 'in', 'is', 'lambda', 'match', 'nonlocal', 'not', 'or', 'pass',
  'raise', 'return', 'try', 'while', 'with', 'yield',
]);

const PYTHON_BUILTINS = new Set([
  'abs', 'all', 'any', 'bool', 'dict', 'enumerate', 'filter', 'float', 'int',
  'len', 'list', 'map', 'max', 'min', 'open', 'print', 'range', 'reversed',
  'round', 'set', 'sorted', 'str', 'sum', 'super', 'tuple', 'type', 'zip',
]);

const PYTHON_CONSTANTS = new Set(['False', 'None', 'True', 'Ellipsis', 'NotImplemented']);

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function highlightPython(source) {
  let output = '';
  let cursor = 0;
  const token = (className, value) => `<span class="token ${className}">${escapeHtml(value)}</span>`;

  while (cursor < source.length) {
    const rest = source.slice(cursor);

    if (/^\s/.test(rest)) {
      const whitespace = rest.match(/^\s+/)[0];
      output += whitespace;
      cursor += whitespace.length;
      continue;
    }

    if (source[cursor] === '#') {
      const end = source.indexOf('\n', cursor);
      const value = source.slice(cursor, end === -1 ? source.length : end);
      output += token('comment', value);
      cursor += value.length;
      continue;
    }

    const stringStart = rest.match(/^(?:(?:[rRuUbBfF]|[rR][fF]|[fF][rR]|[rR][bB]|[bB][rR])?)("""|'''|"|')/);
    if (stringStart) {
      const delimiter = stringStart[1];
      let end = cursor + stringStart[0].length;
      while (end < source.length) {
        if (source.startsWith(delimiter, end)) {
          end += delimiter.length;
          break;
        }
        if (source[end] === '\\' && delimiter.length === 1) end += 2;
        else end += 1;
      }
      output += token('string', source.slice(cursor, end));
      cursor = end;
      continue;
    }

    const number = rest.match(/^(?:0[xX][\da-fA-F]+|0[bB][01]+|0[oO][0-7]+|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?j?)/);
    if (number) {
      output += token('number', number[0]);
      cursor += number[0].length;
      continue;
    }

    const identifier = rest.match(/^[A-Za-z_]\w*/);
    if (identifier) {
      const value = identifier[0];
      const className = PYTHON_KEYWORDS.has(value) ? 'keyword'
        : PYTHON_CONSTANTS.has(value) ? 'constant'
          : PYTHON_BUILTINS.has(value) ? 'builtin'
            : '';
      output += className ? token(className, value) : escapeHtml(value);
      cursor += value.length;
      continue;
    }

    const operator = rest.match(/^(?:\*\*|\/\/|:=|==|!=|<=|>=|<<|>>|->|\+=|-=|\*=|\/=|%=|&=|\|=|\^=|@|[+\-*\/%=&|^~<>:])/);
    if (operator) {
      output += token('operator', operator[0]);
      cursor += operator[0].length;
      continue;
    }

    output += escapeHtml(source[cursor]);
    cursor += 1;
  }

  return output;
}

function renderMath(source, displayMode = false) {
  return katex.renderToString(source.trim(), {
    displayMode,
    output: 'htmlAndMathml',
    throwOnError: false,
    errorColor: '#8f2f2f',
    trust: false,
  });
}

function parseFrontMatter(source) {
  const normalized = source.replace(/\r\n?/g, '\n');
  const lines = normalized.split('\n');

  if (lines[0]?.trim() !== '---') {
    return { metadata: {}, body: normalized };
  }

  const metadata = {};
  let cursor = 1;

  for (; cursor < lines.length && lines[cursor].trim() !== '---'; cursor += 1) {
    const separator = lines[cursor].indexOf(':');
    if (separator === -1) continue;

    const key = lines[cursor].slice(0, separator).trim();
    let value = lines[cursor].slice(separator + 1).trim();
    if ((value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    metadata[key] = value;
  }

  if (cursor === lines.length) {
    throw new Error('Front matter opens with --- but never closes.');
  }

  return { metadata, body: lines.slice(cursor + 1).join('\n') };
}

function isSafeUrl(url) {
  const value = url.trim();
  return /^(https?:|mailto:|\/|#|\.\.?\/)/i.test(value) ||
    (!/^[a-z][a-z\d+.-]*:/i.test(value) && value !== '');
}

function parseDestination(raw) {
  const match = raw.trim().match(/^(\S+?)(?:\s+["']([^"']*)["'])?$/);
  if (!match) return null;
  return { url: match[1], title: match[2] || '' };
}

function formatProgress(timestampValue, options = {}) {
  const timestamp = new Date(timestampValue || options.buildTimestamp || Date.now());
  if (Number.isNaN(timestamp.getTime())) throw new Error('The progress timestamp is invalid.');
  const timeZone = options.timeZone || 'Asia/Kolkata';
  const label = new Intl.DateTimeFormat('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    timeZone,
    timeZoneName: 'short',
  }).format(timestamp);
  return { timestamp, label };
}

const ANNOTATION_LABELS = {
  cite: ['Citation', 'Citations'],
  note: ['Note', 'Notes'],
  definition: ['Definition', 'Definitions'],
};

function annotationLabel(variant, count = 1) {
  const [singular, plural] = ANNOTATION_LABELS[variant] || ANNOTATION_LABELS.note;
  return count > 1 ? plural : singular;
}

function splitAnnotationEntries(source) {
  const entries = [];
  let current = '';
  let cursor = 0;

  while (cursor < source.length) {
    if (source[cursor] === '\\' && cursor + 1 < source.length) {
      current += source.slice(cursor, cursor + 2);
      cursor += 2;
      continue;
    }

    if (source[cursor] === '`') {
      const end = source.indexOf('`', cursor + 1);
      const stop = end === -1 ? source.length : end + 1;
      current += source.slice(cursor, stop);
      cursor = stop;
      continue;
    }

    if (source.startsWith('||', cursor)) {
      entries.push(current);
      current = '';
      cursor += 2;
      continue;
    }

    current += source[cursor];
    cursor += 1;
  }

  entries.push(current);
  const trimmed = entries.map((entry) => entry.trim()).filter((entry) => entry !== '');
  return trimmed.length > 0 ? trimmed : [source.trim()];
}

function renderInline(source, options = {}) {
  let output = '';
  let plain = '';
  let cursor = 0;

  const flushPlain = () => {
    output += escapeHtml(plain);
    plain = '';
  };

  while (cursor < source.length) {
    if (source[cursor] === '\\' && cursor + 1 < source.length) {
      plain += source[cursor + 1];
      cursor += 2;
      continue;
    }

    if (source.startsWith(':::', cursor)) {
      const progress = source.slice(cursor).match(/^:::progress(?:\s+(\S+))?\s+:::/);
      if (progress) {
        plain = plain.replace(/\s+$/, '');
        flushPlain();
        const { timestamp, label } = formatProgress(progress[1], options);
        output += `<span class="progress-mark progress-mark--inline"><time datetime="${escapeHtml(timestamp.toISOString())}">Written ${escapeHtml(label)}</time></span>`;
        cursor += progress[0].length;
        continue;
      }

      const annotation = source.slice(cursor).match(/^:::(definition|cite|note)\s+(.+?)\s+:::/);
      if (annotation) {
        plain = plain.replace(/\s+$/, '');
        flushPlain();
        const variant = annotation[1];
        const entries = splitAnnotationEntries(annotation[2]);
        const label = annotationLabel(variant, entries.length);
        const content = entries
          .map((entry) => `<span class="margin-note__entry">${renderInline(entry, options)}</span>`)
          .join('');
        output += `<span class="margin-note margin-note--inline margin-note--${variant}" role="note" aria-label="${label}"><span class="margin-note-label">${label}</span><span class="margin-note__content">${content}</span></span>`;
        cursor += annotation[0].length;
        continue;
      }
    }

    if (source.startsWith('![', cursor)) {
      const labelEnd = source.indexOf('](', cursor + 2);
      const destinationEnd = labelEnd === -1 ? -1 : source.indexOf(')', labelEnd + 2);
      if (labelEnd !== -1 && destinationEnd !== -1) {
        const alt = source.slice(cursor + 2, labelEnd);
        const destination = parseDestination(source.slice(labelEnd + 2, destinationEnd));
        if (destination && (destination.url === 'placeholder' || isSafeUrl(destination.url))) {
          flushPlain();
          if (destination.url === 'placeholder') {
            output += `<span class="md-image-placeholder" role="img" aria-label="${escapeHtml(alt)}">${escapeHtml(alt || 'Image placeholder')}</span>`;
          } else {
            const title = destination.title ? ` title="${escapeHtml(destination.title)}"` : '';
            output += `<img src="${escapeHtml(destination.url)}" alt="${escapeHtml(alt)}" loading="lazy"${title}>`;
          }
          cursor = destinationEnd + 1;
          continue;
        }
      }
    }

    if (source[cursor] === '[') {
      const labelEnd = source.indexOf('](', cursor + 1);
      const destinationEnd = labelEnd === -1 ? -1 : source.indexOf(')', labelEnd + 2);
      if (labelEnd !== -1 && destinationEnd !== -1) {
        const destination = parseDestination(source.slice(labelEnd + 2, destinationEnd));
        if (destination && isSafeUrl(destination.url)) {
          flushPlain();
          const label = source.slice(cursor + 1, labelEnd);
          const title = destination.title ? ` title="${escapeHtml(destination.title)}"` : '';
          output += `<a href="${escapeHtml(destination.url)}"${title}>${renderInline(label, options)}</a>`;
          cursor = destinationEnd + 1;
          continue;
        }
      }
    }

    if (source[cursor] === '`') {
      const end = source.indexOf('`', cursor + 1);
      if (end !== -1) {
        flushPlain();
        output += `<code>${escapeHtml(source.slice(cursor + 1, end))}</code>`;
        cursor = end + 1;
        continue;
      }
    }

    if (source[cursor] === '$' && source[cursor + 1] !== '$' && !/\s/.test(source[cursor + 1] || '')) {
      let end = cursor + 1;
      while (end < source.length) {
        if (source[end] === '$' && source[end - 1] !== '\\' && !/\s/.test(source[end - 1])) break;
        end += 1;
      }
      if (end < source.length) {
        flushPlain();
        output += `<span class="math-inline">${renderMath(source.slice(cursor + 1, end))}</span>`;
        cursor = end + 1;
        continue;
      }
    }

    const pairedMarkers = [
      ['**', 'strong'],
      ['__', 'strong'],
      ['~~', 'del'],
      ['*', 'em'],
      ['_', 'em'],
    ];
    let matched = false;

    for (const [marker, tag] of pairedMarkers) {
      if (!source.startsWith(marker, cursor)) continue;
      const end = source.indexOf(marker, cursor + marker.length);
      if (end === -1 || end === cursor + marker.length) continue;

      flushPlain();
      output += `<${tag}>${renderInline(source.slice(cursor + marker.length, end), options)}</${tag}>`;
      cursor = end + marker.length;
      matched = true;
      break;
    }

    if (matched) continue;
    plain += source[cursor];
    cursor += 1;
  }

  flushPlain();
  return output;
}

function isHorizontalRule(line) {
  const compact = line.trim().replaceAll(' ', '');
  return compact.length >= 3 && (/^-+$/.test(compact) || /^\*+$/.test(compact) || /^_+$/.test(compact));
}

function isTableDivider(line) {
  const cells = line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|');
  return cells.length > 0 && cells.every((cell) => /^\s*:?-{3,}:?\s*$/.test(cell));
}

function splitTableRow(line) {
  return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((cell) => cell.trim());
}

function splitTokenizerRow(line) {
  return line.split('|');
}

function beginsBlock(lines, index) {
  const line = lines[index] || '';
  const next = lines[index + 1] || '';
  return /^#{1,6}\s+/.test(line) || /^```/.test(line.trim()) || /^>\s?/.test(line) ||
    /^\$\$/.test(line.trim()) ||
    /^:::progress(?:\s+\S+)?\s*$/.test(line.trim()) ||
    /^:::tokens\s*$/.test(line.trim()) ||
    /^:::remark(?:\s+.+)?\s*$/.test(line.trim()) ||
    /^:::definition(?:\s+.+)?\s*$/.test(line.trim()) ||
    /^:::(cite|note)\s*$/.test(line.trim()) ||
    /^\s*[-+*]\s+/.test(line) || /^\s*\d+[.)]\s+/.test(line) || isHorizontalRule(line) ||
    (line.includes('|') && isTableDivider(next));
}

function parseBlocks(source) {
  const lines = source.replace(/\r\n?/g, '\n').split('\n');
  const nodes = [];
  let cursor = 0;

  while (cursor < lines.length) {
    const line = lines[cursor];

    if (line.trim() === '') {
      cursor += 1;
      continue;
    }

    const progress = line.trim().match(/^:::progress(?:\s+(\S+))?\s*$/);
    if (progress) {
      nodes.push({ type: 'progress', timestamp: progress[1] || '' });
      cursor += 1;
      continue;
    }

    if (/^:::tokens\s*$/.test(line.trim())) {
      const tokenRows = [];
      cursor += 1;
      while (cursor < lines.length && lines[cursor].trim() !== ':::') {
        if (lines[cursor] !== '') tokenRows.push(lines[cursor]);
        cursor += 1;
      }
      if (cursor === lines.length) throw new Error('The tokens block was never closed.');
      const tokens = tokenRows.flatMap(splitTokenizerRow);
      if (tokens.length === 0) throw new Error('The tokens block must contain at least one token.');
      nodes.push({ type: 'tokens', tokens });
      cursor += 1;
      continue;
    }

    const remark = line.trim().match(/^:::remark(?:\s+(.+?))?\s*$/);
    if (remark) {
      const content = [];
      cursor += 1;
      while (cursor < lines.length && lines[cursor].trim() !== ':::') {
        content.push(lines[cursor]);
        cursor += 1;
      }
      if (cursor === lines.length) throw new Error('The remark block was never closed.');
      nodes.push({
        type: 'remark',
        title: remark[1] || '',
        children: parseBlocks(content.join('\n')),
      });
      cursor += 1;
      continue;
    }

    const definition = line.trim().match(/^:::definition(?:\s+(.+?))?\s*$/);
    if (definition) {
      const content = [];
      cursor += 1;
      while (cursor < lines.length && lines[cursor].trim() !== ':::') {
        content.push(lines[cursor]);
        cursor += 1;
      }
      if (cursor === lines.length) throw new Error('The definition block was never closed.');
      nodes.push({
        type: 'definition',
        title: definition[1] || '',
        children: parseBlocks(content.join('\n')),
      });
      cursor += 1;
      continue;
    }

    const marginBlock = line.trim().match(/^:::(cite|note)\s*$/);
    if (marginBlock) {
      const content = [];
      cursor += 1;
      while (cursor < lines.length && lines[cursor].trim() !== ':::') {
        content.push(lines[cursor]);
        cursor += 1;
      }
      if (cursor === lines.length) throw new Error(`The ${marginBlock[1]} block was never closed.`);
      const children = parseBlocks(content.join('\n')).flatMap((child) => (
        child.type === 'paragraph'
          ? splitAnnotationEntries(child.value).map((value) => ({ type: 'paragraph', value }))
          : [child]
      ));
      nodes.push({ type: 'marginNote', variant: marginBlock[1], children });
      cursor += 1;
      continue;
    }

    if (line.trim().startsWith('$$')) {
      const trimmed = line.trim();
      if (trimmed.length > 4 && trimmed.endsWith('$$')) {
        nodes.push({ type: 'mathBlock', value: trimmed.slice(2, -2).trim() });
        cursor += 1;
        continue;
      }

      const math = [];
      const openingContent = trimmed.slice(2).trim();
      if (openingContent) math.push(openingContent);
      cursor += 1;
      while (cursor < lines.length && !lines[cursor].trim().endsWith('$$')) {
        math.push(lines[cursor]);
        cursor += 1;
      }
      if (cursor === lines.length) throw new Error('Display math opens with $$ but never closes.');
      const closingContent = lines[cursor].trim().slice(0, -2).trim();
      if (closingContent) math.push(closingContent);
      nodes.push({ type: 'mathBlock', value: math.join('\n') });
      cursor += 1;
      continue;
    }

    const fence = line.trim().match(/^```([\w-]*)\s*$/);
    if (fence) {
      const code = [];
      cursor += 1;
      while (cursor < lines.length && !/^```\s*$/.test(lines[cursor].trim())) {
        code.push(lines[cursor]);
        cursor += 1;
      }
      if (cursor === lines.length) throw new Error('Code fence was never closed.');
      nodes.push({ type: 'codeBlock', language: fence[1], value: code.join('\n') });
      cursor += 1;
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.+?)\s*#*$/);
    if (heading) {
      nodes.push({ type: 'heading', level: heading[1].length, value: heading[2] });
      cursor += 1;
      continue;
    }

    if (isHorizontalRule(line)) {
      nodes.push({ type: 'horizontalRule' });
      cursor += 1;
      continue;
    }

    if (/^>\s?/.test(line)) {
      const quote = [];
      while (cursor < lines.length && (/^>\s?/.test(lines[cursor]) || lines[cursor].trim() === '')) {
        quote.push(lines[cursor].replace(/^>\s?/, ''));
        cursor += 1;
      }
      nodes.push({ type: 'blockquote', children: parseBlocks(quote.join('\n')) });
      continue;
    }

    const unordered = line.match(/^\s*[-+*]\s+(.+)$/);
    const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/);
    if (unordered || ordered) {
      const type = unordered ? 'unorderedList' : 'orderedList';
      const pattern = unordered ? /^\s*[-+*]\s+(.+)$/ : /^\s*\d+[.)]\s+(.+)$/;
      const items = [];
      while (cursor < lines.length) {
        const item = lines[cursor].match(pattern);
        if (!item) break;
        const task = item[1].match(/^\[([ xX])\]\s+(.+)$/);
        items.push(task ? { value: task[2], checked: task[1].toLowerCase() === 'x' } : { value: item[1] });
        cursor += 1;
      }
      nodes.push({ type, items });
      continue;
    }

    if (line.includes('|') && isTableDivider(lines[cursor + 1] || '')) {
      const headers = splitTableRow(line);
      const rows = [];
      cursor += 2;
      while (cursor < lines.length && lines[cursor].includes('|') && lines[cursor].trim() !== '') {
        rows.push(splitTableRow(lines[cursor]));
        cursor += 1;
      }
      nodes.push({ type: 'table', headers, rows });
      continue;
    }

    const paragraph = [line.trim()];
    cursor += 1;
    while (cursor < lines.length && lines[cursor].trim() !== '' && !beginsBlock(lines, cursor)) {
      paragraph.push(lines[cursor].trim());
      cursor += 1;
    }
    nodes.push({ type: 'paragraph', value: paragraph.join(' ') });
  }

  return nodes;
}

function slugify(value) {
  return value.toLowerCase()
    .replace(/<[^>]+>/g, '')
    .replace(/[^a-z\d\s-]/g, '')
    .trim()
    .replace(/\s+/g, '-');
}

function renderNodes(nodes, options = {}) {
  return nodes.map((node) => {
    switch (node.type) {
      case 'heading': {
        const id = slugify(node.value);
        return `<h${node.level}${id ? ` id="${escapeHtml(id)}"` : ''}>${renderInline(node.value, options)}</h${node.level}>`;
      }
      case 'paragraph':
        return `<p>${renderInline(node.value, options)}</p>`;
      case 'horizontalRule':
        return '<hr>';
      case 'codeBlock': {
        const language = node.language ? ` data-language="${escapeHtml(node.language)}"` : '';
        const label = node.language ? `<figcaption>${escapeHtml(node.language)}</figcaption>` : '';
        const normalizedLanguage = node.language.toLowerCase();
        const highlightedCode = normalizedLanguage === 'python' || normalizedLanguage === 'py'
          ? highlightPython(node.value)
          : escapeHtml(node.value);
        const code = highlightedCode
          .split('\n')
          .map((line) => `<span class="code-line">${line || ' '}</span>`)
          .join('');
        const className = normalizedLanguage ? ` class="language-${escapeHtml(normalizedLanguage)}"` : '';
        return `<figure class="code-block"${language}>${label}<pre><code${className}>${code}</code></pre></figure>`;
      }
      case 'mathBlock':
        return `<div class="math-display" role="math">${renderMath(node.value, true)}</div>`;
      case 'tokens': {
        const sentence = node.tokens.join('');
        const tokens = node.tokens.map((value, index) => (
          `<span class="token-strip__token" data-token-number="${index + 1}">${escapeHtml(value)}</span>`
        )).join('');
        return `<code class="token-strip" aria-label="${escapeHtml(`${node.tokens.length} tokens: ${sentence}`)}">${tokens}</code>`;
      }
      case 'blockquote':
        return `<blockquote>${renderNodes(node.children, options)}</blockquote>`;
      case 'definition': {
        const title = node.title
          ? `<span class="definition-block__term">${renderInline(node.title, options)}</span>`
          : '';
        const accessibleName = node.title ? `Definition: ${node.title}` : 'Definition';
        return `<div class="definition-block" role="definition" aria-label="${escapeHtml(accessibleName)}"><div class="definition-block__heading"><span class="definition-block__label">Definition</span>${title}</div><div class="definition-block__content">${renderNodes(node.children, options)}</div></div>`;
      }
      case 'remark': {
        const title = node.title
          ? `<span class="remark-block__title">${renderInline(node.title, options)}</span>`
          : '';
        return `<details class="remark-block"><summary class="remark-block__summary"><span class="remark-block__label">Remark</span>${title}<span class="remark-block__toggle" aria-hidden="true"></span></summary><div class="remark-block__content">${renderNodes(node.children, options)}</div></details>`;
      }
      case 'marginNote': {
        const entryCount = node.children.filter((child) => child.type === 'paragraph').length;
        const label = annotationLabel(node.variant, entryCount);
        return `<aside class="margin-note margin-note--${node.variant}" aria-label="${label}"><span class="margin-note-label">${label}</span>${renderNodes(node.children, options)}</aside>`;
      }
      case 'progress': {
        const { timestamp, label } = formatProgress(node.timestamp, options);
        return `<p class="progress-mark"><time datetime="${escapeHtml(timestamp.toISOString())}">Written ${escapeHtml(label)}</time></p>`;
      }
      case 'unorderedList':
      case 'orderedList': {
        const tag = node.type === 'orderedList' ? 'ol' : 'ul';
        const items = node.items.map((item) => {
          const task = typeof item.checked === 'boolean'
            ? `<input type="checkbox" disabled${item.checked ? ' checked' : ''} aria-hidden="true">`
            : '';
          return `<li${task ? ' class="task-item"' : ''}>${task}${renderInline(item.value, options)}</li>`;
        }).join('');
        return `<${tag}>${items}</${tag}>`;
      }
      case 'table': {
        const headers = node.headers.map((cell) => `<th scope="col">${renderInline(cell, options)}</th>`).join('');
        const rows = node.rows.map((row) => `<tr>${row.map((cell) => `<td>${renderInline(cell, options)}</td>`).join('')}</tr>`).join('');
        return `<div class="table-wrap"><table><thead><tr>${headers}</tr></thead><tbody>${rows}</tbody></table></div>`;
      }
      default:
        throw new Error(`Unknown Markdown node: ${node.type}`);
    }
  }).join('\n');
}

function parseMarkdown(source) {
  const { metadata, body } = parseFrontMatter(source);
  const nodes = parseBlocks(body);
  return { metadata, nodes, html: renderNodes(nodes) };
}

module.exports = {
  escapeHtml,
  highlightPython,
  parseBlocks,
  parseFrontMatter,
  parseMarkdown,
  renderInline,
  renderMath,
  renderNodes,
  slugify,
  splitAnnotationEntries,
};

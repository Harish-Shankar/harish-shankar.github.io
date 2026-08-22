'use strict';

const fs = require('node:fs');
const path = require('node:path');
const {
  ANCHOR_PATTERN,
  escapeHtml,
  parseMarkdown,
  renderInline,
  renderNodes,
  slugify,
} = require('../lib/markdown');

const ROOT = path.resolve(__dirname, '..');
const POSTS_DIRECTORY = path.join(ROOT, 'posts');
const BLOG_DIRECTORY = path.join(ROOT, 'blog');

/* A cross-reference preview shows the opening of the section it points at, not the whole of it. */
const EXCERPT_BLOCK_LIMIT = 4;
const EXCERPT_CHARACTER_LIMIT = 620;

function navigation() {
  return `<nav class="site-nav" aria-label="Primary navigation">
    <a href="/">home</a>
  </nav>`;
}

function pageHead({ title, description, canonical, type = 'article' }) {
  const safeTitle = escapeHtml(title);
  const safeDescription = escapeHtml(description || 'Writing by Harish Shankar.');
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="${safeDescription}">
  <meta name="author" content="Harish Shankar">
  <meta property="og:title" content="${safeTitle}">
  <meta property="og:description" content="${safeDescription}">
  <meta property="og:type" content="${escapeHtml(type)}">
  <meta property="og:url" content="${escapeHtml(canonical)}">
  <meta name="twitter:card" content="summary">
  <link rel="canonical" href="${escapeHtml(canonical)}">
  <title>${safeTitle} — Harish Shankar</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@1,500;1,600&amp;family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&amp;family=Source+Sans+3:ital,wght@0,400;0,500;0,600;0,700;1,400;1,600&amp;family=Spline+Sans+Mono:wght@500;600&amp;display=swap" rel="stylesheet">
  <link rel="stylesheet" href="/styles.css">
  <link rel="stylesheet" href="/blog/vendor/katex/katex.min.css">
  <link rel="stylesheet" href="/blog/blog.css">
</head>`;
}

function formatPostDate(value) {
  const date = new Date(`${value}T00:00:00Z`);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    timeZone: 'UTC',
  }).format(date);
}

function markdownFiles(directory) {
  return fs.readdirSync(directory, { withFileTypes: true })
    .flatMap((entry) => {
      if (entry.name.startsWith('_')) return [];
      const entryPath = path.join(directory, entry.name);
      if (entry.isDirectory()) return markdownFiles(entryPath);
      return entry.isFile() && entry.name.endsWith('.md') ? [entryPath] : [];
    })
    .sort();
}

function sectionHeading(post) {
  return post.nodes.find((node) => node.type === 'heading' && node.level === 2);
}

function seriesOrder(post) {
  const order = Number(post.metadata.series_order);
  return Number.isFinite(order) ? order : Number.MAX_SAFE_INTEGER;
}

function seriesPostsFor(post, posts) {
  if (!post.metadata.series) return [post];
  return posts
    .filter((candidate) => candidate.metadata.series === post.metadata.series)
    .sort((left, right) => seriesOrder(left) - seriesOrder(right));
}

function articleNodes(post) {
  const nodes = [...post.nodes];
  if (nodes[0]?.type === 'heading' && nodes[0].level === 1 && nodes[0].value.trim() === post.metadata.title?.trim()) {
    nodes.shift();
  }
  return nodes;
}

function forEachNodeText(node, visit) {
  if (typeof node.value === 'string') visit(node.value);
  if (typeof node.title === 'string') visit(node.title);
  for (const item of node.items || []) visit(item.value);
  for (const cell of node.headers || []) visit(cell);
  for (const row of node.rows || []) for (const cell of row) visit(cell);
  for (const child of node.children || []) forEachNodeText(child, visit);
}

function nodeTextLength(node) {
  let length = 0;
  forEachNodeText(node, (value) => { length += value.length; });
  return length;
}

/* Every heading, plus every :::anchor marker, is addressable by a cross-reference. */
function buildAnchorIndex(posts) {
  const index = new Map();

  for (const post of posts) {
    const nodes = articleNodes(post);
    const anchors = new Map();
    let heading = null;

    nodes.forEach((node, position) => {
      if (node.type === 'heading') {
        heading = node;
        const id = slugify(node.value);
        if (id && !anchors.has(id)) {
          anchors.set(id, { kind: 'heading', position, level: node.level, title: node.value });
        }
      }
      forEachNodeText(node, (value) => {
        for (const match of value.matchAll(ANCHOR_PATTERN)) {
          if (anchors.has(match[1])) {
            throw new Error(`${post.filename} defines the anchor "${match[1]}" more than once.`);
          }
          anchors.set(match[1], {
            kind: 'anchor',
            position,
            title: heading ? heading.value : post.metadata.title,
          });
        }
      });
    });

    index.set(post.slug, { post, nodes, anchors });
  }

  return index;
}

function collectExcerpt(nodes, entry) {
  if (entry.kind === 'anchor') return { blocks: [nodes[entry.position]], truncated: false };

  const blocks = [];
  let characters = 0;
  let truncated = false;

  for (let position = entry.position + 1; position < nodes.length; position += 1) {
    const node = nodes[position];
    if (node.type === 'heading' && node.level <= entry.level) break;
    if (node.type === 'progress') continue;
    if (blocks.length >= EXCERPT_BLOCK_LIMIT || characters >= EXCERPT_CHARACTER_LIMIT) {
      truncated = true;
      break;
    }
    blocks.push(node);
    characters += nodeTextLength(node);
  }

  return { blocks, truncated };
}

function previewId(slug, anchor) {
  return `xref-${slug.replace(/[^a-z\d]+/g, '-')}--${anchor}`;
}

function renderPreview({ entry, target, href, id, options }) {
  const { blocks, truncated } = collectExcerpt(target.nodes, entry);
  const section = sectionHeading(target.post);
  const source = section ? section.value : target.post.metadata.title;
  const ellipsis = truncated ? '\n        <p class="xref-preview__truncation" aria-hidden="true">…</p>' : '';

  return `      <article class="xref-preview" id="${escapeHtml(id)}">
        <p class="xref-preview__source">${renderInline(source, options)}</p>
        <p class="xref-preview__title">${renderInline(entry.title, options)}</p>
        <div class="xref-preview__body">
${renderNodes(blocks, options)}
        </div>${ellipsis}
        <a class="xref-preview__link" href="${escapeHtml(href)}">Read in full</a>
      </article>`;
}

/* Resolves :::ref targets against the anchor index and collects the previews the page needs. */
function createReferenceResolver({ post, index, previews, options }) {
  const resolve = (target, raw) => {
    const slug = target.slug || post.slug;
    const entryIndex = index.get(slug);
    if (!entryIndex) {
      throw new Error(`${post.filename} references the post "${slug}", which does not exist.`);
    }

    let anchor = target.anchor;
    if (!anchor) {
      const section = sectionHeading(entryIndex.post);
      anchor = section ? slugify(section.value) : '';
    }

    const entry = entryIndex.anchors.get(anchor);
    if (!entry) {
      throw new Error(`${post.filename} references "${raw}", but ${entryIndex.post.filename} has no heading or anchor "${anchor}".`);
    }

    return {
      entry,
      target: entryIndex,
      slug,
      anchor,
      href: slug === post.slug ? `#${anchor}` : `/blog/${slug}/#${anchor}`,
    };
  };

  /* References inside a preview stay plain links, so previews never nest. */
  const previewOptions = {
    ...options,
    resolveReference: (target, raw) => ({ href: resolve(target, raw).href }),
  };

  return (target, raw) => {
    const resolved = resolve(target, raw);
    const id = previewId(resolved.slug, resolved.anchor);
    if (!previews.has(id)) {
      previews.set(id, renderPreview({ ...resolved, id, options: previewOptions }));
    }
    return { href: resolved.href, previewId: id };
  };
}

function postHref(post, heading) {
  const anchor = heading ? `#${slugify(heading.value)}` : '';
  return `/blog/${post.slug}/${anchor}`;
}

function renderTocLink({ post, heading, currentPost, local = false, attributes = '', children = '' }) {
  const isCurrentPage = post.slug === currentPost.slug && heading.level === 2;
  const classes = [
    'article-toc__item',
    `article-toc__item--level-${heading.level}`,
    isCurrentPage ? 'article-toc__item--current' : '',
  ].filter(Boolean).join(' ');
  const href = heading.level === 2
    ? postHref(post)
    : local
      ? `#${slugify(heading.value)}`
      : postHref(post, heading);
  const linkAttributes = [
    local && heading.level > 2 ? 'data-local-section' : '',
    isCurrentPage ? 'data-page-link aria-current="page"' : '',
    attributes,
  ].filter(Boolean).join(' ');
  return `      <li class="${classes}">
        <a href="${escapeHtml(href)}"${linkAttributes ? ` ${linkAttributes}` : ''}>${escapeHtml(heading.value)}</a>${children ? `\n${children}` : ''}
      </li>`;
}

function renderTableOfContents(post, nodes, posts) {
  const siblings = seriesPostsFor(post, posts);
  const links = [];

  for (const sibling of siblings) {
    const siblingSection = sectionHeading(sibling);
    if (!siblingSection) continue;

    const isCurrent = sibling.slug === post.slug;
    if (isCurrent) {
      const subsectionId = `toc-${slugify(siblingSection.value)}-subsections`;
      const subsectionLinks = nodes
        .filter((node) => node.type === 'heading' && (node.level === 3 || node.level === 4))
        .map((heading) => renderTocLink({
          post: sibling,
          heading,
          currentPost: post,
          local: true,
        }));
      const subsections = subsectionLinks.length > 0
        ? `        <ol class="article-toc__subsections" id="${escapeHtml(subsectionId)}" hidden>
${subsectionLinks.map((link) => link.replace(/^/gm, '    ')).join('\n')}
        </ol>`
        : '';
      links.push(renderTocLink({
        post: sibling,
        heading: siblingSection,
        currentPost: post,
        local: true,
        attributes: subsectionLinks.length > 0
          ? `aria-expanded="false" aria-controls="${escapeHtml(subsectionId)}"`
          : '',
        children: subsections,
      }));
    } else {
      links.push(renderTocLink({
        post: sibling,
        heading: siblingSection,
        currentPost: post,
      }));
    }
  }

  if (links.length === 0) return '';

  return `  <nav class="article-toc" aria-label="Article contents">
    <p class="article-toc__label">Contents</p>
    <ol>
${links.join('\n')}
    </ol>
  </nav>`;
}

function renderSeriesPagination(post, posts) {
  if (!post.metadata.series) return '';
  const siblings = seriesPostsFor(post, posts);
  const currentIndex = siblings.findIndex((candidate) => candidate.slug === post.slug);
  const previous = currentIndex > 0 ? siblings[currentIndex - 1] : null;
  const next = currentIndex < siblings.length - 1 ? siblings[currentIndex + 1] : null;

  const item = (direction, target, fallback) => {
    const label = direction === 'previous' ? 'Previous' : 'Next';
    const title = target ? sectionHeading(target)?.value || target.metadata.title : fallback;
    const content = `<span class="series-pagination__label">${label}</span>
        <span class="series-pagination__title">${escapeHtml(title)}</span>`;
    if (!target) {
      return `      <span class="series-pagination__item series-pagination__item--${direction} is-disabled" aria-disabled="true">
        ${content}
      </span>`;
    }
    return `      <a class="series-pagination__item series-pagination__item--${direction}" href="${escapeHtml(postHref(target))}">
        ${content}
      </a>`;
  };

  return `    <nav class="series-pagination" aria-label="Series navigation">
${item('previous', previous, 'Start of series')}
${item('next', next, 'More coming soon')}
    </nav>`;
}

function renderPost(post, posts, index, buildTimestamp) {
  const { metadata } = post;
  const nodes = articleNodes(post);
  const previews = new Map();
  const renderOptions = {
    buildTimestamp,
    timeZone: metadata.timezone || 'Asia/Kolkata',
  };

  const articleHtml = renderNodes(nodes, {
    ...renderOptions,
    resolveReference: createReferenceResolver({ post, index, previews, options: renderOptions }),
  });
  const previewMarkup = previews.size > 0
    ? `      <div class="xref-previews" hidden>\n${[...previews.values()].join('\n')}\n      </div>\n`
    : '';
  const tableOfContents = renderTableOfContents(post, nodes, posts);
  const pagination = renderSeriesPagination(post, posts);
  const section = sectionHeading(post);
  const canonical = `https://harish-shankar.github.io/blog/${post.slug}`;
  const documentTitle = metadata.series && section
    ? `${section.value} — ${metadata.title}`
    : metadata.title;

  return `${pageHead({ title: documentTitle, description: metadata.description, canonical })}
<body class="article-page">
  <main class="article-shell">
${tableOfContents}
    <article class="article">
      <a class="back-link" href="/blog">← All writing</a>
      <header class="article-header">
        <h1>${escapeHtml(metadata.title)}</h1>
        <time class="article-date" datetime="${escapeHtml(metadata.date)}">${escapeHtml(formatPostDate(metadata.date))}</time>
      </header>
      <div class="markdown-body">
${articleHtml}
      </div>
${previewMarkup}${pagination}
    </article>
  </main>
  <script src="/blog/blog.js?v=${encodeURIComponent(buildTimestamp)}"></script>
</body>
</html>
`;
}

function renderBlogIndex(posts) {
  const firstBySeries = new Map();
  for (const post of posts) {
    if (!post.metadata.series) continue;
    const current = firstBySeries.get(post.metadata.series);
    if (!current || seriesOrder(post) < seriesOrder(current)) firstBySeries.set(post.metadata.series, post);
  }
  const visiblePosts = posts.filter((post) =>
    !post.metadata.series || firstBySeries.get(post.metadata.series)?.slug === post.slug);
  visiblePosts.sort((left, right) =>
    (left.metadata.status === 'reference' ? 1 : 0) - (right.metadata.status === 'reference' ? 1 : 0) ||
    right.metadata.date.localeCompare(left.metadata.date) ||
    left.metadata.title.localeCompare(right.metadata.title));
  const entries = visiblePosts.map(({ metadata, slug }) => `
        <li>
          <a href="/blog/${escapeHtml(slug)}">
            <span>${escapeHtml(metadata.title)}</span>
          </a>
          ${metadata.description ? `<p>${escapeHtml(metadata.description)}</p>` : ''}
        </li>`).join('');

  return `${pageHead({
    title: 'Writing',
    description: 'Essays and notes by Harish Shankar.',
    canonical: 'https://harish-shankar.github.io/blog',
    type: 'website',
  })}
<body class="blog-index-page">
  ${navigation()}
  <main class="blog-index">
    <header>
      <h1>Writing</h1>
    </header>
    <ol class="post-list">${entries}
    </ol>
  </main>
</body>
</html>
`;
}

function build() {
  const requestedBuildTimestamp = process.env.BUILD_TIMESTAMP || new Date().toISOString();
  const parsedBuildTimestamp = new Date(requestedBuildTimestamp);
  if (Number.isNaN(parsedBuildTimestamp.getTime())) throw new Error('BUILD_TIMESTAMP is invalid.');
  const buildTimestamp = parsedBuildTimestamp.toISOString();
  const posts = markdownFiles(POSTS_DIRECTORY).map((sourcePath) => {
    const filename = path.relative(POSTS_DIRECTORY, sourcePath).split(path.sep).join('/');
    const originalSource = fs.readFileSync(sourcePath, 'utf8');
    const source = originalSource
      .replace(/^([ \t]*):::progress[ \t]*$/gm, `$1:::progress ${buildTimestamp}`)
      .replace(/:::progress[ \t]+:::/g, `:::progress ${buildTimestamp} :::`);
    if (source !== originalSource) fs.writeFileSync(sourcePath, source);
    const parsed = parseMarkdown(source);
    const defaultSlug = filename.replace(/\.md$/i, '');
    const slug = (parsed.metadata.permalink || defaultSlug).replace(/^\/+|\/+$/g, '');
    if (!/^[a-z0-9][a-z0-9/_-]*$/.test(slug) || slug.includes('..')) {
      throw new Error(`${filename} has an invalid permalink.`);
    }
    if (!parsed.metadata.title || !parsed.metadata.date) {
      throw new Error(`${filename} needs both title and date in its front matter.`);
    }
    return { ...parsed, slug, filename };
  }).sort((left, right) =>
    right.metadata.date.localeCompare(left.metadata.date) ||
    left.metadata.title.localeCompare(right.metadata.title));

  const seriesGroups = new Map();
  for (const post of posts) {
    if (!post.metadata.series) continue;
    const h2Headings = post.nodes.filter((node) => node.type === 'heading' && node.level === 2);
    if (h2Headings.length !== 1) {
      throw new Error(`${post.filename} must contain exactly one h2 heading because it belongs to a series.`);
    }
    if (!Number.isInteger(Number(post.metadata.series_order)) || Number(post.metadata.series_order) < 1) {
      throw new Error(`${post.filename} needs a positive integer series_order.`);
    }
    const group = seriesGroups.get(post.metadata.series) || [];
    group.push(post);
    seriesGroups.set(post.metadata.series, group);
  }
  for (const [series, group] of seriesGroups) {
    const orders = group.map(seriesOrder);
    if (new Set(orders).size !== orders.length) {
      throw new Error(`Series ${series} contains duplicate series_order values.`);
    }
  }

  const anchorIndex = buildAnchorIndex(posts);

  fs.mkdirSync(BLOG_DIRECTORY, { recursive: true });
  const katexRoot = path.dirname(require.resolve('katex/package.json'));
  const katexDestination = path.join(BLOG_DIRECTORY, 'vendor', 'katex');
  fs.mkdirSync(katexDestination, { recursive: true });
  fs.copyFileSync(path.join(katexRoot, 'dist', 'katex.min.css'), path.join(katexDestination, 'katex.min.css'));
  fs.cpSync(path.join(katexRoot, 'dist', 'fonts'), path.join(katexDestination, 'fonts'), { recursive: true });
  for (const post of posts) {
    const destination = path.join(BLOG_DIRECTORY, post.slug);
    fs.mkdirSync(destination, { recursive: true });
    fs.writeFileSync(path.join(destination, 'index.html'), renderPost(post, posts, anchorIndex, buildTimestamp));
  }
  fs.writeFileSync(path.join(BLOG_DIRECTORY, 'index.html'), renderBlogIndex(posts));
  process.stdout.write(`Built ${posts.length} post${posts.length === 1 ? '' : 's'}.\n`);
}

build();

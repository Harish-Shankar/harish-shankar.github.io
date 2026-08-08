'use strict';

const fs = require('node:fs');
const path = require('node:path');
const {
  escapeHtml,
  parseMarkdown,
  renderNodes,
} = require('../lib/markdown');

const ROOT = path.resolve(__dirname, '..');
const POSTS_DIRECTORY = path.join(ROOT, 'posts');
const BLOG_DIRECTORY = path.join(ROOT, 'blog');

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
  <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@1,500;1,600&amp;family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&amp;family=Spline+Sans+Mono:wght@500;600&amp;display=swap" rel="stylesheet">
  <link rel="stylesheet" href="/styles.css">
  <link rel="stylesheet" href="/blog/vendor/katex/katex.min.css">
  <link rel="stylesheet" href="/blog/blog.css">
</head>`;
}

// Article chrome is intentionally minimal. Authors own every visible heading in
// Markdown; do not add category labels, dates, status rows, or the full site nav.
function renderPost(post, buildTimestamp) {
  const { metadata } = post;
  const nodes = [...post.nodes];
  if (nodes[0]?.type === 'heading' && nodes[0].level === 1 && nodes[0].value.trim() === metadata.title?.trim()) {
    nodes.shift();
  }

  const articleHtml = renderNodes(nodes, {
    buildTimestamp,
    timeZone: metadata.timezone || 'Asia/Kolkata',
  });
  const canonical = `https://harish-shankar.github.io/blog/${post.slug}`;

  return `${pageHead({ title: metadata.title, description: metadata.description, canonical })}
<body class="article-page">
  ${navigation()}
  <main class="article-shell">
    <a class="back-link" href="/blog">← All writing</a>
    <article class="article">
      <header class="article-header">
        <h1>${escapeHtml(metadata.title)}</h1>
        ${metadata.description ? `<p class="article-deck">${escapeHtml(metadata.description)}</p>` : ''}
      </header>
      <nav class="section-marker" aria-label="Current article section" hidden>
        <span class="section-marker__level" aria-hidden="true"></span>
        <a class="section-marker__link" href="#"></a>
      </nav>
      <div class="markdown-body">
${articleHtml}
      </div>
    </article>
  </main>
  <script src="/blog/blog.js"></script>
</body>
</html>
`;
}

function renderBlogIndex(posts) {
  const entries = posts.map(({ metadata, slug }) => `
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
  const filenames = fs.readdirSync(POSTS_DIRECTORY)
    .filter((filename) => filename.endsWith('.md') && !filename.startsWith('_'))
    .sort();

  const posts = filenames.map((filename) => {
    const sourcePath = path.join(POSTS_DIRECTORY, filename);
    const originalSource = fs.readFileSync(sourcePath, 'utf8');
    const source = originalSource.replace(/^([ \t]*):::progress[ \t]*$/gm, `$1:::progress ${buildTimestamp}`);
    if (source !== originalSource) fs.writeFileSync(sourcePath, source);
    const parsed = parseMarkdown(source);
    const slug = path.basename(filename, '.md');
    if (!parsed.metadata.title || !parsed.metadata.date) {
      throw new Error(`${filename} needs both title and date in its front matter.`);
    }
    return { ...parsed, slug };
  }).sort((left, right) =>
    right.metadata.date.localeCompare(left.metadata.date) ||
    left.metadata.title.localeCompare(right.metadata.title));

  fs.mkdirSync(BLOG_DIRECTORY, { recursive: true });
  const katexRoot = path.dirname(require.resolve('katex/package.json'));
  const katexDestination = path.join(BLOG_DIRECTORY, 'vendor', 'katex');
  fs.mkdirSync(katexDestination, { recursive: true });
  fs.copyFileSync(path.join(katexRoot, 'dist', 'katex.min.css'), path.join(katexDestination, 'katex.min.css'));
  fs.cpSync(path.join(katexRoot, 'dist', 'fonts'), path.join(katexDestination, 'fonts'), { recursive: true });
  for (const post of posts) {
    const destination = path.join(BLOG_DIRECTORY, post.slug);
    fs.mkdirSync(destination, { recursive: true });
    fs.writeFileSync(path.join(destination, 'index.html'), renderPost(post, buildTimestamp));
  }
  fs.writeFileSync(path.join(BLOG_DIRECTORY, 'index.html'), renderBlogIndex(posts));
  process.stdout.write(`Built ${posts.length} post${posts.length === 1 ? '' : 's'}.\n`);
}

build();

'use strict';

(() => {
  const links = [...document.querySelectorAll('.article-toc a')];
  const headings = links
    .map((link) => document.getElementById(link.hash.slice(1)))
    .filter(Boolean);

  if (links.length === 0 || headings.length === 0) return;

  let activeHeading = null;
  let frameRequested = false;

  function updateTableOfContents() {
    const activationLine = 96;
    let nextHeading = headings[0];

    for (const heading of headings) {
      if (heading.getBoundingClientRect().top > activationLine) break;
      nextHeading = heading;
    }

    if (nextHeading && nextHeading !== activeHeading) {
      for (const link of links) {
        const isActive = link.hash === `#${nextHeading.id}`;
        link.classList.toggle('is-active', isActive);
        if (isActive) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      }
      activeHeading = nextHeading;
    }

    frameRequested = false;
  }

  function requestUpdate() {
    if (frameRequested) return;
    frameRequested = true;
    window.requestAnimationFrame(updateTableOfContents);
  }

  window.addEventListener('scroll', requestUpdate, { passive: true });
  window.addEventListener('resize', requestUpdate);
  updateTableOfContents();
})();

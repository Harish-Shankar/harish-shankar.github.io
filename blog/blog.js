'use strict';

(() => {
  const pageLink = document.querySelector('.article-toc a[data-page-link][aria-controls]');
  const subsectionList = pageLink
    ? document.getElementById(pageLink.getAttribute('aria-controls'))
    : null;

  if (pageLink && subsectionList) {
    pageLink.addEventListener('click', (event) => {
      event.preventDefault();
      const shouldExpand = pageLink.getAttribute('aria-expanded') !== 'true';
      pageLink.setAttribute('aria-expanded', String(shouldExpand));
      subsectionList.hidden = !shouldExpand;
    });
  }

  const links = [...document.querySelectorAll('.article-toc a[data-local-section]')];
  const headings = links
    .map((link) => document.getElementById(link.hash.slice(1)))
    .filter(Boolean);

  if (links.length === 0 || headings.length === 0) return;

  let activeHeading;
  let frameRequested = false;

  function updateTableOfContents() {
    const activationLine = 96;
    let nextHeading = null;

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
    } else if (!nextHeading && activeHeading !== null) {
      for (const link of links) {
        link.classList.remove('is-active');
        link.removeAttribute('aria-current');
      }
      activeHeading = null;
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

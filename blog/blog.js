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

/* Cross-reference previews: a passage from elsewhere in the series, shown in place. */
(() => {
  const links = [...document.querySelectorAll('a.xref[data-xref-preview]')];
  if (links.length === 0) return;

  const OPEN_DELAY = 130;
  const CLOSE_DELAY = 220;
  const GAP = 10;
  const EDGE = 12;
  const MINIMUM_HEIGHT = 170;

  const popover = document.createElement('div');
  popover.className = 'xref-popover';
  popover.id = 'xref-popover';
  popover.setAttribute('role', 'note');
  popover.hidden = true;
  document.body.append(popover);

  let activeLink = null;
  let openTimer = 0;
  let closeTimer = 0;
  let frameRequested = false;

  const finePointer = () => window.matchMedia('(hover: hover) and (pointer: fine)').matches;

  function anchorRect(link, event) {
    const rects = [...link.getClientRects()];
    if (rects.length === 0) return link.getBoundingClientRect();
    if (!event) return rects[0];
    return rects.find((rect) => event.clientY >= rect.top && event.clientY <= rect.bottom) || rects[0];
  }

  function position(rect) {
    popover.style.maxHeight = '';
    const spaceBelow = window.innerHeight - rect.bottom - GAP - EDGE;
    const spaceAbove = rect.top - GAP - EDGE;
    const below = spaceBelow >= Math.min(popover.offsetHeight, MINIMUM_HEIGHT) || spaceBelow >= spaceAbove;

    popover.style.maxHeight = `${Math.max(MINIMUM_HEIGHT, Math.round(below ? spaceBelow : spaceAbove))}px`;
    popover.dataset.placement = below ? 'below' : 'above';

    const width = popover.offsetWidth;
    const height = popover.offsetHeight;
    const left = Math.min(
      Math.max(EDGE, rect.left + rect.width / 2 - width / 2),
      Math.max(EDGE, window.innerWidth - width - EDGE),
    );
    const top = below ? rect.bottom + GAP : Math.max(EDGE, rect.top - height - GAP);

    popover.style.left = `${Math.round(left)}px`;
    popover.style.top = `${Math.round(top)}px`;
  }

  function open(link, event) {
    const source = document.getElementById(link.dataset.xrefPreview);
    if (!source) return;

    if (activeLink !== link) {
      const content = source.cloneNode(true);
      content.removeAttribute('id');
      for (const element of content.querySelectorAll('[id]')) element.removeAttribute('id');
      popover.replaceChildren(content);
      if (activeLink) {
        activeLink.removeAttribute('aria-describedby');
        activeLink.classList.remove('is-previewed');
      }
      activeLink = link;
      link.setAttribute('aria-describedby', popover.id);
      link.classList.add('is-previewed');
    }

    popover.hidden = false;
    position(anchorRect(link, event));
    popover.classList.add('is-open');
  }

  function close() {
    window.clearTimeout(openTimer);
    window.clearTimeout(closeTimer);
    if (!activeLink) return;
    activeLink.removeAttribute('aria-describedby');
    activeLink.classList.remove('is-previewed');
    activeLink = null;
    popover.classList.remove('is-open');
    popover.hidden = true;
    popover.replaceChildren();
  }

  function scheduleOpen(link, event) {
    window.clearTimeout(closeTimer);
    window.clearTimeout(openTimer);
    openTimer = window.setTimeout(() => open(link, event), OPEN_DELAY);
  }

  function scheduleClose() {
    window.clearTimeout(openTimer);
    window.clearTimeout(closeTimer);
    closeTimer = window.setTimeout(close, CLOSE_DELAY);
  }

  function reposition() {
    frameRequested = false;
    if (!activeLink) return;
    const rect = anchorRect(activeLink);
    if (rect.bottom < 0 || rect.top > window.innerHeight) {
      close();
      return;
    }
    position(rect);
  }

  function requestReposition() {
    if (!activeLink || frameRequested) return;
    frameRequested = true;
    window.requestAnimationFrame(reposition);
  }

  for (const link of links) {
    link.addEventListener('pointerenter', (event) => {
      if (event.pointerType !== 'mouse') return;
      scheduleOpen(link, event);
    });
    link.addEventListener('pointerleave', (event) => {
      if (event.pointerType !== 'mouse') return;
      scheduleClose();
    });
    link.addEventListener('focus', () => {
      window.clearTimeout(closeTimer);
      open(link);
    });
    link.addEventListener('blur', () => {
      if (!popover.contains(document.activeElement)) scheduleClose();
    });
    /* Without a hovering pointer the first tap opens the preview; the preview itself links onward. */
    link.addEventListener('click', (event) => {
      if (finePointer()) return;
      if (activeLink === link && !popover.hidden) return;
      event.preventDefault();
      open(link);
    });
  }

  popover.addEventListener('pointerenter', () => window.clearTimeout(closeTimer));
  popover.addEventListener('pointerleave', scheduleClose);

  document.addEventListener('pointerdown', (event) => {
    if (!activeLink) return;
    if (popover.contains(event.target) || activeLink.contains(event.target)) return;
    close();
  });

  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape' || !activeLink) return;
    const link = activeLink;
    close();
    link.focus();
  });

  window.addEventListener('scroll', requestReposition, { passive: true });
  window.addEventListener('resize', requestReposition);
})();

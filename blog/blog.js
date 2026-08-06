'use strict';

(() => {
  const marker = document.querySelector('.section-marker');
  const headings = [...document.querySelectorAll('.markdown-body h2, .markdown-body h3')];

  if (!marker || headings.length === 0) return;

  const level = marker.querySelector('.section-marker__level');
  const link = marker.querySelector('.section-marker__link');
  let activeHeading = null;
  let frameRequested = false;

  marker.hidden = false;

  function updateMarker() {
    const activationLine = 60;
    let nextHeading = null;

    for (const heading of headings) {
      if (heading.getBoundingClientRect().top > activationLine) break;
      nextHeading = heading;
    }

    marker.classList.toggle('is-visible', Boolean(nextHeading));

    if (nextHeading && nextHeading !== activeHeading) {
      const parentSection = nextHeading.matches('h3')
        ? headings.slice(0, headings.indexOf(nextHeading)).reverse().find((heading) => heading.matches('h2'))
        : null;

      level.textContent = nextHeading.matches('h2') ? 'SECTION' : 'SUBSECTION';
      link.textContent = parentSection
        ? `${parentSection.textContent} · ${nextHeading.textContent}`
        : nextHeading.textContent;
      link.href = `#${nextHeading.id}`;
      link.title = link.textContent;
      activeHeading = nextHeading;
    }

    frameRequested = false;
  }

  function requestUpdate() {
    if (frameRequested) return;
    frameRequested = true;
    window.requestAnimationFrame(updateMarker);
  }

  window.addEventListener('scroll', requestUpdate, { passive: true });
  window.addEventListener('resize', requestUpdate);
  updateMarker();
})();

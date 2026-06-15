// KaTeX rendering for MkDocs Material + pymdownx.arithmatex (generic mode).
// arithmatex wraps math in \(...\) (inline) and \[...\] (display) inside
// elements with class .arithmatex. We re-render after every instant-nav load.
document$.subscribe(() => {
  renderMathInElement(document.body, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
    ],
    // Don't choke on a stray dollar sign in prose/code.
    throwOnError: false,
    ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
  });
});

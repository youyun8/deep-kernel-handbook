# Contributing / How to extend this site

Thanks for wanting to extend the ML Perf Handbook. This guide explains the
structure so you can add a page or a code example without guesswork.

## Repository layout

```
ml-perf-handbook/
├── mkdocs.yml              # site config + the entire nav tree
├── requirements.txt       # docs toolchain (mkdocs-material, extensions)
├── docs/                   # all written content (Markdown)
│   ├── index.md            # landing page
│   ├── reading-path.md
│   ├── foundations/        # Part I
│   ├── moe/                # Part II (flagship)
│   ├── performance/        # Part III
│   ├── capstones/          # Part IV
│   ├── glossary.md
│   ├── references.md
│   ├── stylesheets/extra.css
│   └── javascripts/katex.js
├── code/                   # runnable, tested reference implementations
│   ├── requirements.txt    # heavier deps (torch, triton, pytest)
│   ├── moe/ attention/ kernels/
│   └── tests run with pytest
└── .github/workflows/deploy.yml
```

## Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdocs serve            # live-reload preview at http://127.0.0.1:8000
mkdocs build --strict   # exactly what CI runs; fails on broken links/refs
```

## Adding a new page

1. Create `docs/<section>/<slug>.md`.
2. Register it in the `nav:` tree in `mkdocs.yml` (pages not in `nav`
   trigger a warning under `--strict`).
3. Start the page with the standard header block (see any existing page):
   a `# Title`, then a `<div class="page-meta">` chip row declaring **level**
   and **prerequisites**, then a one-paragraph "what you'll get" summary.
4. End every page with three `##` sections: **Key takeaways**,
   **Exercises**, and **References** (link primary papers).

## House style

- **Intuition first, then math, then code, then systems.** Don't hand-wave the
  hard parts — if a step is subtle (online softmax rescaling, all-to-all
  bucketing), show the algebra or the array shapes.
- **Math** uses KaTeX via `$...$` (inline) and `$$...$$` (display).
- **Code** lives in fenced blocks with a language tag so it gets highlighting
  and a copy button. Prefer extracting non-trivial code into `code/` and
  linking to it, so it can be tested.
- **Diagrams**: prefer Mermaid (` ```mermaid `) or inline SVG so they live in
  version control and render without a build step.
- **Numbers**: when you quote a speedup, say what hardware and shapes produced
  it, or label it "representative" and give the methodology to reproduce.

## Adding runnable code

- Put it under `code/<topic>/`, add a docstring explaining how to run it.
- Add or extend a `test_*.py` next to it. Reference implementations should be
  checked against PyTorch (`torch.allclose`) where possible.
- CPU-only tests must pass without a GPU. Gate GPU/Triton tests with
  `pytest.importorskip("triton")` and `torch.cuda.is_available()`.

## Marking unfinished work

If you scaffold a page, put this right under the title so nothing looks more
finished than it is:

```markdown
!!! warning "部分實作"
    This page is an outline. The sections below are stubs.
```

## Commit conventions

Use conventional-commit subjects (`feat:`, `fix:`, `docs:`, `refactor:`,
`test:`, `chore:`) and capitalized bullet points in the body.

# LaTeX Conventions for this Project

> **Required rules for editing `main.tex`, chapters, or any `.tex`/`.sty` file in this project.**
> Each rule documents a bug that has actually occurred in this codebase.
> Reading this before editing will save you (and the next maintainer) hours of debugging.

---

## 1. Page numbers

| Goal | CORRECT syntax | WRONG syntax (causes bugs) |
|------|----------------|-----------------------------|
| Normal page numbering, number in footer-center (default) | `\pagestyle{plain}` (set once in `preamble.sty`) | — |
| Hide page number on **one specific** page (e.g. cover, GVHD review) | `\thispagestyle{empty}` (local to one page, safe) | `\let\@oddfoot\@empty` |
| Switch numbering style (Roman ↔ Arabic) | `\pagenumbering{Roman}` / `\pagenumbering{arabic}` (in `main.tex` only) | Placing it inside a chapter file |

### Why `\let\@oddfoot\@empty` is FORBIDDEN

- This `\let` mutates the LaTeX-kernel macro `\@oddfoot` **permanently**.
- Every page **after** the one where you wrote it loses its footer (page number, running header rule, anything that lives in `\@oddfoot`).
- There is no clean way to undo it within the same document — you must fix the source and rebuild from scratch.
- This bug actually happened in `front/pages/GVHDNhanXet.tex` on **2026-09-13** and was the cause of "missing page numbers from Chapter 1 onward". Use `\thispagestyle{empty}` instead, which is local to one page.

---

## 2. Captions containing `\imagesourcemark` (image source marker)

| Goal | CORRECT syntax | WRONG syntax |
|------|----------------|--------------|
| Figure with a source footnote (superscript "1" marker in the figure, real footnote at page bottom) | `\caption[short caption]{full caption \protect\imagesourcemark}` | `\caption{full caption \protect\imagesourcemark}` |
| Figure WITHOUT a source footnote | `\caption{full caption}` | — |

### Why the optional `[short]` argument is required

- `\listoffigures` (the "List of Figures" / "Danh mục hình vẽ") writes the **entire** caption text into the `.lof` file, including the superscript "1" emitted by `\imagesourcemark`.
- Without `[short]`, the "1" appears glued to every line in the List of Figures — looks terrible.
- The `[short]` form gives you **two** captions:
  - `[short]` → used in the List of Figures (no marker)
  - `{long}` → used in the actual figure (with marker, looks correct in the figure)
- This bug actually happened on all **10 figures in `Chuong02_CoSoLyThuyet.tex`** on **2026-09-13** and was fixed in one pass.

---

## 3. Counters and numbering format

| Goal | Correct location |
|------|------------------|
| `\setcounter{page}{1}` | Inside `main.tex`, immediately after `\pagenumbering{...}` |
| `\thispagestyle{empty}` | Inside the `.tex` file of **the specific page** you want to hide |
| `\pagestyle{plain}` / `\pagestyle{fancy}` | Inside `preamble.sty`, exactly once |
| `\pagenumbering{...}` | Inside `main.tex` only — never in a chapter |

---

## 4. Figures and tables

- Every `\begin{figure}` must have `\label{fig:...}` **immediately after** `\caption` (not before, not in a different order).
- Cross-reference figures/tables with `\cref{fig:...}` (loaded via `cleveref`) or `\autoref{...}`. **Do not** use bare `\ref{...}` — it gives just a number with no "Figure"/"Table" prefix.
- Prefer placement specifier `[H]` (provided by `float` package) for figures/tables inside long chapters — `[t]`/`[b]` will float them to potentially distant pages and break the visual flow.

---

## 5. Page styles and headers

- The default page style for this document is **`plain`**: page number centered at the bottom, no header.
- If you later need a fancy header (chapter name on the left, section name on the right, page number centered at the bottom), change **one** line in `preamble.sty`:
  ```latex
  \pagestyle{fancy}
  \fancyhf{}
  \fancyfoot[C]{\thepage}
  \fancyhead[L]{\leftmark}
  ```
  Do **not** sprinkle `\pagestyle{...}` calls throughout the chapters.

---

## 6. File headers (every new `.tex` / `.sty` file)

Every new file must start with a descriptive header comment:

```latex
% =====================================================================
%  Filename.tex — Short description of what this file does
%  (added/edited YYYY-MM-DD by <author>, reason: <reason>)
% =====================================================================
```

This makes `git log -p` and file-by-file diffs immediately readable.

---

## 7. General LaTeX hygiene

- **Always** run the full 4-step build cycle (`./build.sh`) after non-trivial changes — single-pass `pdflatex` will leave stale `.aux` / `.toc` files and you'll see wrong references.
- **Never** edit `main.pdf` directly — it is a generated artifact.
- **Never** commit `.aux`, `.toc`, `.out`, `.log`, `.bbl`, `.blg` — they are regenerated on every build.
- When in doubt about a LaTeX command, **read this file first**, then check `preamble.sty` for project-level custom commands, then ask.

# Design System Master File — MIPT Normative Desk

**Project:** Нормативная база МФТИ (MIPT Normative & Archival Reference Desk)
**Aesthetic Direction:** Swiss / Modernist Academic & Editorial Reference Desk
**Stack:** Semantic HTML5 + Three-Layer CSS Custom Properties + Vanilla JS (Zero build step)

---

## 1. Studio Anti-Slop Mandate

- **Zero AI-chatbot clichés:** No purple/indigo gradients (`#6366f1`, `#7c3aed`), no 4-point AI sparkles, no right-aligned iMessage bubbles, no centered "Чем могу помочь?" splash screen, no fake example prompt chips, no glowing pill chat bar.
- **Zero template typography tells:** No `Inter` everywhere, no tracked-out ALL-CAPS eyebrow labels, no middle-dot (`A · B · C`) metadata soup, no `→` appended to button labels.
- **One memorable focal signature:** **Archival Excerpt Plates (`.doc-quote`)** — crisp folio sheets with a solid Phystech Cobalt left spine, set in **Minsk** typography to distinguish verbatim legal text cleanly, paired with emerald verification seals and expandable full-clause context with warm highlighter-ochre `<mark>` passages.

---

## 2. Three-Layer Token Summary

### Layer 1: Primitives (Raw Palette)

| Token | Light Mode Value | Dark Blueprint Value | Role |
|-------|------------------|----------------------|------|
| `--prim-paper-100` / `--prim-night-950` | `#F6F7F9` | `#0D131F` | Desk canvas |
| `--prim-paper-0` / `--prim-night-900` | `#FFFFFF` | `#131C2E` | Folio sheet / elevated plate |
| `--prim-paper-200` / `--prim-night-850` | `#EEF1F5` | `#18243A` | Subtle control / recessed ledger |
| `--prim-paper-300` / `--prim-night-700` | `#E3E8EF` | `#263754` | Active press state (`:active`) |
| `--prim-ink-900` / `--prim-night-50` | `#101828` | `#F1F5F9` | Primary slate/chalk ink (>= 15:1) |
| `--prim-ink-700` / `--prim-night-200` | `#344054` | `#CBD5E1` | Secondary prose ink (>= 9.5:1) |
| `--prim-ink-600` / `--prim-night-400` | `#475467` | `#94A3B8` | Muted metadata ink (>= 5.4:1) |
| `--prim-cobalt-700` / `--prim-cobalt-300` | `#0B4A8C` | `#6BAEFF` | Phystech Cobalt accent & spine |
| `--prim-cobalt-800` / `--prim-cobalt-500` | `#08386B` | `#1D68BD` | Solid primary hover (>= 5.4:1 w/ `#FFFFFF`) |
| `--prim-emerald-700` / `--prim-emerald-300` | `#0A6644` | `#4ADE9B` | Archival verification seal |
| `--prim-ochre-200` / `--prim-ochre-dark-bg` | `#FDE68A` | `rgba(234, 179, 8, 0.26)` | Clause highlight `<mark>` |
| `--prim-amber-600` / `--prim-amber-200` | `#D97706` | `#FDE68A` | Duplicate-upload & non-fatal warning states (`--color-warning-*`) |
| `--prim-crimson-700` / `--prim-crimson-300` | `#B42318` | `#FDA29B` | Destructive / error ink |

### Layer 2: Typography Roles

- **`--font-sans` & `--font-serif` (`Minsk`, fallback `Golos Text`)**: Primary typeface across UI controls, dossier headings, analytical prose (`max-width: 68ch`, `line-height: 1.62`), and verbatim normative quotes (`/static/fonts/ofont.ru_Minsk.ttf` with `unicode-range` excluding empty `@`/`№` slots so symbols fall back to `Golos Text`).
- **`--font-mono` (`JetBrains Mono`)**: Tabular numerals and archival identifiers (`№ 5990-1`, `п. 2.5`, `стр. 3`, dates, page counters, zoom percentages, latency, file format codes `PDF`/`DOCX`/`DOC`/`XLSX`/`XLS`).

### Layer 3: Component Tokens

Every UI component (`--masthead-*`, `--history-*`, `--composer-*`, `--dossier-*`, `--ans-*`, `--quote-*`, `--source-*`, `--viewer-*`, `--ledger-*`, `--upload-*`, `--url-upload-*`, `--btn-*`, `--dialog-*`) references Layer 2 semantic tokens exclusively. Zero raw hex values appear outside Layer 1.

### Workspace Split Architecture & v8 Components

- **Left Query History Rail (`--history-width: 272px`)**: Docked alongside `#/` on desktop (`> 1024px`, collapsible via masthead toggle; auto-collapses `< 1280px` when the Document Viewer opens) and transitions into a modal slide-over drawer with scrim on `<= 1024px`.
- **Structured Grounded Dossier Answers (`.ans-structured`)**:
  - **Framed Short Answer (`.ans-brief-frame`)**: Always-visible top editorial plate with a solid `--color-accent` architectural left rule, `КРАТКИЙ ОТВЕТ` micro-label, and crisp elevated surface (`--ans-brief-bg`).
  - **Conditional Action Plate (`.ans-actions-box`)**: Rendered right below `.ans-brief-frame` strictly when procedural steps (`### Порядок действий`) exist; completely omitted on factual/numeric queries.
  - **Full Details & Verbatim Quotes (`.ans-details-wrap`)**: Detailed normative analysis (`### Подробный ответ и цитаты`), Markdown tables, and `.doc-quote` plates stay open by default behind a collapsible `.ans-toggle-details-btn` with a live quote-count badge (`Цитат: N`) and streaming progress indicator.
  - **Calm Technical Error Stub (`.ans-error-frame`)**: Replaces raw tracebacks or 5xx/network exceptions with a polite Russian sysadmin notice (`TECH_ERROR_STUB_RU`) and one-click retry.
- **Right Side-by-Side Document Viewer (`--viewer-width: min(48vw, 680px)`)**: Docked right split panel on desktop (`> 768px`) and full-viewport archival sheet on mobile (`<= 768px`), rendering real paginated A4 page images (`.dv-pdf-page`) with paper shadow, zoom controls, and clean target-page scroll for `PDF`, `DOCX`, and `DOC` (`has_page_images: true`), plus structured `XLSX`/`XLS` sheet tables (`.dv-xlsx-table`).
- **Fixed-Track Knowledge Base Ledger (`--ledger-columns: 60px minmax(200px, 1fr) 152px 76px 116px 156px`)**: Exact 6-track CSS Grid shared identically between `.ledger-head` and `.doc` rows (`.doc-cell-fmt`, `.doc-main`, `.doc-req`, `.doc-pages`, `.doc-cell-status`, `.doc-row-actions`) so every header aligns pixel-for-pixel with its column, paired with amber duplicate-rejection upload rows (`.upload.duplicate` + `.toast.warn`).

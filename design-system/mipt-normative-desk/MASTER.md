# Design System Master File — MIPT Normative Desk

**Project:** Нормативная база МФТИ (MIPT Normative & Archival Reference Desk)
**Aesthetic Direction:** Swiss / Modernist Academic & Editorial Reference Desk
**Stack:** Semantic HTML5 + Three-Layer CSS Custom Properties + Vanilla JS (Zero build step)

---

## 1. Studio Anti-Slop Mandate

- **Zero AI-chatbot clichés:** No purple/indigo gradients (`#6366f1`, `#7c3aed`), no 4-point AI sparkles, no right-aligned iMessage bubbles, no centered "Чем могу помочь?" splash screen, no fake example prompt chips, no glowing pill chat bar.
- **Zero template typography tells:** No `Inter` everywhere, no tracked-out ALL-CAPS eyebrow labels, no middle-dot (`A · B · C`) metadata soup, no `→` appended to button labels.
- **One memorable focal signature:** **Archival Excerpt Plates (`.doc-quote`)** — crisp folio sheets with a solid Phystech Cobalt left spine, set in optical-size **Literata** serif to distinguish verbatim legal text from analytical prose, paired with emerald verification seals and expandable full-clause context with warm highlighter-ochre `<mark>` passages.

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
| `--prim-crimson-700` / `--prim-crimson-300` | `#B42318` | `#FDA29B` | Destructive / error ink |

### Layer 2: Typography Roles

- **`--font-sans` (`Golos Text`)**: Russian modernist neo-grotesque for UI controls, dossier headings, and analytical prose (`max-width: 68ch`, `line-height: 1.62`).
- **`--font-serif` (`Literata`)**: Optical-size Cyrillic book serif strictly for verbatim normative quotes, Word archival folio prose, and expanded full-clause excerpts (`line-height: 1.68`).
- **`--font-mono` (`JetBrains Mono`)**: Tabular numerals and archival identifiers (`№ 5990-1`, `п. 2.5`, `стр. 3`, dates, page counters, zoom percentages, latency, file format codes `PDF`/`DOCX`/`DOC`/`XLSX`/`XLS`).

### Layer 3: Component Tokens

Every UI component (`--masthead-*`, `--history-*`, `--composer-*`, `--dossier-*`, `--quote-*`, `--source-*`, `--viewer-*`, `--ledger-*`, `--upload-*`, `--url-upload-*`, `--btn-*`, `--dialog-*`) references Layer 2 semantic tokens exclusively. Zero raw hex values appear outside Layer 1.

### Workspace Split Architecture (`v6`)

- **Left Query History Rail (`--history-width: 272px`)**: Docked alongside `#/` on desktop (`> 1024px`, collapsible via masthead toggle; auto-collapses `< 1280px` when the Document Viewer opens) and transitions into a modal slide-over drawer with scrim on `<= 1024px`.
- **Right Side-by-Side Document Viewer (`--viewer-width: min(48vw, 680px)`)**: Docked right split panel on desktop (`> 768px`) and full-viewport archival sheet on mobile (`<= 768px`), rendering clean, authentic PDF page frames and structured Word/Excel folios navigated straight to the target page or sheet.

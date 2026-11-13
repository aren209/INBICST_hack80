/* ==========================================================================
   Нормативная база МФТИ — Frontend Application (Vanilla JS, Zero Build Step)
   Features & Preserved Mechanics:
     - Flicker-free incremental DOM morphing during POST /api/ask/stream SSE
     - AbortController Stop button + prioritized global Escape handling
     - Automatic POST /api/ask fallback on 404/405
     - Placeholder stripping ([___], б/н, без номера)
     - Verbatim quote verification seals + expandable clause_text with <mark>
     - Left Query History Sidebar (localStorage: mipt_rag_threads_v1, max 50)
     - Right Side-by-Side Document Viewer Drawer (PDF exact page + 0..1 ochre
       highlights, Word/Excel archival folios + sheet tabs + <mark> highlights)
     - Drag-and-drop upload (PDF, DOCX, DOC, XLSX, XLS) + /api/tasks/{id} polling
     - Accessible confirmation modal for DELETE /api/documents/{doc_id} & Clear
   ========================================================================== */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  };

  const ICONS = {
    chev: '<svg class="chev" viewBox="0 0 24 24" aria-hidden="true"><path d="M6 9l6 6 6-6"/></svg>',
    doc: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 3.5H7A1.5 1.5 0 0 0 5.5 5v14A1.5 1.5 0 0 0 7 20.5h10a1.5 1.5 0 0 0 1.5-1.5V8L14 3.5Z"/><path d="M13.5 3.5V8.5H18.5"/></svg>',
    openDoc:
      '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 3.5H7A1.5 1.5 0 0 0 5.5 5v14A1.5 1.5 0 0 0 7 20.5h10a1.5 1.5 0 0 0 1.5-1.5V8L14 3.5Z"/><path d="M13.5 3.5V8.5H18.5"/><path d="M9 13h6M9 16.5h4"/></svg>',
    copy: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="9" width="11" height="11" rx="1.8"/><path d="M5 15V5.8A1.8 1.8 0 0 1 6.8 4H16"/></svg>',
    check: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>',
    retry: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 12a8 8 0 0 1 14-5.3L20 9M20 4v5h-5M20 12a8 8 0 0 1-14 5.3L4 15M4 20v-5h5"/></svg>',
    shield: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3l7 2.8v5.4c0 4.5-3 8.3-7 9.8-4-1.5-7-5.3-7-9.8V5.8L12 3Z"/><path d="M8.8 12.2l2.2 2.2 4.5-4.6"/></svg>',
    alert: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8.8"/><path d="M12 7.8v4.8M12 16h.01"/></svg>',
    trash: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4.5 7h15M9.8 11v5.5M14.2 11v5.5M6.5 7l.8 11.8A1.7 1.7 0 0 0 9 20.5h6a1.7 1.7 0 0 0 1.7-1.7L17.5 7M9.5 7V4.5h5V7"/></svg>',
    close: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6.5 6.5l11 11M17.5 6.5l-11 11"/></svg>',
    stop: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="6.5" y="6.5" width="11" height="11" rx="1.8"/></svg>',
    warn: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 9v4M12 16.8h.01M10.4 4.2L2.6 18a1.8 1.8 0 0 0 1.6 2.7h15.6A1.8 1.8 0 0 0 21.4 18L13.6 4.2a1.8 1.8 0 0 0-3.2 0Z"/></svg>',
  };

  /* ---------------- Helpers ---------------- */
  function plural(n, one, few, many) {
    const m10 = n % 10;
    const m100 = n % 100;
    if (m10 === 1 && m100 !== 11) return one;
    if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few;
    return many;
  }

  function fmtDateNumeric(s) {
    if (!s) return "";
    const iso = String(s).match(/^(\d{4})-(\d{2})-(\d{2})/);
    if (iso) return `${iso[3]}.${iso[2]}.${iso[1]}`;
    const d = new Date(s);
    if (isNaN(d)) return String(s);
    const dd = String(d.getDate()).padStart(2, "0");
    const mm = String(d.getMonth() + 1).padStart(2, "0");
    return `${dd}.${mm}.${d.getFullYear()}`;
  }

  function fmtHistoryDate(isoStr) {
    if (!isoStr) return "";
    const d = new Date(isoStr);
    if (isNaN(d)) return "";
    const now = new Date();
    const hh = String(d.getHours()).padStart(2, "0");
    const min = String(d.getMinutes()).padStart(2, "0");
    const sameDay =
      d.getFullYear() === now.getFullYear() &&
      d.getMonth() === now.getMonth() &&
      d.getDate() === now.getDate();
    if (sameDay) return `сегодня, ${hh}:${min}`;

    const yest = new Date(now);
    yest.setDate(now.getDate() - 1);
    const isYest =
      d.getFullYear() === yest.getFullYear() &&
      d.getMonth() === yest.getMonth() &&
      d.getDate() === yest.getDate();
    if (isYest) return `вчера, ${hh}:${min}`;

    const dd = String(d.getDate()).padStart(2, "0");
    const mm = String(d.getMonth() + 1).padStart(2, "0");
    return `${dd}.${mm}.${d.getFullYear()}`;
  }

  /* ---------------- API ---------------- */
  async function api(path, opts = {}) {
    const res = await fetch(path, {
      ...opts,
      headers:
        opts.body && !(opts.body instanceof FormData)
          ? { "Content-Type": "application/json", ...(opts.headers || {}) }
          : opts.headers,
    });
    let data = null;
    try {
      data = await res.json();
    } catch {
      /* non-JSON response */
    }
    if (!res.ok) {
      const detail =
        data &&
        (typeof data.detail === "string"
          ? data.detail
          : data.detail
            ? JSON.stringify(data.detail)
            : null);
      const err = new Error(detail || `Ошибка сервера (${res.status})`);
      err.status = res.status;
      throw err;
    }
    return data;
  }

  /* ---------------- Toasts & Shared Confirm Dialog ---------------- */
  function toast(message, type = "info", ms = 4200) {
    const t = el("div", `toast ${type}`, message);
    $("#toasts").appendChild(t);
    setTimeout(() => {
      t.classList.add("out");
      setTimeout(() => t.remove(), 220);
    }, ms);
  }

  function openConfirmDialog({ title, text, okLabel }) {
    return new Promise((resolve) => {
      const dlg = $("#confirmDialog");
      $("#confirmTitle").textContent = title;
      $("#confirmText").textContent = text;
      const okBtn = $("#confirmOkBtn");
      if (okBtn) okBtn.textContent = okLabel || "Подтвердить";
      dlg.returnValue = "";
      dlg.showModal();
      dlg.addEventListener(
        "close",
        function onClose() {
          dlg.removeEventListener("close", onClose);
          resolve(dlg.returnValue === "ok");
        },
        { once: true }
      );
    });
  }

  /* ---------------- Markdown & Flicker-Free Morphing ---------------- */
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[c]));
  }

  function markdownHtml(text) {
    const src = String(text || "");
    if (window.marked && window.DOMPurify) {
      return window.DOMPurify.sanitize(window.marked.parse(src, { gfm: true, breaks: false }));
    }
    return src
      .split(/\n{2,}/)
      .map(
        (p) =>
          "<p>" +
          escapeHtml(p)
            .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
            .replace(/\n/g, "<br>") +
          "</p>"
      )
      .join("");
  }

  function fillMarkdown(box, text) {
    box.innerHTML = markdownHtml(text);
    box.querySelectorAll("table").forEach((t) => {
      const wrap = el("div", "table-wrap");
      t.replaceWith(wrap);
      wrap.appendChild(t);
    });
    box.querySelectorAll("a[href]").forEach((a) => {
      a.target = "_blank";
      a.rel = "noopener noreferrer";
    });
    decorateQuotes(box);
    return box;
  }

  function syncAttrs(a, b) {
    for (const { name } of [...a.attributes]) {
      if (!b.hasAttribute(name)) a.removeAttribute(name);
    }
    for (const { name, value } of [...b.attributes]) {
      if (a.getAttribute(name) !== value) a.setAttribute(name, value);
    }
  }

  function morph(from, to) {
    const a = [...from.childNodes];
    const b = [...to.childNodes];
    for (let i = 0; i < b.length; i++) {
      const na = a[i];
      const nb = b[i];
      if (!na) {
        from.appendChild(nb);
        continue;
      }
      if (na.nodeType !== nb.nodeType || (na.nodeType === 1 && na.tagName !== nb.tagName)) {
        from.replaceChild(nb, na);
        continue;
      }
      if (na.nodeType !== 1) {
        if (na.nodeValue !== nb.nodeValue) na.nodeValue = nb.nodeValue;
        continue;
      }
      syncAttrs(na, nb);
      morph(na, nb);
    }
    for (let i = b.length; i < a.length; i++) a[i].remove();
  }

  const BLOCK_TAGS = new Set(["P", "UL", "OL", "LI", "BLOCKQUOTE", "H1", "H2", "H3", "H4", "H5", "H6"]);
  function placeCaret(box) {
    let target = box;
    while (target.lastElementChild && BLOCK_TAGS.has(target.lastElementChild.tagName)) {
      target = target.lastElementChild;
    }
    target.appendChild(el("span", "caret"));
  }

  function patchMarkdown(box, text, withCaret) {
    const next = fillMarkdown(el("div"), text);
    if (withCaret) placeCaret(next);
    morph(box, next);
    return box;
  }

  function renderMarkdown(text) {
    return fillMarkdown(el("div", "md"), text);
  }

  /* ---------------- Verbatim Document Excerpt Plates ---------------- */
  const OPEN_Q = '«“„"';

  /** Strip placeholder numbers («№ [___]», «№ б/н», «без номера», «___») from citations. */
  const cleanSource = (s) =>
    String(s || "")
      .replace(/\s*(?:№|(?<![A-Za-zА-Яа-яЁё])N)\s*(?:\[[\s_.…-]*\]|_{2,}|б\s*\/\s*н\.?|без\s+номера)/gi, "")
      .replace(/\[\s*_+\s*\]|_{3,}/g, "")
      .replace(/\s*,(?:\s*,)+/g, ",")
      .replace(/\s{2,}/g, " ")
      .replace(/^[\s,;·|—–-]+|[\s,;·|]+$/g, "");

  const normQ = (s) =>
    String(s || "")
      .toLowerCase()
      .replace(/ё/g, "е")
      .replace(/[^0-9a-zа-я]+/g, " ")
      .trim();

  function cleanDocNo(raw) {
    const s = cleanSource(raw || "");
    if (!s || /^(?:б\s*\/\s*н\.?|без\s+номера)$/i.test(s)) return "";
    return s;
  }

  function splitQuote(raw) {
    let text = String(raw || "").replace(/\u00a0/g, " ").trim();
    let source = "";
    const m =
      text.match(/\n\s*[—–-]{1,2}\s*([^\n]+)\s*$/) ||
      text.match(/[»”"]\s*[—–]\s*([^\n«»]+)$/);
    if (m) {
      source = cleanSource(m[1]);
      text = text.slice(0, m.index + (m[0].startsWith("\n") ? 0 : 1)).trim();
    }
    if (OPEN_Q.includes(text[0])) text = text.slice(1);
    const close = Math.max(text.lastIndexOf("»"), text.lastIndexOf("”"), text.lastIndexOf('"'));
    if (close >= 0 && close >= text.length - 3) text = text.slice(0, close);
    return { text: text.replace(/\s*\n\s*/g, " ").trim(), source };
  }

  function decorateQuotes(box) {
    let idx = 0;
    box.querySelectorAll("blockquote").forEach((bq) => {
      if (bq.closest(".doc-quote") || bq.parentElement.closest("blockquote")) return;
      const raw = bq.textContent.replace(/^\s+/, "");
      if (!raw || !OPEN_Q.includes(raw[0])) return;
      const joined = Array.from(bq.querySelectorAll("p"))
        .map((p) => p.textContent)
        .join("\n");
      const { text, source } = splitQuote(joined || raw);
      if (!text) return;

      const fig = el("figure", "doc-quote");
      fig.dataset.qidx = String(idx++);
      fig.dataset.qnorm = normQ(text);

      const body = el("blockquote", "dq-text", text);
      fig.appendChild(body);

      if (source) {
        const cap = el("figcaption", "dq-foot");
        const src = el("button", "dq-src");
        src.type = "button";
        src.title = "Перейти к источнику в перечне";
        src.innerHTML = ICONS.doc;
        src.appendChild(el("span", null, source));
        cap.appendChild(src);
        fig.appendChild(cap);
      }
      bq.replaceWith(fig);
    });
  }

  /** Build full-clause node with verbatim quoted spans wrapped in <mark>. */
  function clauseNode(q) {
    const box = el("div", "dq-clause");
    const t = String(q.clause_text || "");
    const hl = (Array.isArray(q.clause_highlights) ? q.clause_highlights : [])
      .filter((r) => Array.isArray(r) && r[0] >= 0 && r[1] > r[0] && r[1] <= t.length)
      .sort((x, y) => x[0] - y[0]);
    let pos = 0;
    for (const [a, b] of hl) {
      if (a < pos) continue;
      if (a > pos) box.appendChild(document.createTextNode(t.slice(pos, a)));
      box.appendChild(el("mark", null, t.slice(a, b)));
      pos = b;
    }
    if (pos < t.length) box.appendChild(document.createTextNode(t.slice(pos)));
    return box;
  }

  function parseFirstPageInt(val) {
    if (val == null) return null;
    if (typeof val === "number" && Number.isFinite(val) && val >= 1) return Math.floor(val);
    const m = String(val).match(/\d+/);
    if (!m) return null;
    const n = parseInt(m[0], 10);
    return Number.isFinite(n) && n >= 1 ? n : null;
  }

  function parsePageFromCitationLabel(label) {
    if (!label) return null;
    const m = String(label).match(/(?:стр\.|с\.|лист)\s*(\d+)/i);
    if (!m) return null;
    const n = parseInt(m[1], 10);
    return Number.isFinite(n) && n >= 1 ? n : null;
  }

  function matchBestSource(q, sources, labelText) {
    const lbl = normQ(labelText || (q && (q.source_label || q.source_title)) || "");
    const qDocId = (q && (q.doc_id || q.matched_doc_id)) || "";
    const qDocNum = normQ(q && q.doc_number);
    const qTitle = normQ(q && q.source_title).slice(0, 32);
    const clause = q && q.clause ? String(q.clause).replace(/\.$/, "") : "";

    let best = null;
    let bestSc = 0;
    for (const s of sources || []) {
      let sc = 0;
      if (qDocId && s.docId && s.docId === qDocId) sc += 6;
      if (qDocNum && s.docNumber && normQ(s.docNumber) === qDocNum) sc += 4;
      else if (s.docNumber && lbl.includes(normQ(s.docNumber))) sc += 3;
      const nm = normQ(s.name).slice(0, 30);
      if (nm && (lbl.includes(nm) || (qTitle && (nm.includes(qTitle) || qTitle.includes(nm))))) {
        sc += 3;
      }
      if (clause && s.clause && s.clause.replace(/\.$/, "").endsWith(" " + clause)) sc += 1;
      if (sc > bestSc) {
        best = s;
        bestSc = sc;
      }
    }
    return best;
  }

  function resolveQuoteViewerTarget(q, sources, labelText, cardQuoteText) {
    const bestSrc = matchBestSource(q, sources, labelText);
    const docId =
      (q && (q.doc_id || q.matched_doc_id)) || (bestSrc && bestSrc.docId) || "";
    const page =
      (q && parseFirstPageInt(q.page_num)) ||
      (q && parseFirstPageInt(q.page)) ||
      parsePageFromCitationLabel(labelText) ||
      (bestSrc && Array.isArray(bestSrc.pages) && bestSrc.pages[0]) ||
      null;
    const clause =
      (q && q.clause) ||
      (bestSrc && bestSrc.rawClause) ||
      "";
    const quote = (q && q.text) || cardQuoteText || "";
    const docNumber = (q && q.doc_number) || (bestSrc && bestSrc.docNumber) || "";
    const fallbackTitle =
      (q && q.source_title) || (bestSrc && bestSrc.name) || labelText || "";
    return { docId, page, quote, clause, docNumber, fallbackTitle };
  }

  /** Attach verification seals, normalized source metadata, full-clause disclosure & Document Viewer trigger. */
  function applyQuoteMeta(box, quotes, sources, sourcesApi) {
    const cards = [...box.querySelectorAll(".doc-quote")];
    if (!cards.length) return;
    const list = (quotes || []).map((q) => ({ ...q, _n: normQ(q.text), _used: false }));

    cards.forEach((card, i) => {
      const n = card.dataset.qnorm;
      const q =
        list.find((x) => !x._used && x._n === n) ||
        list.find(
          (x) =>
            !x._used &&
            n &&
            x._n &&
            (x._n.startsWith(n.slice(0, 40)) || n.startsWith(x._n.slice(0, 40)))
        ) ||
        (list[i] && !list[i]._used ? list[i] : null);

      let foot = card.querySelector(".dq-foot");
      if (!foot) {
        foot = el("figcaption", "dq-foot");
        card.appendChild(foot);
      }
      let btn = card.querySelector(".dq-src");

      let actionsWrap = foot.querySelector(".dq-foot-actions");
      if (!actionsWrap) {
        actionsWrap = el("div", "dq-foot-actions");
        foot.appendChild(actionsWrap);
      }

      if (q) {
        q._used = true;
        const label = cleanSource(q.source_label || "").replace(
          /(№|пп?\.|стр\.)\s+(?=\S)/g,
          "$1\u00a0"
        );
        if (label) {
          if (!btn) {
            btn = el("button", "dq-src");
            btn.type = "button";
            btn.title = "Перейти к документу в перечне источников";
            btn.innerHTML = ICONS.doc;
            btn.appendChild(el("span"));
            foot.prepend(btn);
          }
          const span = btn.querySelector("span");
          if (span.textContent !== label) span.textContent = label;
        }

        if (!actionsWrap.querySelector(".dq-badge")) {
          const badge = el("span", `dq-badge ${q.verified ? "ok" : "warn"}`);
          badge.innerHTML = q.verified ? ICONS.shield : ICONS.alert;
          badge.appendChild(
            el("span", null, q.verified ? "Проверено в документе" : "Не найдено дословно")
          );
          badge.title = q.verified
            ? "Выдержка сверена с текстом проиндексированного документа"
            : "Дословное совпадение не найдено в фрагментах — сверьтесь с оригиналом";
          actionsWrap.appendChild(badge);
        }

        card.classList.add(q.verified ? "is-verified" : "is-unverified");

        const ct = String(q.clause_text || "");
        if (
          q.verified &&
          ct &&
          normQ(ct).length > normQ(q.text).length + 25 &&
          !card.querySelector(".dq-expand")
        ) {
          const full = clauseNode(q);
          full.hidden = true;
          const quoteEl = card.querySelector(".dq-text");
          quoteEl.after(full);

          const tg = el("button", "dq-expand");
          tg.type = "button";
          tg.setAttribute("aria-expanded", "false");
          const closedLabel = q.clause
            ? `Показать пункт ${q.clause} целиком`
            : "Показать пункт целиком";
          const lbl = el("span", null, closedLabel);
          tg.append(lbl);
          tg.insertAdjacentHTML("beforeend", ICONS.chev);
          tg.addEventListener("click", () => {
            const open = full.hidden;
            full.hidden = !open;
            quoteEl.hidden = open;
            card.classList.toggle("is-expanded", open);
            tg.setAttribute("aria-expanded", String(open));
            lbl.textContent = open ? "Свернуть до цитаты" : closedLabel;
          });
          full.after(tg);
        }
      }

      const cardQuoteText = (card.querySelector(".dq-text") || {}).textContent || "";
      const labelText = (btn && btn.textContent) || "";
      const target = resolveQuoteViewerTarget(q, sources, labelText, cardQuoteText);

      if (!actionsWrap.querySelector(".dq-open-doc") && (target.docId || target.fallbackTitle)) {
        const openBtn = el("button", "dq-open-doc");
        openBtn.type = "button";
        openBtn.innerHTML = ICONS.openDoc;
        openBtn.appendChild(el("span", null, "Открыть документ"));
        if (target.page) {
          openBtn.appendChild(el("span", "dq-open-page", `стр.\u00a0${target.page}`));
        }
        openBtn.title = target.page
          ? `Открыть документ на стр. ${target.page}`
          : "Открыть документ в боковой панели";
        openBtn.addEventListener("click", () => {
          openDocumentViewer(target);
        });
        actionsWrap.appendChild(openBtn);
      }

      if (btn) {
        if (sourcesApi) {
          btn.addEventListener("click", () => sourcesApi.reveal(q, btn.textContent));
        } else if (target.docId || target.fallbackTitle) {
          btn.title = "Открыть документ в боковой панели";
          btn.addEventListener("click", () => openDocumentViewer(target));
        } else {
          btn.disabled = true;
        }
      }
    });
  }

  /* ---------------- Theme Toggle ---------------- */
  const appShell = $("#appShell");
  const themeBtn = $("#themeBtn");
  function currentTheme() {
    const t = document.documentElement.dataset.theme;
    if (t) return t;
    return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function syncThemeAria() {
    const isDark = currentTheme() === "dark";
    themeBtn.setAttribute(
      "aria-label",
      isDark ? "Включить светлую тему" : "Включить тёмную тему"
    );
  }

  themeBtn.addEventListener("click", () => {
    const next = currentTheme() === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("theme", next);
    } catch {
      /* storage blocked */
    }
    syncThemeAria();
  });
  syncThemeAria();

  /* ==========================================================================
     LEFT QUERY HISTORY SIDEBAR (localStorage: mipt_rag_threads_v1)
     ========================================================================== */
  const THREADS_KEY = "mipt_rag_threads_v1";
  const SIDEBAR_PREF_KEY = "mipt_rag_sidebar_open_v1";
  const MAX_THREADS = 50;

  const historyRail = $("#historyRail");
  const historyToggleBtn = $("#historyToggleBtn");
  const historyCloseBtn = $("#historyCloseBtn");
  const historyClearBtn = $("#historyClearBtn");
  const historyBackdrop = $("#historyBackdrop");
  const historyList = $("#historyList");
  const historyEmpty = $("#historyEmpty");

  let threads = loadThreads();
  let activeThreadId = null;
  let desktopSidebarOpen = loadSidebarPref();
  let mobileSidebarOpen = false;
  let autoCollapsedByViewer = false;

  function loadSidebarPref() {
    try {
      const v = localStorage.getItem(SIDEBAR_PREF_KEY);
      if (v === "0") return false;
      if (v === "1") return true;
    } catch {
      /* ignore */
    }
    return true;
  }

  function saveSidebarPref(open) {
    try {
      localStorage.setItem(SIDEBAR_PREF_KEY, open ? "1" : "0");
    } catch {
      /* ignore */
    }
  }

  function loadThreads() {
    try {
      const raw = localStorage.getItem(THREADS_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      return parsed
        .filter((t) => t && typeof t.id === "string" && Array.isArray(t.turns))
        .slice(0, MAX_THREADS);
    } catch {
      return [];
    }
  }

  function saveThreads() {
    threads = threads.slice(0, MAX_THREADS);
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        localStorage.setItem(THREADS_KEY, JSON.stringify(threads));
        return;
      } catch {
        if (threads.length > 5) {
          threads = threads.slice(0, Math.max(5, Math.floor(threads.length * 0.7)));
        } else {
          break;
        }
      }
    }
  }

  function sanitizeQuotesForStorage(quotes) {
    return (quotes || []).map((q) => ({
      text: q.text || "",
      source_title: q.source_title || "",
      source_label: q.source_label || "",
      doc_id: q.doc_id || q.matched_doc_id || "",
      matched_doc_id: q.matched_doc_id || q.doc_id || "",
      doc_number: q.doc_number || "",
      clause: q.clause || "",
      page: q.page != null ? q.page : "",
      page_num: q.page_num != null ? q.page_num : null,
      verified: Boolean(q.verified),
      clause_text: String(q.clause_text || "").slice(0, 1800),
      clause_highlights: Array.isArray(q.clause_highlights) ? q.clause_highlights : [],
    }));
  }

  function makeThreadTitle(query) {
    const clean = String(query || "").replace(/\s+/g, " ").trim();
    if (!clean) return "Новый запрос";
    return clean.length > 78 ? clean.slice(0, 77).trimEnd() + "…" : clean;
  }

  function isCompactScreen() {
    return window.innerWidth <= 1024;
  }

  function isHistoryOpen() {
    return isCompactScreen() ? mobileSidebarOpen : desktopSidebarOpen;
  }

  function syncHistoryUi() {
    const isKb = location.hash === "#/kb";
    if (isKb) {
      historyToggleBtn.hidden = true;
      historyRail.hidden = true;
      historyBackdrop.hidden = true;
      return;
    }
    historyToggleBtn.hidden = false;
    historyRail.hidden = false;
    const open = isHistoryOpen();
    historyRail.classList.toggle("is-collapsed", !open);
    historyBackdrop.hidden = !(isCompactScreen() && open);
    historyToggleBtn.setAttribute("aria-expanded", String(open));
    historyToggleBtn.setAttribute(
      "aria-label",
      open ? "Скрыть историю запросов" : "Показать историю запросов"
    );
  }

  function toggleHistorySidebar() {
    if (isCompactScreen()) {
      mobileSidebarOpen = !mobileSidebarOpen;
    } else {
      desktopSidebarOpen = !desktopSidebarOpen;
      autoCollapsedByViewer = false;
      saveSidebarPref(desktopSidebarOpen);
    }
    syncHistoryUi();
  }

  function closeMobileHistory() {
    if (mobileSidebarOpen) {
      mobileSidebarOpen = false;
      syncHistoryUi();
    }
  }

  historyToggleBtn.addEventListener("click", toggleHistorySidebar);
  historyCloseBtn.addEventListener("click", closeMobileHistory);
  historyBackdrop.addEventListener("click", closeMobileHistory);

  function renderHistoryList() {
    const saved = threads.filter((t) => Array.isArray(t.turns) && t.turns.length > 0);
    // Also show an in-flight active thread at top if it was just created
    const visible = threads.filter(
      (t) => (Array.isArray(t.turns) && t.turns.length > 0) || t.id === activeThreadId
    );

    historyEmpty.hidden = visible.length > 0;
    historyClearBtn.hidden = saved.length === 0;
    historyList.innerHTML = "";

    const frag = document.createDocumentFragment();
    for (const thr of visible) {
      const li = el("li", "history-item" + (thr.id === activeThreadId ? " is-active" : ""));
      li.dataset.threadId = thr.id;

      const selBtn = el("button", "history-select");
      selBtn.type = "button";
      selBtn.appendChild(el("span", "history-item-title", thr.title || "Запрос"));

      const metaParts = [fmtHistoryDate(thr.updatedAt || thr.createdAt)];
      if (thr.turns && thr.turns.length > 1) {
        metaParts.push(`${thr.turns.length} ${plural(thr.turns.length, "запрос", "запроса", "запросов")}`);
      }
      selBtn.appendChild(el("span", "history-item-meta", metaParts.filter(Boolean).join(" · ").replace(" · ", ", ")));
      selBtn.addEventListener("click", () => selectThread(thr.id));

      const delBtn = el("button", "history-del");
      delBtn.type = "button";
      delBtn.setAttribute("aria-label", `Удалить запрос «${thr.title}» из истории`);
      delBtn.title = "Удалить из истории";
      delBtn.innerHTML = ICONS.close;
      delBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        deleteThread(thr.id);
      });

      li.append(selBtn, delBtn);
      frag.appendChild(li);
    }
    historyList.appendChild(frag);
  }

  function ensureActiveThread(firstQuery) {
    if (activeThreadId) {
      const existing = threads.find((t) => t.id === activeThreadId);
      if (existing) return existing;
    }
    const now = new Date().toISOString();
    const thr = {
      id: "thr_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 7),
      title: makeThreadTitle(firstQuery),
      createdAt: now,
      updatedAt: now,
      turns: [],
    };
    threads.unshift(thr);
    activeThreadId = thr.id;
    renderHistoryList();
    return thr;
  }

  function recordCompletedTurn(threadId, turn) {
    const idx = threads.findIndex((t) => t.id === threadId);
    if (idx < 0) return;
    const thr = threads[idx];
    thr.turns.push(turn);
    thr.updatedAt = new Date().toISOString();
    if (!thr.title || thr.turns.length === 1) {
      thr.title = makeThreadTitle(turn.query);
    }
    if (idx > 0) {
      threads.splice(idx, 1);
      threads.unshift(thr);
    }
    saveThreads();
    renderHistoryList();
  }

  function discardEmptyThread(threadId) {
    const thr = threads.find((t) => t.id === threadId);
    if (thr && (!thr.turns || thr.turns.length === 0)) {
      threads = threads.filter((t) => t.id !== threadId);
      if (activeThreadId === threadId) activeThreadId = null;
      saveThreads();
      renderHistoryList();
    }
  }

  function selectThread(threadId) {
    const thr = threads.find((t) => t.id === threadId);
    if (!thr) return;

    chatEpoch++;
    if (controller) controller.abort();
    controller = null;
    setBusy(false);
    closeMobileHistory();

    if (location.hash === "#/kb") {
      location.hash = "#/";
    }

    activeThreadId = thr.id;
    queryCount = thr.turns.length;
    messages.innerHTML = "";

    if (thr.turns.length > 0) {
      syncDeskState(true);
      thr.turns.forEach((turn, idx) => {
        addUserMessage(turn.query, idx + 1);
        const { body } = addBotShell();
        renderAnswer(body, {
          answer: turn.markdown || "",
          sources: Array.isArray(turn.sources) ? turn.sources : [],
          quotes: Array.isArray(turn.quotes) ? turn.quotes : [],
          latencyMs: turn.latencySec ? Number(turn.latencySec) * 1000 : 0,
          note: turn.stopped ? "Генерация остановлена" : null,
          noteKind: "stop",
        });
      });
      requestAnimationFrame(() => {
        chatScroll.scrollTop = 0;
      });
    } else {
      syncDeskState(false);
    }

    renderHistoryList();
  }

  function deleteThread(threadId) {
    threads = threads.filter((t) => t.id !== threadId);
    saveThreads();
    if (activeThreadId === threadId) {
      chatEpoch++;
      if (controller) controller.abort();
      controller = null;
      setBusy(false);
      activeThreadId = null;
      queryCount = 0;
      messages.innerHTML = "";
      syncDeskState(false);
    }
    renderHistoryList();
  }

  historyClearBtn.addEventListener("click", async () => {
    if (!threads.length) return;
    const ok = await openConfirmDialog({
      title: "Очистить историю запросов?",
      text: "Все сохранённые запросы в этом браузере будут удалены из боковой панели.",
      okLabel: "Очистить историю",
    });
    if (!ok) return;
    threads = [];
    saveThreads();
    activeThreadId = null;
    queryCount = 0;
    if (!busy) {
      messages.innerHTML = "";
      syncDeskState(false);
    }
    renderHistoryList();
    toast("История запросов очищена");
  });

  window.addEventListener(
    "resize",
    () => {
      syncHistoryUi();
    },
    { passive: true }
  );

  /* ==========================================================================
     RIGHT SIDE-BY-SIDE DOCUMENT VIEWER DRAWER
     ========================================================================== */
  const docViewer = $("#docViewer");
  const dvFormatBadge = $("#dvFormatBadge");
  const dvTitle = $("#dvTitle");
  const dvMeta = $("#dvMeta");
  const dvCloseBtn = $("#dvCloseBtn");
  const dvPrevBtn = $("#dvPrevBtn");
  const dvNextBtn = $("#dvNextBtn");
  const dvPagePrefix = $("#dvPagePrefix");
  const dvPageInput = $("#dvPageInput");
  const dvPageTotal = $("#dvPageTotal");
  const dvZoomOut = $("#dvZoomOut");
  const dvZoomReset = $("#dvZoomReset");
  const dvZoomIn = $("#dvZoomIn");
  const dvDownloadLink = $("#dvDownloadLink");
  const dvSheets = $("#dvSheets");
  const dvBody = $("#dvBody");

  const ZOOM_STEPS = [75, 90, 100, 115, 130, 150, 175, 200];

  const viewerState = {
    open: false,
    reqId: 0,
    docId: "",
    format: "pdf",
    totalPages: 1,
    currentPage: 1,
    zoomIdx: 2, // 100%
    programmaticScrollUntil: 0,
  };

  function applyViewerZoom() {
    const pct = ZOOM_STEPS[viewerState.zoomIdx] || 100;
    dvZoomReset.textContent = `${pct}%`;
    dvZoomOut.disabled = viewerState.zoomIdx <= 0;
    dvZoomIn.disabled = viewerState.zoomIdx >= ZOOM_STEPS.length - 1;
    dvBody.style.setProperty("--dv-zoom-pct", `${pct}%`);
    dvBody.style.setProperty("--dv-zoom-factor", String(pct / 100));
  }

  function syncViewerNavControls(pageNum) {
    const p = Math.max(1, Math.min(viewerState.totalPages, Number(pageNum) || 1));
    viewerState.currentPage = p;
    if (document.activeElement !== dvPageInput) {
      dvPageInput.value = String(p);
    }
    dvPrevBtn.disabled = p <= 1;
    dvNextBtn.disabled = p >= viewerState.totalPages;
    dvSheets.querySelectorAll(".dv-sheet-tab").forEach((tab) => {
      const active = Number(tab.dataset.page) === p;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", String(active));
    });
  }

  function yWithinBody(node) {
    const bodyRect = dvBody.getBoundingClientRect();
    const nodeRect = node.getBoundingClientRect();
    return nodeRect.top - bodyRect.top + dvBody.scrollTop;
  }

  function scrollToViewerPage(pageNum, { instant = false, focusHighlight = false } = {}) {
    const targetNum = Math.max(1, Math.min(viewerState.totalPages, Number(pageNum) || 1));
    syncViewerNavControls(targetNum);

    dvBody.querySelectorAll(".dv-page-card, .dv-folio-sheet").forEach((elNode) => {
      elNode.classList.toggle("is-target", Number(elNode.dataset.page) === targetNum);
    });

    const pageEl = dvBody.querySelector(`[data-page="${targetNum}"]`);
    if (!pageEl) return;

    const pageTop = yWithinBody(pageEl);
    let desiredTop = Math.max(0, pageTop - 10);

    if (focusHighlight) {
      const hlTarget =
        pageEl.querySelector(".dv-pdf-hl") ||
        pageEl.querySelector("mark.dv-mark") ||
        pageEl.querySelector(".is-clause-match") ||
        dvBody.querySelector("mark.dv-mark");
      if (hlTarget) {
        const hlY = yWithinBody(hlTarget);
        const hlBottom = hlY + (hlTarget.getBoundingClientRect().height || 24);
        if (hlBottom - desiredTop > dvBody.clientHeight * 0.78 || !pageEl.contains(hlTarget)) {
          desiredTop = Math.max(0, hlY - dvBody.clientHeight * 0.28);
        }
      }
    }

    viewerState.programmaticScrollUntil = Date.now() + (instant ? 150 : 500);
    dvBody.scrollTo({
      top: desiredTop,
      behavior: instant ? "instant" : "smooth",
    });
  }

  function closeDocumentViewer() {
    if (!viewerState.open) return;
    viewerState.open = false;
    viewerState.reqId++;
    docViewer.hidden = true;
    appShell.classList.remove("has-viewer");
    if (autoCollapsedByViewer && window.innerWidth >= 1025) {
      desktopSidebarOpen = true;
      autoCollapsedByViewer = false;
      syncHistoryUi();
    }
  }

  dvCloseBtn.addEventListener("click", closeDocumentViewer);

  dvPrevBtn.addEventListener("click", () => {
    if (viewerState.currentPage > 1) {
      scrollToViewerPage(viewerState.currentPage - 1, { instant: false });
    }
  });

  dvNextBtn.addEventListener("click", () => {
    if (viewerState.currentPage < viewerState.totalPages) {
      scrollToViewerPage(viewerState.currentPage + 1, { instant: false });
    }
  });

  const commitPageInput = () => {
    const val = parseInt(dvPageInput.value, 10);
    if (Number.isFinite(val)) {
      const clamped = Math.max(1, Math.min(viewerState.totalPages, val));
      dvPageInput.value = String(clamped);
      scrollToViewerPage(clamped, { instant: false });
    } else {
      dvPageInput.value = String(viewerState.currentPage);
    }
  };

  dvPageInput.addEventListener("change", commitPageInput);
  dvPageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      commitPageInput();
      dvPageInput.blur();
    }
  });

  dvZoomOut.addEventListener("click", () => {
    if (viewerState.zoomIdx > 0) {
      viewerState.zoomIdx--;
      applyViewerZoom();
    }
  });

  dvZoomIn.addEventListener("click", () => {
    if (viewerState.zoomIdx < ZOOM_STEPS.length - 1) {
      viewerState.zoomIdx++;
      applyViewerZoom();
    }
  });

  dvZoomReset.addEventListener("click", () => {
    viewerState.zoomIdx = 2;
    applyViewerZoom();
  });

  let dvScrollRaf = 0;
  dvBody.addEventListener(
    "scroll",
    () => {
      if (Date.now() < viewerState.programmaticScrollUntil) return;
      if (dvScrollRaf) return;
      dvScrollRaf = requestAnimationFrame(() => {
        dvScrollRaf = 0;
        const probeY = dvBody.scrollTop + dvBody.clientHeight * 0.3;
        let current = viewerState.currentPage;
        dvBody.querySelectorAll("[data-page]").forEach((frame) => {
          if (frame.offsetTop <= probeY) {
            current = Number(frame.dataset.page) || current;
          }
        });
        if (current !== viewerState.currentPage) {
          syncViewerNavControls(current);
        }
      });
    },
    { passive: true }
  );

  /* ---- Normalized Substring Highlighter for Word / Excel Blocks ---- */
  function buildQuoteNeedles(quote) {
    const raw = String(quote || "").trim();
    if (!raw) return [];
    const parts = raw
      .split(/\s*(?:\.{3,}|…|\[\s*(?:\.{3}|…)\s*\])\s*/)
      .map((s) => normMatchString(s).norm)
      .filter((s) => s.length >= 8);

    const needles = [...parts];
    for (const p of parts) {
      const words = p.split(" ").filter(Boolean);
      if (words.length > 9) {
        const head = words.slice(0, 7).join(" ");
        const tail = words.slice(-7).join(" ");
        if (head.length >= 18) needles.push(head);
        if (tail.length >= 18) needles.push(tail);
      }
    }
    return needles;
  }

  /** Fold case, ё->е, soft hyphens, and collapse whitespace/punctuation runs while mapping to original indices. */
  function normMatchString(str) {
    const src = String(str || "");
    const chars = [];
    const starts = [];
    const ends = [];
    let prevSpace = true;

    for (let i = 0; i < src.length; i++) {
      const ch = src[i];
      if (ch === "\u00ad") continue;
      const folded = ch.toLowerCase().replace("ё", "е");
      if (/[0-9a-zа-я]/i.test(folded)) {
        chars.push(folded);
        starts.push(i);
        ends.push(i + 1);
        prevSpace = false;
      } else if (!prevSpace && chars.length > 0) {
        chars.push(" ");
        starts.push(i);
        ends.push(i + 1);
        prevSpace = true;
      }
    }
    if (chars[chars.length - 1] === " ") {
      chars.pop();
      starts.pop();
      ends.pop();
    }
    return { norm: chars.join(""), starts, ends };
  }

  function appendHighlightedText(container, text, needles) {
    const raw = String(text || "");
    if (!raw) return false;
    if (!needles || !needles.length) {
      container.appendChild(document.createTextNode(raw));
      return false;
    }

    const mapped = normMatchString(raw);
    if (!mapped.norm) {
      container.appendChild(document.createTextNode(raw));
      return false;
    }

    const ranges = [];
    for (const needle of needles) {
      let from = 0;
      let found = false;
      while (from <= mapped.norm.length - needle.length) {
        const idx = mapped.norm.indexOf(needle, from);
        if (idx < 0) break;
        const origStart = mapped.starts[idx];
        const origEnd = mapped.ends[idx + needle.length - 1];
        ranges.push([origStart, origEnd]);
        found = true;
        from = idx + needle.length;
      }
      // If a full segment matched, skip shorter fallback anchors
      if (found && needle === needles[0]) break;
    }

    if (!ranges.length) {
      container.appendChild(document.createTextNode(raw));
      return false;
    }

    ranges.sort((a, b) => a[0] - b[0] || b[1] - a[1]);
    const merged = [];
    for (const [s, e] of ranges) {
      if (!merged.length || s > merged[merged.length - 1][1]) {
        merged.push([s, e]);
      } else {
        merged[merged.length - 1][1] = Math.max(merged[merged.length - 1][1], e);
      }
    }

    let pos = 0;
    for (const [s, e] of merged) {
      if (s > pos) container.appendChild(document.createTextNode(raw.slice(pos, s)));
      container.appendChild(el("mark", "dv-mark", raw.slice(s, e)));
      pos = e;
    }
    if (pos < raw.length) container.appendChild(document.createTextNode(raw.slice(pos)));
    return true;
  }

  async function resolveDocIdFromRegistry(docNumber, fallbackTitle) {
    if (!documents.length) {
      try {
        const res = await api("/api/documents");
        documents = (res && res.documents) || [];
      } catch {
        /* ignore */
      }
    }
    if (!documents.length) return "";
    const wantNum = normQ(docNumber);
    const wantTitle = normQ(fallbackTitle);
    let best = null;
    let bestSc = 0;
    for (const d of documents) {
      let sc = 0;
      if (wantNum && d.doc_number && normQ(d.doc_number) === wantNum) sc += 5;
      const dt = normQ(d.title || d.file_name);
      if (dt && wantTitle) {
        if (dt === wantTitle) sc += 5;
        else if (wantTitle.includes(dt.slice(0, 28)) || dt.includes(wantTitle.slice(0, 28))) sc += 3;
      }
      if (sc > bestSc) {
        best = d;
        bestSc = sc;
      }
    }
    return best ? best.doc_id : "";
  }

  async function openDocumentViewer({ docId, page, quote, clause, docNumber, fallbackTitle } = {}) {
    const reqId = ++viewerState.reqId;

    // Open drawer immediately so interaction feels instantaneous
    viewerState.open = true;
    docViewer.hidden = false;
    appShell.classList.add("has-viewer");
    closeMobileHistory();

    if (window.innerWidth < 1280 && window.innerWidth >= 1025 && desktopSidebarOpen) {
      desktopSidebarOpen = false;
      autoCollapsedByViewer = true;
      syncHistoryUi();
    }

    dvTitle.textContent = cleanSource(fallbackTitle) || "Загрузка документа…";
    dvMeta.hidden = true;
    dvSheets.hidden = true;
    dvSheets.innerHTML = "";
    dvBody.innerHTML = "";
    dvBody.appendChild(el("div", "dv-status-box", "Открытие документа на нужной странице…"));

    let resolvedId = docId || "";
    if (!resolvedId && (docNumber || fallbackTitle)) {
      resolvedId = await resolveDocIdFromRegistry(docNumber, fallbackTitle);
      if (reqId !== viewerState.reqId) return;
    }

    if (!resolvedId) {
      dvBody.innerHTML = "";
      dvBody.appendChild(
        el("div", "dv-status-box", "Документ не найден в активном реестре базы знаний.")
      );
      return;
    }

    const params = new URLSearchParams();
    if (page && Number.isFinite(Number(page)) && Number(page) >= 1) {
      params.set("page", String(Math.floor(Number(page))));
    }
    if (quote) {
      params.set("quote", String(quote).slice(0, 700));
    }
    if (clause) {
      const cleanCl = String(clause).replace(/^(?:пп?\.|пункт)\s*/i, "").trim();
      if (cleanCl) params.set("clause", cleanCl);
    }

    try {
      const qs = params.toString();
      const data = await api(
        `/api/documents/${encodeURIComponent(resolvedId)}/view${qs ? "?" + qs : ""}`
      );
      if (reqId !== viewerState.reqId) return;
      renderViewerPayload(data, { requestedPage: page, quote, clause });
    } catch (err) {
      if (reqId !== viewerState.reqId) return;
      dvBody.innerHTML = "";
      dvBody.appendChild(
        el(
          "div",
          "dv-status-box",
          err.message || "Не удалось загрузить предпросмотр документа."
        )
      );
    }
  }

  function renderViewerPayload(data, { requestedPage, quote, clause }) {
    const fmt = String(data.format || "pdf").toLowerCase();
    const known = ["pdf", "docx", "doc", "xlsx", "xls"].includes(fmt);
    const isExcel = fmt === "xlsx" || fmt === "xls";
    const pages = Array.isArray(data.pages) ? data.pages : [];
    const total = Math.max(1, Number(data.total_pages) || pages.length || 1);
    const targetPage = Math.min(
      total,
      Math.max(1, Number(data.target_page) || Number(requestedPage) || 1)
    );

    viewerState.docId = data.doc_id || "";
    viewerState.format = fmt;
    viewerState.totalPages = total;
    viewerState.currentPage = targetPage;

    dvFormatBadge.className = `doc-fmt ${known ? `${fmt} ext-${fmt}` : "other"}`;
    dvFormatBadge.textContent = (fmt || "doc").toUpperCase().slice(0, 4);

    dvTitle.textContent = data.title || data.file_name || data.doc_id || "Документ";
    const numClean = cleanDocNo(data.doc_number);
    const dateClean = fmtDateNumeric(data.doc_date);
    const metaText = [numClean ? `№ ${numClean}` : "", dateClean ? `от ${dateClean}` : ""]
      .filter(Boolean)
      .join(", ");
    dvMeta.textContent = metaText;
    dvMeta.hidden = !metaText;

    dvDownloadLink.href =
      data.download_url || `/api/documents/${encodeURIComponent(viewerState.docId)}/file`;

    dvPagePrefix.textContent = isExcel ? "Лист" : "Стр.";
    dvPageTotal.textContent = String(total);
    dvPageInput.max = String(total);
    dvPageInput.value = String(targetPage);
    applyViewerZoom();

    // Excel sheet switcher tabs
    dvSheets.innerHTML = "";
    if (isExcel && pages.length > 1) {
      dvSheets.hidden = false;
      pages.forEach((p) => {
        const pNum = Number(p.page_number) || 1;
        const tab = el(
          "button",
          "dv-sheet-tab" + (pNum === targetPage ? " is-active" : ""),
          p.label || `Лист ${pNum}`
        );
        tab.type = "button";
        tab.setAttribute("role", "tab");
        tab.setAttribute("aria-selected", String(pNum === targetPage));
        tab.dataset.page = String(pNum);
        tab.addEventListener("click", () => {
          scrollToViewerPage(pNum, { instant: false });
        });
        dvSheets.appendChild(tab);
      });
    } else {
      dvSheets.hidden = true;
    }

    dvBody.innerHTML = "";
    if (!pages.length) {
      dvBody.appendChild(el("div", "dv-status-box", "Содержимое документа пусто."));
      syncViewerNavControls(1);
      return;
    }

    const frag = document.createDocumentFragment();

    if (fmt === "pdf") {
      pages.forEach((p) => {
        const pNum = Number(p.page_number) || 1;
        const card = el("div", "dv-page-card" + (pNum === targetPage ? " is-target" : ""));
        card.dataset.page = String(pNum);

        const bar = el("div", "dv-page-bar");
        bar.appendChild(el("span", null, p.label || `Стр. ${pNum}`));
        const hls = Array.isArray(p.highlights) ? p.highlights : [];
        if (hls.length) {
          bar.appendChild(el("span", "dv-page-match-tag", "Выдержка на странице"));
        }
        card.appendChild(bar);

        const stage = el("div", "dv-pdf-stage");
        const w = Number(p.width) || 595.3;
        const h = Number(p.height) || 841.9;
        stage.style.aspectRatio = `${w} / ${h}`;

        const img = el("img", "dv-pdf-img");
        const isNear = Math.abs(pNum - targetPage) <= 1;
        img.loading = isNear ? "eager" : "lazy";
        img.decoding = "async";
        img.alt = `Страница ${pNum}`;
        img.src = p.image_url;
        stage.appendChild(img);

        for (const hl of hls) {
          const x = Math.max(0, Math.min(1, Number(hl.x) || 0));
          const y = Math.max(0, Math.min(1, Number(hl.y) || 0));
          const bw = Math.max(0.012, Math.min(1 - x, Number(hl.w) || 0));
          const bh = Math.max(0.01, Math.min(1 - y, Number(hl.h) || 0));
          const overlay = el("span", "dv-pdf-hl");
          overlay.style.left = `${(x * 100).toFixed(3)}%`;
          overlay.style.top = `${(y * 100).toFixed(3)}%`;
          overlay.style.width = `${(bw * 100).toFixed(3)}%`;
          overlay.style.height = `${(bh * 100).toFixed(3)}%`;
          stage.appendChild(overlay);
        }

        card.appendChild(stage);
        frag.appendChild(card);
      });
    } else {
      const needles = buildQuoteNeedles(quote);
      const cleanClause = String(clause || "")
        .replace(/^(?:пп?\.|пункт)\s*/i, "")
        .replace(/\.$/, "")
        .trim();

      pages.forEach((p) => {
        const pNum = Number(p.page_number) || 1;
        const sheet = el("article", "dv-folio-sheet" + (pNum === targetPage ? " is-target" : ""));
        sheet.dataset.page = String(pNum);

        const head = el("div", "dv-folio-head");
        head.appendChild(
          el("span", null, p.label || (isExcel ? `Лист ${pNum}` : `Стр. ${pNum}`))
        );
        sheet.appendChild(head);

        let sheetHasMatch = false;
        const blocks = Array.isArray(p.blocks) ? p.blocks : [];
        blocks.forEach((b) => {
          const bType = String(b.type || "paragraph").toLowerCase();
          const bClause = String(b.clause_no || "").replace(/\.$/, "").trim();
          const clauseMatched = Boolean(cleanClause && bClause && bClause === cleanClause);

          if (
            bType === "table" &&
            ((Array.isArray(b.headers) && b.headers.length) ||
              (Array.isArray(b.rows) && b.rows.length))
          ) {
            const wrap = el("div", "dv-table-wrap");
            const tbl = el("table", "dv-table");
            if (Array.isArray(b.headers) && b.headers.length) {
              const thead = el("thead");
              const tr = el("tr");
              b.headers.forEach((cell) => {
                const th = el("th");
                if (appendHighlightedText(th, String(cell ?? ""), needles)) {
                  sheetHasMatch = true;
                }
                tr.appendChild(th);
              });
              thead.appendChild(tr);
              tbl.appendChild(thead);
            }
            if (Array.isArray(b.rows) && b.rows.length) {
              const tbody = el("tbody");
              b.rows.forEach((row) => {
                const tr = el("tr");
                let rowMatched = false;
                (Array.isArray(row) ? row : []).forEach((cell) => {
                  const td = el("td");
                  if (appendHighlightedText(td, String(cell ?? ""), needles)) {
                    rowMatched = true;
                    sheetHasMatch = true;
                  }
                  tr.appendChild(td);
                });
                if (rowMatched) tr.classList.add("is-match-row");
                tbody.appendChild(tr);
              });
              tbl.appendChild(tbody);
            }
            wrap.appendChild(tbl);
            sheet.appendChild(wrap);
          } else {
            const tag = bType === "heading" ? "h3" : "p";
            const cls =
              (bType === "heading" ? "dv-block-heading" : "dv-block-para") +
              (clauseMatched ? " is-clause-match" : "");
            const blk = el(tag, cls);
            const rawText = String(b.text || "");
            if (bClause && !rawText.trim().startsWith(bClause)) {
              blk.appendChild(el("span", "dv-clause-tag", `${bClause}.`));
            }
            if (appendHighlightedText(blk, rawText, needles)) {
              sheetHasMatch = true;
            }
            if (blk.textContent) sheet.appendChild(blk);
          }
        });

        if (sheetHasMatch) {
          head.appendChild(el("span", "dv-page-match-tag", "Найдена цитата"));
        }
        frag.appendChild(sheet);
      });
    }

    dvBody.appendChild(frag);
    syncViewerNavControls(targetPage);

    // Jump immediately on mount and reinforce on next animation frame for exact scrollTop
    scrollToViewerPage(targetPage, { instant: true, focusHighlight: true });
    requestAnimationFrame(() => {
      scrollToViewerPage(targetPage, { instant: true, focusHighlight: true });
    });
  }

  /* ---------------- Router (#/ & #/kb) ---------------- */
  const viewChat = $("#viewChat");
  const viewKb = $("#viewKb");
  const backBtn = $("#backBtn");
  const kbBtn = $("#kbBtn");
  const newChatBtn = $("#newChatBtn");

  function route() {
    const isKb = location.hash === "#/kb";
    const show = isKb ? viewKb : viewChat;
    const hide = isKb ? viewChat : viewKb;

    hide.hidden = true;
    hide.classList.remove("is-active");
    if (show.hidden) {
      show.hidden = false;
      show.classList.remove("entering");
      void show.offsetWidth;
      show.classList.add("entering");
    }
    show.classList.add("is-active");

    backBtn.hidden = !isKb;
    newChatBtn.hidden = isKb;
    kbBtn.hidden = isKb;
    syncHistoryUi();

    document.title = isKb
      ? "Реестр документов — Нормативная база МФТИ"
      : "Нормативная база МФТИ — Справочник документов";

    if (isKb) loadDocuments();
    else setTimeout(() => input.focus({ preventScroll: true }), 40);
  }

  kbBtn.addEventListener("click", () => {
    location.hash = "#/kb";
  });
  backBtn.addEventListener("click", () => {
    location.hash = "#/";
  });
  window.addEventListener("hashchange", route);

  /* ---------------- Live Registry Health ---------------- */
  async function checkHealth() {
    const dot = $("#statusDot");
    const lbl = $("#healthLabel");
    try {
      const h = await api("/api/health");
      const ok = h && h.status === "ok" && (!h.qdrant || h.qdrant.connected !== false);
      dot.className = "status-dot " + (ok ? "ok" : "bad");
      if (ok) {
        const n = Number(h.registry_docs_count);
        lbl.textContent = Number.isFinite(n)
          ? `${n} ${plural(n, "акт", "акта", "актов")} в реестре`
          : "Реестр подключён";
      } else {
        lbl.textContent = "Ограниченный режим";
      }
    } catch {
      dot.className = "status-dot bad";
      lbl.textContent = "Сервер недоступен";
    }
  }

  /* ==========================================================================
     REFERENCE DESK / STREAMING DOSSIER (#/)
     ========================================================================== */
  const input = $("#input");
  const sendBtn = $("#sendBtn");
  const sendLabel = $("#sendLabel");
  const composerHint = $("#composerHint");
  const composer = $("#composer");
  const messages = $("#messages");
  const emptyState = $("#emptyState");
  const chatScroll = $("#chatScroll");

  let busy = false;
  let controller = null;
  let chatEpoch = 0;
  let queryCount = 0;

  function syncDeskState(hasThread) {
    viewChat.classList.toggle("is-empty", !hasThread);
    viewChat.classList.toggle("has-thread", hasThread);
    emptyState.hidden = hasThread;
    input.rows = hasThread ? 1 : 2;
  }

  function autosize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 180) + "px";
    sendBtn.disabled = busy ? false : !input.value.trim();
  }

  function setBusy(b) {
    busy = b;
    sendBtn.classList.toggle("is-stop", b);
    sendLabel.textContent = b ? "Стоп" : "Найти";
    sendBtn.setAttribute("aria-label", b ? "Остановить генерацию" : "Отправить запрос");
    sendBtn.title = b ? "Остановить генерацию (Esc)" : "Отправить запрос";
    if (composerHint) {
      composerHint.textContent = b
        ? "Формирование справки — нажмите Esc для остановки"
        : "Enter — найти, Shift + Enter — перенос строки";
    }
    autosize();
  }

  function stopGeneration() {
    if (busy && controller) controller.abort();
  }

  input.addEventListener("input", autosize);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      if (!busy) composer.requestSubmit();
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (busy) {
        e.preventDefault();
        stopGeneration();
        return;
      }
      const dlg = $("#confirmDialog");
      if (dlg && dlg.open) return;
      if (isCompactScreen() && mobileSidebarOpen) {
        e.preventDefault();
        closeMobileHistory();
        return;
      }
      if (viewerState.open) {
        e.preventDefault();
        closeDocumentViewer();
        return;
      }
      if (location.hash === "#/kb") {
        location.hash = "#/";
      }
    }
  });

  sendBtn.addEventListener("click", (e) => {
    if (busy) {
      e.preventDefault();
      stopGeneration();
      input.focus({ preventScroll: true });
    }
  });

  composer.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q || busy) return;
    input.value = "";
    autosize();
    ask(q);
  });

  newChatBtn.addEventListener("click", () => {
    if (activeThreadId) discardEmptyThread(activeThreadId);
    chatEpoch++;
    activeThreadId = null;
    queryCount = 0;
    if (controller) controller.abort();
    controller = null;
    setBusy(false);
    closeMobileHistory();
    messages.innerHTML = "";
    syncDeskState(false);
    renderHistoryList();
    input.value = "";
    autosize();
    input.focus({ preventScroll: true });
  });

  /* ---- Smart Auto-Scroll ---- */
  let stick = true;
  let lastTop = 0;
  chatScroll.addEventListener(
    "scroll",
    () => {
      const top = chatScroll.scrollTop;
      const dist = chatScroll.scrollHeight - top - chatScroll.clientHeight;
      if (top < lastTop - 2 && dist > 48) stick = false;
      else if (dist < 32) stick = true;
      lastTop = top;
    },
    { passive: true }
  );

  function scrollToBottom(force = false) {
    if (force) stick = true;
    if (!stick) return;
    requestAnimationFrame(() => {
      chatScroll.scrollTop = chatScroll.scrollHeight;
    });
  }

  function addUserMessage(text, idx) {
    const li = el("li", "msg msg-user");
    const meta = el("div", "query-meta", `Запрос ${String(idx).padStart(2, "0")}`);
    const heading = el("h2", "query-heading", text);
    li.append(meta, heading);
    messages.appendChild(li);
  }

  function addBotShell() {
    const li = el("li", "msg msg-bot");
    const body = el("div", "msg-body");
    li.appendChild(body);
    messages.appendChild(li);
    return { li, body };
  }

  const STAGES = [
    "Поиск по нормативным актам…",
    "Сверка пунктов и таблиц…",
    "Подготовка текста справки…",
    "Завершение проверки цитат…",
  ];

  function showTyping(body) {
    const t = el("div", "typing");
    t.setAttribute("aria-label", "Подготовка ответа");
    t.appendChild(el("span", "typing-pulse"));
    const label = el("span", "typing-label", STAGES[0]);
    t.appendChild(label);
    body.appendChild(t);

    let i = 0;
    const setLabel = (text) => {
      if (label.textContent === text) return;
      label.style.opacity = "0";
      setTimeout(() => {
        label.textContent = text;
        label.style.opacity = "1";
      }, 160);
    };
    const timer = setInterval(() => {
      i = Math.min(i + 1, STAGES.length - 1);
      setLabel(STAGES[i]);
      if (i === STAGES.length - 1) clearInterval(timer);
    }, 5500);

    return {
      set(text) {
        clearInterval(timer);
        setLabel(text);
      },
      stop() {
        clearInterval(timer);
        t.remove();
      },
    };
  }

  function prettyPages(pages) {
    const p = [...new Set((pages || []).filter((n) => Number.isFinite(n)))].sort((a, b) => a - b);
    if (!p.length) return "";
    if (p.length === 1) return `стр. ${p[0]}`;
    const contiguous = p.every((n, i) => i === 0 || n === p[i - 1] + 1);
    return contiguous ? `стр. ${p[0]}–${p[p.length - 1]}` : `стр. ${p.join(", ")}`;
  }

  function buildSources(hits) {
    const seen = new Map();
    for (const h of hits || []) {
      const name = (h.title || h.file_name || "Документ").replace(/\.(pdf|docx?|xlsx?)$/i, "");
      let rawClause = String(h.clause_no || "").trim();
      if (rawClause && /б\/н|\[_*\]/i.test(rawClause)) rawClause = "";
      let clause = rawClause;
      if (clause && !/^(п\.|пп\.|раздел|глава|ст\.)/i.test(clause)) clause = "п. " + clause;
      const docNumber = cleanDocNo(h.doc_number);
      const key = `${h.doc_id || name}|${clause}`;
      if (seen.has(key)) {
        seen.get(key).pages.push(...(h.page_numbers || []));
        continue;
      }
      seen.set(key, {
        name,
        docId: h.doc_id || "",
        docNumber,
        clause,
        rawClause,
        pages: [...(h.page_numbers || [])],
        section: cleanSource(h.section_title || ""),
      });
    }
    return [...seen.values()].slice(0, 8).map((s) => {
      const uniquePages = [...new Set((s.pages || []).filter((n) => Number.isFinite(n)))].sort(
        (a, b) => a - b
      );
      const loc = [s.docNumber ? `№ ${s.docNumber}` : "", s.clause, prettyPages(uniquePages)]
        .filter(Boolean)
        .join(", ");
      return {
        name: s.name,
        loc: loc || s.section || "",
        docId: s.docId || "",
        docNumber: s.docNumber || "",
        clause: s.clause || "",
        rawClause: s.rawClause || "",
        pages: uniquePages,
      };
    });
  }

  function renderAnswer(
    body,
    { answer, hits, sources: prebuiltSources, quotes, latencyMs, note, noteKind, mdBox: liveBox }
  ) {
    let mdBox = liveBox;
    if (mdBox) {
      patchMarkdown(mdBox, answer, false);
      mdBox.classList.remove("streaming");
    } else {
      mdBox = renderMarkdown(answer || (note ? "" : "_Текст справки не получен._"));
      body.appendChild(mdBox);
    }

    if (note) {
      const n = el("div", `gen-note ${noteKind === "warn" ? "warn" : ""}`);
      n.innerHTML = noteKind === "warn" ? ICONS.warn : ICONS.stop;
      n.appendChild(el("span", null, note));
      body.appendChild(n);
    }

    const actions = el("div", "msg-actions");
    const sources = Array.isArray(prebuiltSources) ? prebuiltSources : buildSources(hits);
    const cardMap = new Map();
    let panel = null;
    let sourcesApi = null;

    if (sources.length) {
      const btn = el("button", "mini-btn");
      btn.type = "button";
      btn.setAttribute("aria-expanded", "false");
      btn.innerHTML = `${ICONS.doc}<span>Источники (${sources.length})</span>${ICONS.chev}`;

      panel = el("div", "sources");
      const inner = el("div");
      const list = el("div", "source-list");

      sources.forEach((s, i) => {
        const card = el("button", "source");
        card.type = "button";
        cardMap.set(s, card);
        const firstPage = Array.isArray(s.pages) && s.pages.length ? s.pages[0] : null;
        card.title = [
          s.name,
          s.loc,
          firstPage ? `Открыть документ на стр. ${firstPage}` : "Открыть документ",
        ]
          .filter(Boolean)
          .join("\n");
        card.appendChild(el("span", "source-idx", String(i + 1)));
        const txt = el("span", "source-text");
        txt.appendChild(el("span", "source-name", s.name));
        if (s.loc) txt.appendChild(el("span", "source-loc", s.loc));
        const openHint = el("span", "source-open-label");
        openHint.innerHTML = ICONS.openDoc;
        openHint.appendChild(
          el("span", null, firstPage ? `Открыть на стр. ${firstPage}` : "Открыть документ")
        );
        txt.appendChild(openHint);
        card.appendChild(txt);

        card.addEventListener("click", () => {
          openDocumentViewer({
            docId: s.docId,
            page: firstPage || 1,
            clause: s.rawClause || s.clause || "",
            docNumber: s.docNumber || "",
            fallbackTitle: s.name,
          });
        });

        list.appendChild(card);
      });

      inner.appendChild(list);
      panel.appendChild(inner);
      const id = "src-" + Math.random().toString(36).slice(2, 8);
      panel.id = id;
      btn.setAttribute("aria-controls", id);

      const setOpen = (open) => {
        panel.classList.toggle("open", open);
        btn.setAttribute("aria-expanded", String(open));
      };

      btn.addEventListener("click", () => {
        const open = !panel.classList.contains("open");
        setOpen(open);
        if (open) {
          setTimeout(() => panel.scrollIntoView({ behavior: "smooth", block: "nearest" }), 240);
        }
      });

      actions.appendChild(btn);

      sourcesApi = {
        reveal(q, label) {
          const best = matchBestSource(q, sources, label);
          setOpen(true);
          cardMap.forEach((c) => c.classList.remove("is-target"));
          setTimeout(() => {
            const targetCard = best ? cardMap.get(best) : null;
            const scrollTarget = targetCard || panel;
            scrollTarget.scrollIntoView({ behavior: "smooth", block: "nearest" });
            if (targetCard) {
              void targetCard.offsetWidth;
              targetCard.classList.add("is-target");
              setTimeout(() => targetCard.classList.remove("is-target"), 2400);
            } else {
              toast("Документ не найден в кратком перечне источников");
            }
          }, 260);
        },
      };
    }

    if (answer) {
      const copyBtn = el("button", "mini-btn");
      copyBtn.type = "button";
      copyBtn.setAttribute("aria-label", "Скопировать текст справки");
      copyBtn.innerHTML = `${ICONS.copy}<span>Скопировать</span>`;
      copyBtn.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(answer);
          copyBtn.innerHTML = `${ICONS.check}<span>Скопировано</span>`;
          setTimeout(() => {
            copyBtn.innerHTML = `${ICONS.copy}<span>Скопировать</span>`;
          }, 1600);
        } catch {
          toast("Не удалось скопировать текст", "error");
        }
      });
      actions.appendChild(copyBtn);
    }

    if (latencyMs) {
      actions.appendChild(el("span", "meta-note", `${(latencyMs / 1000).toFixed(1)} с`));
    }
    if (actions.childElementCount) body.appendChild(actions);
    if (panel) body.appendChild(panel);
    applyQuoteMeta(mdBox, quotes, sources, sourcesApi);
    return sources;
  }

  function renderError(li, body, err, question) {
    li.classList.add("msg-error");
    body.appendChild(
      renderMarkdown(
        "Не удалось сформировать справку из-за сбоя соединения. Проверьте доступность сервера и повторите запрос."
      )
    );
    const actions = el("div", "msg-actions");
    const retry = el("button", "mini-btn");
    retry.type = "button";
    retry.innerHTML = `${ICONS.retry}<span>Повторить запрос</span>`;
    retry.addEventListener("click", () => {
      li.remove();
      const prev = messages.lastElementChild;
      if (prev && prev.classList.contains("msg-user")) {
        prev.remove();
        queryCount = Math.max(0, queryCount - 1);
      }
      if (!messages.childElementCount) syncDeskState(false);
      ask(question);
    });
    actions.appendChild(retry);
    body.appendChild(actions);
    toast(
      err.message && err.message.length < 160 ? err.message : "Ошибка соединения с сервером",
      "error"
    );
  }

  /* ---- SSE Stream Reader ---- */
  async function readSSE(res, onEvent) {
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const raw = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        let event = "message";
        const data = [];
        for (const line of raw.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
        }
        if (!data.length) continue;
        let payload;
        try {
          payload = JSON.parse(data.join("\n"));
        } catch {
          payload = data.join("\n");
        }
        onEvent(event, payload);
      }
    }
  }

  async function ask(question) {
    if (busy) return;
    const ctrl = new AbortController();
    controller = ctrl;
    setBusy(true);
    syncDeskState(true);

    const thr = ensureActiveThread(question);
    const threadId = thr.id;

    queryCount++;
    addUserMessage(question, queryCount);

    const { li, body } = addBotShell();
    const typing = showTyping(body);
    scrollToBottom(true);
    const epoch = chatEpoch;

    const state = {
      text: "",
      hits: [],
      quotes: [],
      latencyMs: 0,
      stopped: false,
      error: null,
    };
    let live = null;
    let raf = 0;

    const paint = () => {
      raf = 0;
      if (epoch !== chatEpoch) return;
      if (!live) {
        typing.stop();
        live = el("div", "md streaming");
        body.appendChild(live);
      }
      patchMarkdown(live, state.text, true);
      scrollToBottom();
    };

    const schedule = () => {
      if (!raf) raf = requestAnimationFrame(paint);
    };

    const payload = JSON.stringify({ query: question });

    try {
      const res = await fetch("/api/ask/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: payload,
        signal: ctrl.signal,
      });

      if (res.status === 404 || res.status === 405) {
        const resp = await api("/api/ask", { method: "POST", body: payload, signal: ctrl.signal });
        state.text = (resp && resp.answer) || "";
        state.hits = (resp && resp.search_response && resp.search_response.hits) || [];
        state.latencyMs = (resp && resp.latency_ms) || 0;
        state.quotes = (resp && resp.quotes) || [];
      } else if (!res.ok) {
        let detail = null;
        try {
          const d = await res.json();
          detail = typeof d.detail === "string" ? d.detail : null;
        } catch {
          /* ignore */
        }
        throw new Error(detail || `Ошибка сервера (${res.status})`);
      } else {
        await readSSE(res, (event, data) => {
          if (epoch !== chatEpoch) return;
          if (event === "sources") {
            state.hits = (data && data.hits) || [];
            typing.set("Подготовка текста справки…");
          } else if (event === "status") {
            if (data && data.text) typing.set(data.text);
          } else if (event === "token") {
            state.text += (data && data.text) || "";
            schedule();
          } else if (event === "quotes") {
            state.quotes = (data && data.quotes) || [];
          } else if (event === "done") {
            state.latencyMs = (data && data.latency_ms) || 0;
          } else if (event === "error") {
            state.error = new Error((data && data.message) || "Ошибка генерации ответа");
          }
        });
      }
    } catch (err) {
      if (err && err.name === "AbortError") state.stopped = true;
      else state.error = err;
    }

    if (epoch !== chatEpoch) return;
    if (raf) cancelAnimationFrame(raf);
    typing.stop();
    if (live && !state.text) {
      live.remove();
      live = null;
    }

    if (!state.text && state.error && !state.stopped) {
      discardEmptyThread(threadId);
      renderError(li, body, state.error, question);
    } else {
      let note = null;
      let noteKind = "stop";
      if (state.stopped) note = "Генерация остановлена";
      else if (state.error) {
        note = "Генерация прервана из-за ошибки";
        noteKind = "warn";
      }
      const builtSources = renderAnswer(body, {
        answer: state.text,
        hits: state.hits,
        quotes: state.quotes,
        latencyMs: state.stopped ? 0 : state.latencyMs,
        note,
        noteKind,
        mdBox: live,
      });

      if (state.text || state.stopped) {
        recordCompletedTurn(threadId, {
          query: question,
          markdown: state.text,
          quotes: sanitizeQuotesForStorage(state.quotes),
          sources: builtSources,
          latencySec:
            state.stopped || !state.latencyMs ? 0 : Number((state.latencyMs / 1000).toFixed(1)),
          stopped: Boolean(state.stopped),
        });
      } else {
        discardEmptyThread(threadId);
      }
    }

    if (controller === ctrl) controller = null;
    setBusy(false);
    scrollToBottom();
    input.focus({ preventScroll: true });
  }

  /* ==========================================================================
     KNOWLEDGE BASE REGISTER (#/kb)
     ========================================================================== */
  const docList = $("#docList");
  const docSearch = $("#docSearch");
  const kbStats = $("#kbStats");
  const dropzone = $("#dropzone");
  const fileInput = $("#fileInput");
  const uploads = $("#uploads");
  let documents = [];

  const STATUS_LABEL = {
    active: "Действует",
    superseded: "Утратил силу",
    superseded_template_reference: "Справочный эталон",
    expired: "Истёк",
    draft: "Проект",
    amended: "Изменён",
  };
  const ALLOWED = /\.(pdf|docx|doc|xlsx|xls)$/i;

  const foldSearch = (s) =>
    String(s || "")
      .toLowerCase()
      .replace(/ё/g, "е");

  /** Clean internal English enum tags & snake_case doc_ids from version_note. */
  const cleanVersionNote = (raw) =>
    cleanSource(raw || "")
      .replace(/\s*\([a-z][a-z0-9_]*\)/gi, "")
      .replace(/\s*Отменен\s*\/\s*заменен\s+документом\s+№?\s*[a-z0-9_]+\.?/gi, "")
      .replace(/^ВНИМАНИЕ\s*\(([^)]+)\):\s*/i, "$1: ")
      .replace(/\s{2,}/g, " ")
      .trim();

  async function loadDocuments() {
    if (!documents.length) {
      docList.innerHTML = "";
      for (let i = 0; i < 4; i++) docList.appendChild(el("li", "skeleton"));
    }
    try {
      const data = await api("/api/documents");
      documents = (data && data.documents) || [];
      documents.sort((a, b) => String(b.indexed_at || "").localeCompare(String(a.indexed_at || "")));
      const n = documents.length;
      kbStats.textContent = n
        ? `${n} ${plural(n, "документ", "документа", "документов")} в активном индексе`
        : "В реестре пока нет документов — загрузите файлы для индексации";
      renderDocs();
    } catch (err) {
      kbStats.textContent = "Не удалось получить реестр документов";
      docList.innerHTML = "";
      docList.appendChild(
        el("li", "doc-empty", "Сервер недоступен. Обновите страницу или проверьте соединение.")
      );
      toast(err.message, "error");
    }
  }

  function renderDocs() {
    const q = foldSearch(docSearch.value.trim());
    const list = q
      ? documents.filter((d) => {
          const status = String(d.status || "active");
          return [
            d.title,
            d.file_name,
            d.doc_number,
            d.doc_type,
            cleanVersionNote(d.version_note),
            STATUS_LABEL[status] || status,
          ].some((v) => foldSearch(v).includes(q));
        })
      : documents;

    docList.innerHTML = "";
    if (!list.length) {
      docList.appendChild(
        el("li", "doc-empty", q ? "По вашему фильтру документы не найдены" : "Реестр пуст")
      );
      return;
    }

    const frag = document.createDocumentFragment();
    for (const d of list) {
      const li = el("li", "doc");
      const fmt = String(d.file_format || (d.file_name || "").split(".").pop() || "").toLowerCase();
      const known = ["pdf", "docx", "doc", "xlsx", "xls"].includes(fmt);
      const fmtLabel = (fmt || "file").toUpperCase().slice(0, 4);
      li.appendChild(el("span", `doc-fmt ${known ? `${fmt} ext-${fmt}` : "other"}`, fmtLabel));

      const main = el("div", "doc-main");
      const title = d.title || d.file_name || d.doc_id;
      const titleBtn = el("button", "doc-title-btn");
      titleBtn.type = "button";
      titleBtn.title = `Открыть «${d.file_name || title}»`;
      titleBtn.appendChild(el("span", "doc-title", title));
      titleBtn.addEventListener("click", () => {
        openDocumentViewer({
          docId: d.doc_id,
          page: 1,
          docNumber: d.doc_number,
          fallbackTitle: title,
        });
      });
      main.appendChild(titleBtn);

      const noteText = cleanVersionNote(d.version_note);
      if (noteText) {
        const sub = el("div", "doc-note", noteText);
        sub.title = noteText;
        main.appendChild(sub);
      }
      li.appendChild(main);

      // Tabular act number & date
      const req = el("div", "doc-req");
      const numClean = cleanDocNo(d.doc_number);
      const dateClean = fmtDateNumeric(d.doc_date);
      req.appendChild(el("span", null, numClean ? `№ ${numClean}` : "Без номера"));
      if (dateClean) req.appendChild(el("span", "doc-req-sub", `от ${dateClean}`));
      li.appendChild(req);

      // Tabular page count
      const isExcel = fmt === "xlsx" || fmt === "xls";
      const pagesText = d.pages_total
        ? `${d.pages_total} ${isExcel ? plural(d.pages_total, "лист", "листа", "листов") : "стр."}`
        : "—";
      li.appendChild(el("div", "doc-pages", pagesText));

      // Status badge
      const status = String(d.status || "active");
      const statusText = STATUS_LABEL[status] || status;
      li.appendChild(el("span", `badge ${status}`, statusText));

      // Compact mobile metadata row
      const mobMeta = el("div", "doc-mobile-meta");
      mobMeta.appendChild(el("span", `badge ${status}`, statusText));
      if (numClean) mobMeta.appendChild(el("span", null, `№ ${numClean}`));
      if (dateClean) mobMeta.appendChild(el("span", null, dateClean));
      if (d.pages_total) mobMeta.appendChild(el("span", null, pagesText));
      li.appendChild(mobMeta);

      // Preview & Delete actions
      const rowActions = el("div", "doc-row-actions");

      const openBtn = el("button", "doc-open-btn");
      openBtn.type = "button";
      openBtn.setAttribute("aria-label", `Открыть документ «${title}»`);
      openBtn.title = "Просмотреть документ";
      openBtn.innerHTML = ICONS.openDoc;
      openBtn.appendChild(el("span", "hide-sm", "Открыть"));
      openBtn.addEventListener("click", () => {
        openDocumentViewer({
          docId: d.doc_id,
          page: 1,
          docNumber: d.doc_number,
          fallbackTitle: title,
        });
      });
      rowActions.appendChild(openBtn);

      const del = el("button", "icon-btn doc-del");
      del.type = "button";
      del.setAttribute("aria-label", `Удалить документ «${title}»`);
      del.title = "Удалить из реестра";
      del.innerHTML = ICONS.trash;
      del.addEventListener("click", () => confirmDelete(d, title, li));
      rowActions.appendChild(del);

      li.appendChild(rowActions);
      frag.appendChild(li);
    }
    docList.appendChild(frag);
  }

  docSearch.addEventListener("input", renderDocs);

  async function confirmDelete(doc, title, li) {
    const ok = await openConfirmDialog({
      title: "Исключить документ из реестра?",
      text: `Документ «${title}» будет удалён из поискового индекса и перестанет учитываться при подготовке справок.`,
      okLabel: "Удалить из базы",
    });
    if (!ok) return;
    li.style.opacity = "0.4";
    try {
      await api(`/api/documents/${encodeURIComponent(doc.doc_id)}`, { method: "DELETE" });
      if (viewerState.open && viewerState.docId === doc.doc_id) {
        closeDocumentViewer();
      }
      documents = documents.filter((x) => x.doc_id !== doc.doc_id);
      renderDocs();
      toast("Документ удалён из реестра");
      loadDocuments();
      checkHealth();
    } catch (err) {
      li.style.opacity = "";
      toast(err.message, "error");
    }
  }

  /* ---- Drag-and-Drop Upload Queue ---- */
  ["dragenter", "dragover"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.add("drag");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      if (ev === "dragleave" && dropzone.contains(e.relatedTarget)) return;
      dropzone.classList.remove("drag");
    })
  );
  dropzone.addEventListener("drop", (e) => handleFiles(e.dataTransfer.files));
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener("change", () => {
    handleFiles(fileInput.files);
    fileInput.value = "";
  });
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  const queue = [];
  let uploading = false;

  function handleFiles(fileList) {
    const files = [...(fileList || [])];
    const bad = files.filter((f) => !ALLOWED.test(f.name));
    if (bad.length) {
      toast(
        `Пропущено (${bad.map((f) => f.name).join(", ")}): поддерживаются форматы PDF, DOCX, DOC, XLSX, XLS`,
        "error",
        5500
      );
    }
    files
      .filter((f) => ALLOWED.test(f.name))
      .forEach((f) => {
        queue.push({ file: f, ui: createUploadRow(f.name) });
      });
    pump();
  }

  function createUploadRow(name) {
    const li = el("li", "upload");
    const row = el("div", "upload-row");
    row.appendChild(el("span", "upload-name", name));
    const pct = el("span", "upload-pct", "в очереди");
    row.appendChild(pct);
    const bar = el("div", "bar");
    const fill = el("i");
    bar.appendChild(fill);
    const msg = el("div", "upload-msg", "Ожидает индексации");
    li.append(row, bar, msg);
    uploads.prepend(li);
    return {
      li,
      set(p, text) {
        if (p != null) {
          fill.style.width = `${Math.max(0, Math.min(100, p))}%`;
          pct.textContent = `${Math.round(p)}%`;
        }
        if (text != null) msg.textContent = text;
      },
      finish(ok, text) {
        li.classList.add(ok ? "done" : "failed");
        fill.style.width = "100%";
        pct.textContent = ok ? "Готово" : "Ошибка";
        msg.textContent = text;
        if (ok) {
          setTimeout(() => {
            li.style.transition = "opacity 0.35s";
            li.style.opacity = "0";
            setTimeout(() => li.remove(), 360);
          }, 6000);
        }
      },
    };
  }

  async function pump() {
    if (uploading) return;
    uploading = true;
    while (queue.length) {
      const job = queue.shift();
      await uploadOne(job.file, job.ui);
    }
    uploading = false;
    loadDocuments();
    checkHealth();
  }

  function sendFile(file, onProgress) {
    return new Promise((resolve, reject) => {
      const fd = new FormData();
      fd.append("files", file, file.name);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/documents/upload");
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) onProgress(e.loaded / e.total);
      };
      xhr.onload = () => {
        let data = null;
        try {
          data = JSON.parse(xhr.responseText);
        } catch {
          /* ignore */
        }
        if (xhr.status >= 200 && xhr.status < 300) resolve(data || {});
        else reject(new Error((data && data.detail) || `Ошибка загрузки (${xhr.status})`));
      };
      xhr.onerror = () => reject(new Error("Сеть недоступна"));
      xhr.send(fd);
    });
  }

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  async function uploadOne(file, ui) {
    try {
      ui.set(0, "Передача файла на сервер…");
      const res = await sendFile(file, (f) => ui.set(f * 10, "Передача файла на сервер…"));
      if (!res.task_id) {
        ui.finish(true, "Файл принят");
        return;
      }
      ui.set(10, "Разбор структуры и индексация…");
      for (let i = 0; i < 3600; i++) {
        await sleep(i < 5 ? 600 : 1200);
        let t;
        try {
          t = await api(`/api/tasks/${encodeURIComponent(res.task_id)}`);
        } catch {
          continue;
        }
        const msg = String(t.message || t.stage_label || "Индексация…").replace(
          /^\[\d+\/\d+\]\s*/,
          ""
        );
        ui.set(10 + (Number(t.percent) || 0) * 0.9, msg);
        if (t.status === "completed") {
          const st = (t.result && t.result.files && t.result.files[0]) || {};
          if (st.status === "failed" || st.error_message) {
            ui.finish(false, st.error_message || "Не удалось проиндексировать документ");
          } else if (st.status === "skipped") {
            ui.finish(true, "Документ уже присутствует в реестре без изменений");
          } else {
            const parts = [];
            if (st.chunks_created) {
              parts.push(
                `${st.chunks_created} ${plural(st.chunks_created, "фрагмент", "фрагмента", "фрагментов")}`
              );
            }
            if (st.tables_count) {
              parts.push(
                `${st.tables_count} ${plural(st.tables_count, "таблица", "таблицы", "таблиц")}`
              );
            }
            ui.finish(
              true,
              "Включён в реестр" + (parts.length ? ` (${parts.join(", ")})` : "")
            );
          }
          return;
        }
        if (t.status === "failed") {
          ui.finish(false, t.error || "Ошибка индексации");
          return;
        }
      }
      ui.finish(false, "Превышено время ожидания индексации");
    } catch (err) {
      ui.finish(false, err.message);
      toast(`${file.name}: ${err.message}`, "error");
    }
  }

  /* ---------------- Init ---------------- */
  const mq = matchMedia("(max-width: 640px)");
  const setPlaceholder = () => {
    input.placeholder = mq.matches
      ? "Вопрос по документам МФТИ…"
      : "Введите вопрос по приказам, положениям или регламентам МФТИ…";
    autosize();
  };
  mq.addEventListener("change", setPlaceholder);
  window.addEventListener("resize", autosize, { passive: true });
  setPlaceholder();
  renderHistoryList();
  syncDeskState(false);
  autosize();
  route();
  checkHealth();
  setInterval(checkHealth, 60000);
})();

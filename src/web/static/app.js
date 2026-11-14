/* ==========================================================================
   Нормативная база МФТИ — Frontend Application (Vanilla JS, Zero Build Step)
   Features & Preserved Mechanics:
     - Flicker-free incremental DOM morphing during POST /api/ask/stream SSE
     - AbortController Stop button + prioritized global Escape handling
     - Automatic POST /api/ask fallback on 404/405
     - Placeholder stripping ([___], б/н, без номера)
     - Verbatim quote verification seals + expandable clause_text with <mark>
     - Left Query History Sidebar (localStorage: mipt_rag_threads_v1, max 50)
     - Right Side-by-Side Document Viewer Drawer (PDF exact page + Word/Excel
       archival folios & sheet tabs, clean authentic page presentation)
     - Drag-and-drop & URL upload (PDF, DOCX, DOC, XLSX, XLS) + /api/tasks/{id} polling
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
    steps: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 6.5h10M9 12h10M9 17.5h10"/><circle cx="5" cy="6.5" r="1.2" fill="currentColor"/><circle cx="5" cy="12" r="1.2" fill="currentColor"/><circle cx="5" cy="17.5" r="1.2" fill="currentColor"/></svg>',
  };

  const TECH_ERROR_STUB_RU =
    "Возникли технические неполадки при обработке запроса. Пожалуйста, повторите попытку позже или обратитесь к системному администратору.";

  const TECH_LEAK_RE =
    /Traceback \(most recent call last\)|TypeError|ReferenceError|SyntaxError|NetworkError|Failed to fetch|Load failed|FetchError|AbortSignal|JSON\.parse|Unexpected token|Internal Server Error|Bad Gateway|Service Unavailable|Gateway Timeout|Errno\s*\d+|HTTPException|KeyError|AttributeError|ValueError|RuntimeError|OperationalError|ConnectionRefusedError|ConnectError|ReadTimeout|PoolTimeout|qdrant|polza|openai|httpx|litellm|sqlite3|sqlalchemy|pydantic/i;

  function sanitizeUserErrorMessage(err, fallback = TECH_ERROR_STUB_RU) {
    if (!err) return fallback;
    if (typeof err === "object") {
      if (err.isTechnical || (err.status && err.status >= 500)) return TECH_ERROR_STUB_RU;
      if (err.data && err.data.error_type === "technical_error") return TECH_ERROR_STUB_RU;
    }
    const raw =
      typeof err === "string"
        ? err.trim()
        : String((err && (err.userMessage || err.message)) || "").trim();
    if (!raw) return fallback;
    if (raw === TECH_ERROR_STUB_RU) return TECH_ERROR_STUB_RU;
    if (/^\s*[\[{]/.test(raw)) return TECH_ERROR_STUB_RU;
    if (/Ошибка сервера\s*\(5\d\d\)/i.test(raw)) return TECH_ERROR_STUB_RU;
    if (
      /^Ошибка (?:генерации ответа|выполнения|обработки|рендеринга|чтения|обновления|переиндексации|сверки|соединения)/i.test(
        raw
      )
    ) {
      return TECH_ERROR_STUB_RU;
    }
    if (TECH_LEAK_RE.test(raw)) return TECH_ERROR_STUB_RU;
    if (raw.length > 360) return TECH_ERROR_STUB_RU;
    return raw;
  }

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
    let res;
    try {
      res = await fetch(path, {
        ...opts,
        headers:
          opts.body && !(opts.body instanceof FormData)
            ? { "Content-Type": "application/json", ...(opts.headers || {}) }
            : opts.headers,
      });
    } catch (netErr) {
      if (netErr && netErr.name === "AbortError") throw netErr;
      const err = new Error(TECH_ERROR_STUB_RU);
      err.isTechnical = true;
      throw err;
    }
    let data = null;
    try {
      data = await res.json();
    } catch {
      /* non-JSON response */
    }
    if (!res.ok) {
      if (res.status === 409 || (data && data.status === "duplicate")) {
        const dupDetail =
          (data && (typeof data.detail === "string" ? data.detail : data.message)) ||
          "Дубликат документа: документ уже загружен в базу знаний";
        const dupErr = new Error(dupDetail);
        dupErr.status = 409;
        dupErr.isDuplicate = true;
        dupErr.data = data;
        throw dupErr;
      }
      if (res.status >= 500 || (data && data.error_type === "technical_error")) {
        const srvErr = new Error(TECH_ERROR_STUB_RU);
        srvErr.status = res.status;
        srvErr.data = data;
        srvErr.isTechnical = true;
        throw srvErr;
      }
      const detail = data && typeof data.detail === "string" ? data.detail : null;
      const err = new Error(
        detail ? sanitizeUserErrorMessage(detail, TECH_ERROR_STUB_RU) : TECH_ERROR_STUB_RU
      );
      err.status = res.status;
      err.data = data;
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
  const showToast = toast;

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

  function fillMarkdown(box, text, { allowQuotes = true, stepCite = null } = {}) {
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
    if (allowQuotes) decorateQuotes(box);
    if (stepCite) decorateActionStepSources(box, stepCite);
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

  function patchMarkdown(box, text, withCaret, { allowQuotes = true, stepCite = null } = {}) {
    const next = fillMarkdown(el("div"), text, { allowQuotes, stepCite });
    if (withCaret) placeCaret(next);
    morph(box, next);
    return box;
  }

  function renderMarkdown(text) {
    return fillMarkdown(el("div", "md"), text);
  }

  /* ---- Structured Normative Response Parser & Collapsible Disclosure (Item #5) ---- */
  function isBriefHeadingText(h) {
    return /^кратк(?:ий|ое)\s+(?:ответ|резюме|вывод|справка)/iu.test(h);
  }

  function isActionsHeadingText(h) {
    return /^(?:порядок\s+действий|краткий\s+список\s+действий|алгоритм\s+действий|шаги(?:\s+выполнения)?|что\s+нужно\s+сделать)/iu.test(
      h
    );
  }

  function isDetailsHeadingText(h) {
    return /^(?:подробн(?:ый|ое)\s+(?:ответ|разбор|обоснование)|полный\s+ответ|обоснование\s+и\s+цитаты|нормативное\s+обоснование|цитаты(?:\s+и\s+источники|\s+из\s+документов)?)/iu.test(
      h
    );
  }

  function extractStrayQuoteBlocks(mdText) {
    const lines = String(mdText || "").split("\n");
    const kept = [];
    const extracted = [];
    let i = 0;
    while (i < lines.length) {
      if (/^\s*>\s*[«“„"]/.test(lines[i])) {
        const qBlock = [];
        while (i < lines.length && /^\s*>/.test(lines[i])) {
          qBlock.push(lines[i]);
          i++;
        }
        extracted.push(qBlock.join("\n"));
      } else {
        kept.push(lines[i]);
        i++;
      }
    }
    return {
      cleaned: kept.join("\n").trim(),
      quotesMd: extracted.join("\n\n").trim(),
    };
  }

  function parseStructuredAnswer(rawMarkdown, { isStreaming = false } = {}) {
    let src = String(rawMarkdown || "").replace(/\r\n/g, "\n");
    if (!src.trim()) {
      return {
        waitingFirstHeading: Boolean(isStreaming),
        briefMd: "",
        actionsMd: "",
        detailsMd: "",
        hasDetails: false,
        detailsForming: false,
        activeSection: "brief",
      };
    }

    let detailsForming = false;
    if (isStreaming) {
      if (/^\s*#{1,4}(?:\s+[^\n]*)?$/.test(src)) {
        return {
          waitingFirstHeading: true,
          briefMd: "",
          actionsMd: "",
          detailsMd: "",
          hasDetails: false,
          detailsForming: false,
          activeSection: "brief",
        };
      }
      const tailMatch = src.match(/\n(\s*#{1,4}(?:\s+([^\n]*))?)$/);
      if (tailMatch) {
        const partialHeading = (tailMatch[2] || "").trim();
        if (/^(?:подроб|полный|обоснован|цитат)/iu.test(partialHeading)) {
          detailsForming = true;
        }
        src = src.slice(0, tailMatch.index);
      }
    }

    const lines = src.split("\n");
    let sawStructuredHeading = false;
    let sawDetailsHeading = false;
    let current = "pre";
    const preLines = [];
    const briefLines = [];
    const actionsLines = [];
    const detailsLines = [];

    for (const line of lines) {
      const m = line.match(/^\s*#{2,3}\s+(.+?)\s*$/);
      if (m) {
        const hText = m[1].replace(/[:.]+$/, "").trim();
        if (!sawDetailsHeading && isBriefHeadingText(hText)) {
          sawStructuredHeading = true;
          current = "brief";
          continue;
        }
        if (!sawDetailsHeading && isActionsHeadingText(hText)) {
          sawStructuredHeading = true;
          current = "actions";
          continue;
        }
        if (isDetailsHeadingText(hText)) {
          sawStructuredHeading = true;
          sawDetailsHeading = true;
          current = "details";
          continue;
        }
      }
      if (current === "pre") preLines.push(line);
      else if (current === "brief") briefLines.push(line);
      else if (current === "actions") actionsLines.push(line);
      else detailsLines.push(line);
    }

    let briefMd = "";
    let actionsMd = "";
    let detailsMd = "";
    let activeSection = current === "pre" ? "brief" : current;

    if (sawStructuredHeading) {
      const preTrim = preLines.join("\n").trim();
      const bTrim = briefLines.join("\n").trim();
      briefMd = [preTrim, bTrim].filter(Boolean).join("\n\n");
      actionsMd = actionsLines.join("\n").trim();
      detailsMd = detailsLines.join("\n").trim();
    } else {
      const fullTrim = src.trim();
      const hasQuotes = /^\s*>\s*[«“„"]/m.test(fullTrim);
      const hasTable = /^\s*\|.+\|\s*\n\s*\|[-:\s|]+\|/m.test(fullTrim);
      if (hasQuotes || (fullTrim.length > 600 && hasTable)) {
        const blocks = fullTrim.split(/\n{2,}/);
        const firstIdx = blocks.findIndex(
          (blk) => !/^\s*(?:>|#{1,4}\s|\|.+\|)/.test(blk.trim()) && blk.trim().length > 0
        );
        if (firstIdx >= 0 && blocks.length > 1) {
          briefMd = blocks[firstIdx].trim();
          detailsMd = blocks
            .filter((_, idx) => idx !== firstIdx)
            .join("\n\n")
            .trim();
          sawDetailsHeading = Boolean(detailsMd);
          activeSection = sawDetailsHeading ? "details" : "brief";
        } else {
          briefMd = fullTrim;
        }
      } else {
        briefMd = fullTrim;
      }
    }

    // Relocate any verbatim quote blocks accidentally emitted inside Brief into Details
    // (Quotes inside Actions are kept and converted into compact .step-src-chip buttons)
    const briefSplit = extractStrayQuoteBlocks(briefMd);
    if (briefSplit.quotesMd && briefSplit.cleaned) {
      briefMd = briefSplit.cleaned;
      detailsMd = [briefSplit.quotesMd, detailsMd].filter(Boolean).join("\n\n");
      sawDetailsHeading = true;
    }

    if (
      !actionsMd ||
      /^(?:нет|не\s+требуется|не\s+применимо|отсутствует|действий\s+не\s+требуется|—|-)\.?$/iu.test(
        actionsMd.trim()
      )
    ) {
      actionsMd = "";
    }

    const hasDetails = Boolean(sawDetailsHeading || detailsForming || detailsMd.trim());
    if (detailsForming) activeSection = "details";

    return {
      waitingFirstHeading: false,
      briefMd,
      actionsMd,
      detailsMd,
      hasDetails,
      detailsForming,
      activeSection,
    };
  }

  let detailsSeq = 0;

  function syncDetailsDisclosure(mdBox) {
    const parts = mdBox && mdBox._parts;
    if (!parts || !parts.detailsWrap) return;
    const expanded = Boolean(mdBox._detailsExpanded);
    const isDetailsStreaming = Boolean(mdBox._detailsStreaming);
    const qCount = Number(mdBox._quoteCount) || 0;

    parts.detailsPanel.hidden = !expanded;
    parts.detailsPanel.classList.toggle("is-open", expanded);
    parts.toggleBtn.setAttribute("aria-expanded", String(expanded));
    parts.pulseEl.hidden = !(isDetailsStreaming && !expanded);

    if (expanded) {
      parts.labelEl.textContent = "Скрыть полный ответ и цитаты";
    } else if (isDetailsStreaming) {
      parts.labelEl.textContent = "Формируется подробный разбор…";
    } else {
      parts.labelEl.textContent = "Показать полный ответ и цитаты";
    }

    if (qCount > 0) {
      parts.badgeEl.hidden = false;
      parts.badgeEl.textContent = `Цитат: ${qCount}`;
    } else {
      parts.badgeEl.hidden = true;
    }
  }

  /** Sanitize version, discrepancy, and quality alerts for user-facing display. */
  function sanitizeAlertText(raw) {
    let s = String(raw || "").replace(/\u00a0/g, " ").trim();
    if (!s) return "";
    // Strip leading ALL-CAPS "ВНИМАНИЕ (...):" or "ВНИМАНИЕ:"
    s = s.replace(/^\s*ВНИМАНИЕ\s*(?:\([^)]*\))?\s*:\s*/u, "");
    // Translate residual technical enum statuses & OCR tags
    s = s
      .replace(/\bsuperseded_template_reference\b/gi, "справочный шаблон")
      .replace(/\bsuperseded\b/gi, "утратил силу")
      .replace(/\bactive\b/gi, "действует")
      .replace(
        /\bOCR[-–—]?\s*(?:распознавани[ея]|текст[а]?|сло[йя])(?=$|[^A-Za-zА-Яа-яЁё0-9_])/giu,
        "скан-копия"
      )
      .replace(/\bOCR\b/g, "скан-копия")
      .replace(/(?:скан-копи[яиейю])(?:\s*\(\s*скан-копи[яиейю]\s*\)|\s+скан-копи[яиейю])+/giu, "скан-копия");
    // Strip uploads/ paths and source origin boilerplate (keep balanced filename if present)
    s = s
      .replace(/,?\s*источник:\s*загруженный документ\s*`?[\w./-]*\/?uploads\/[^`\s,;)]*`?/giu, "")
      .replace(/`[\w./-]*\/?uploads\/([^`\s]*)`/giu, "$1")
      .replace(/[\w./-]*\/?uploads\//giu, "");
    // Strip snake_case doc_ids (safely preserving filenames that carry extensions via trailing \b)
    s = s
      .replace(
        /(?:заменен|отменен)(?:\s*\/\s*(?:заменен|отменен))?\s+(?:документом|актом)\s*:?\s*`?\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b(?![.\w/])`?\.?/giu,
        "заменен более новым актом"
      )
      .replace(/\s*\(\s*`?\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b(?![.\w/])`?\s*\)/g, "")
      .replace(/\bdoc_id\s*[:=]\s*`?[a-z0-9_.-]+`?/gi, "")
      .replace(/\bdoc_id\b/gi, "идентификатор документа")
      .replace(/\s*`?\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b(?![.\w/])`?/g, "");
    // Clean dangling punctuation & whitespace artifacts
    s = s
      .replace(/:\s*(?=[.,;)]|$)/g, "")
      .replace(/\(\s*[,;:\s]*\)/g, "")
      .replace(/\(\s*[,;:]+\s*/g, "(")
      .replace(/\s*[,;:]+\s*\)/g, ")")
      .replace(/\s*,(?:\s*,)+/g, ",")
      .replace(/\s+([,.;:!?])/g, "$1")
      .replace(/\s{2,}/g, " ")
      .replace(/^[\s,;:-]+|[\s,;:-]+$/g, "")
      .trim();
    if (!s) return "";
    return s.charAt(0).toUpperCase() + s.slice(1);
  }

  function sanitizeAlertList(list) {
    const out = [];
    const seen = new Set();
    for (const item of Array.isArray(list) ? list : []) {
      const clean = sanitizeAlertText(item);
      if (!clean) continue;
      const key = normQ(clean) || clean.toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(clean);
    }
    return out;
  }

  function collectSanitizedAlerts({ discrepancyAlerts, versionAlerts, qualityAlerts } = {}) {
    const items = [];
    const seen = new Set();
    const pushGroup = (rawList, kind, badge) => {
      for (const clean of sanitizeAlertList(rawList)) {
        const key = normQ(clean) || clean.toLowerCase();
        if (seen.has(key)) continue;
        seen.add(key);
        items.push({ kind, badge, text: clean });
      }
    };
    pushGroup(discrepancyAlerts, "discrepancy", "Расхождение");
    pushGroup(versionAlerts, "version", "Актуальность");
    pushGroup(qualityAlerts, "quality", "Источник");
    return items;
  }

  function buildAlertBadgeIcon(kind) {
    if (kind === "discrepancy") return ICONS.warn;
    if (kind === "quality") return ICONS.doc;
    return ICONS.alert;
  }

  function updateStructuredAnswerDom(
    mdBox,
    text,
    {
      isStreaming = false,
      quotes = null,
      sources = null,
      hits = null,
      versionAlerts = null,
      discrepancyAlerts = null,
      qualityAlerts = null,
    } = {}
  ) {
    mdBox.classList.add("md", "ans-structured");
    if (typeof mdBox._detailsExpanded !== "boolean") {
      mdBox._detailsExpanded = true;
    }
    if (!mdBox._parts) {
      mdBox._parts = {
        briefFrame: null,
        briefBody: null,
        actionsBox: null,
        actionsBody: null,
        alertsPlate: null,
        detailsWrap: null,
        toggleBtn: null,
        pulseEl: null,
        labelEl: null,
        badgeEl: null,
        detailsPanel: null,
        detailsBody: null,
      };
    }
    const parts = mdBox._parts;
    const parsed = parseStructuredAnswer(text, { isStreaming });
    if (parsed.waitingFirstHeading) {
      return false;
    }

    const effectiveHits = Array.isArray(hits) ? hits : currentStreamHits;
    const effectiveSources = Array.isArray(sources)
      ? sources
      : Array.isArray(effectiveHits) && effectiveHits.length
        ? buildSources(effectiveHits)
        : [];

    // 1. Framed Short Answer (.ans-brief-frame) — always present when non-empty
    const effectiveBrief =
      parsed.briefMd || (!isStreaming && !parsed.hasDetails ? "_Текст справки не получен._" : "");
    if (effectiveBrief || isStreaming) {
      if (!parts.briefFrame) {
        const frame = el("section", "ans-brief-frame");
        frame.setAttribute("aria-label", "Краткий ответ");
        const head = el("div", "ans-section-head");
        head.appendChild(el("span", "ans-section-kicker", "КРАТКИЙ ОТВЕТ"));
        const bBody = el("div", "ans-brief-body");
        frame.append(head, bBody);
        parts.briefFrame = frame;
        parts.briefBody = bBody;
        mdBox.prepend(frame);
      }
      const showBriefCaret = Boolean(
        isStreaming && parsed.activeSection === "brief" && !parsed.actionsMd && !parsed.hasDetails
      );
      patchMarkdown(parts.briefBody, effectiveBrief, showBriefCaret, { allowQuotes: false });
    } else if (parts.briefFrame) {
      parts.briefFrame.remove();
      parts.briefFrame = null;
      parts.briefBody = null;
    }

    // 2. Conditional Brief Action List (.ans-actions-box) — ONLY when procedural steps exist
    if (parsed.actionsMd) {
      if (!parts.actionsBox) {
        const box = el("section", "ans-actions-box");
        box.setAttribute("aria-label", "Порядок действий");
        const head = el("div", "ans-section-head");
        const kicker = el("span", "ans-actions-kicker");
        kicker.innerHTML = ICONS.steps;
        kicker.appendChild(el("span", null, "Порядок действий"));
        head.appendChild(kicker);
        const aBody = el("div", "ans-actions-body");
        box.append(head, aBody);
        parts.actionsBox = box;
        parts.actionsBody = aBody;
        const anchor =
          parts.alertsPlate && parts.alertsPlate.parentNode === mdBox
            ? parts.alertsPlate
            : parts.detailsWrap && parts.detailsWrap.parentNode === mdBox
              ? parts.detailsWrap
              : null;
        if (anchor) {
          mdBox.insertBefore(box, anchor);
        } else {
          mdBox.appendChild(box);
        }
      }
      const showActionsCaret = Boolean(
        isStreaming && parsed.activeSection === "actions" && !parsed.hasDetails
      );
      patchMarkdown(parts.actionsBody, parsed.actionsMd, showActionsCaret, {
        allowQuotes: false,
        stepCite: {
          quotes,
          sources: effectiveSources,
          hits: effectiveHits,
          isStreaming,
        },
      });
    } else if (parts.actionsBox) {
      parts.actionsBox.remove();
      parts.actionsBox = null;
      parts.actionsBody = null;
    }

    // 2.5. Refined Alert Plate (.ans-alerts-plate) right above .ans-details-wrap
    const alertItems = collectSanitizedAlerts({
      discrepancyAlerts,
      versionAlerts,
      qualityAlerts,
    });
    if (alertItems.length > 0) {
      const sig = JSON.stringify(alertItems);
      if (!parts.alertsPlate) {
        const plate = el("aside", "ans-alerts-plate");
        plate.setAttribute("aria-label", "Примечания по актуальности и качеству источников");
        const list = el("ul", "ans-alerts-list");
        plate.appendChild(list);
        parts.alertsPlate = plate;
        if (parts.detailsWrap && parts.detailsWrap.parentNode === mdBox) {
          mdBox.insertBefore(plate, parts.detailsWrap);
        } else {
          mdBox.appendChild(plate);
        }
      }
      if (mdBox._alertsSig !== sig) {
        mdBox._alertsSig = sig;
        parts.alertsPlate.classList.toggle(
          "has-discrepancy",
          alertItems.some((a) => a.kind === "discrepancy")
        );
        parts.alertsPlate.classList.toggle(
          "has-version",
          alertItems.some((a) => a.kind === "version")
        );
        const list = parts.alertsPlate.querySelector(".ans-alerts-list");
        list.innerHTML = "";
        for (const item of alertItems) {
          const li = el("li", `ans-alert-item is-${item.kind}`);
          const badge = el("span", `ans-alert-badge is-${item.kind}`);
          badge.innerHTML = buildAlertBadgeIcon(item.kind);
          badge.appendChild(el("span", null, item.badge));
          const txt = el("span", "ans-alert-text", item.text);
          li.append(badge, txt);
          list.appendChild(li);
        }
      }
    } else if (parts.alertsPlate) {
      parts.alertsPlate.remove();
      parts.alertsPlate = null;
      mdBox._alertsSig = "";
    }

    // 3. Collapsible Full Answer & Quotes (.ans-details-wrap) — open by default
    if (parsed.hasDetails) {
      if (!parts.detailsWrap) {
        const wrap = el("div", "ans-details-wrap");
        const panelId = `ans-details-${++detailsSeq}`;
        const btn = el("button", "ans-toggle-details-btn");
        btn.type = "button";
        btn.setAttribute("aria-expanded", "true");
        btn.setAttribute("aria-controls", panelId);

        const pulse = el("span", "ans-toggle-pulse");
        pulse.hidden = true;
        const lbl = el("span", "ans-toggle-label", "Скрыть полный ответ и цитаты");
        const badge = el("span", "ans-toggle-badge");
        badge.hidden = true;
        btn.append(pulse, lbl, badge);
        btn.insertAdjacentHTML("beforeend", ICONS.chev);

        const panel = el("div", "ans-details-panel is-open");
        panel.id = panelId;
        panel.hidden = false;
        const dBody = el("div", "ans-details-body");
        panel.appendChild(dBody);

        btn.addEventListener("click", () => {
          mdBox._detailsExpanded = !mdBox._detailsExpanded;
          syncDetailsDisclosure(mdBox);
        });

        wrap.append(btn, panel);
        parts.detailsWrap = wrap;
        parts.toggleBtn = btn;
        parts.pulseEl = pulse;
        parts.labelEl = lbl;
        parts.badgeEl = badge;
        parts.detailsPanel = panel;
        parts.detailsBody = dBody;
        mdBox.appendChild(wrap);
      }

      mdBox._detailsStreaming = Boolean(
        isStreaming && (parsed.activeSection === "details" || parsed.detailsForming)
      );
      const showDetailsCaret = Boolean(mdBox._detailsStreaming && mdBox._detailsExpanded);
      patchMarkdown(parts.detailsBody, parsed.detailsMd, showDetailsCaret, { allowQuotes: true });

      const domQuotes = parts.detailsBody.querySelectorAll(".doc-quote").length;
      const apiQuotes = Array.isArray(quotes) ? quotes.length : 0;
      mdBox._quoteCount = domQuotes || apiQuotes;
      syncDetailsDisclosure(mdBox);
    } else if (parts.detailsWrap) {
      parts.detailsWrap.remove();
      parts.detailsWrap = null;
      parts.toggleBtn = null;
      parts.pulseEl = null;
      parts.labelEl = null;
      parts.badgeEl = null;
      parts.detailsPanel = null;
      parts.detailsBody = null;
    }

    return true;
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
        const pg = parsePageFromCitationLabel(source);
        src.title = pg ? `Открыть документ на стр. ${pg}` : "Открыть документ";
        src.innerHTML = ICONS.openDoc;
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
    const str = String(label);
    const m =
      str.match(/(?:^|[\s,;(\u00a0])(?:стр\.|страниц[а-я]*|с\.)\s*(\d+)/iu) ||
      str.match(/(?:^|[\s,;(\u00a0])лист\s+(\d+)(?!\s*:)/iu);
    if (!m) return null;
    const n = parseInt(m[1], 10);
    return Number.isFinite(n) && n >= 1 ? n : null;
  }

  function parseSheetFromLabel(label) {
    if (!label) return "";
    const m = String(label).match(/(?:^|[\s,;(\u00a0])лист\s*:\s*([^,;()\]]+)/iu);
    return m ? m[1].trim() : "";
  }

  function parseDocNumberFromLabel(label) {
    const cleaned = cleanSource(label || "").replace(
      /(?:редакци[яию]|ред\.)\s*№\s*[0-9А-Яа-яЁёA-Za-z./-]+/giu,
      ""
    );
    const m = cleaned.match(/№\s*([0-9А-Яа-яЁёA-Za-z][0-9А-Яа-яЁёA-Za-z./-]*)/u);
    if (!m) return "";
    return cleanDocNo(m[1].replace(/[./-]+$/, ""));
  }

  function parseClauseFromLabel(label) {
    if (!label) return "";
    const m = String(label).match(
      /(?:^|[\s,;(\u00a0])(?:пп?\.|пункт[а-я]*)\s*([0-9]+(?:\.[0-9]+)*)/iu
    );
    return m ? m[1] : "";
  }

  function stripCitationLocators(raw) {
    let s = cleanSource(raw || "").replace(/\.(pdf|docx?|xlsx?)$/i, "");
    const tailRe =
      /\s*[,;]\s*(?:№\s*[^,;]+|от\s+\d{1,2}[./-]\d{1,2}[./-]\d{2,4}(?:\s*г\.?)?|лист\s*:[^,;]*|(?:пп?\.|пункт[а-я]*|подпункт[а-я]*|разд(?:ел)?\.?|глав[а-я]*|ст\.|стать[яие]|ч\.|част[ьи]|абз(?:ац)?\.?|стр\.?|страниц[а-я]*|с\.|лист[а-я]*|табл(?:ица)?\.?)(?:\s+|\s*(?=[0-9IVXLCDM№]))[^,;]*)\s*$/iu;
    let guard = 0;
    while (guard++ < 10 && tailRe.test(s)) {
      s = s.replace(tailRe, "").trim();
    }
    return s.replace(/^[\s,;.«»"“”'—–-]+|[\s,;.«»"“”'—–-]+$/g, "").trim();
  }

  const GENERIC_DOC_WORDS = new Set([
    "мфти",
    "приказ",
    "приказа",
    "положение",
    "положения",
    "регламент",
    "регламента",
    "порядок",
    "порядка",
    "правила",
    "правил",
    "инструкция",
    "инструкции",
    "документ",
    "утверждении",
    "редакция",
    "федерального",
    "государственного",
    "автономного",
    "образовательного",
    "учреждения",
    "высшего",
    "образования",
  ]);

  function extractTitleTokens(normStr) {
    return String(normStr || "")
      .split(" ")
      .filter((w) => w.length >= 4 || (w.length >= 2 && /\d/.test(w)));
  }

  function scoreTitleSimilarity(wantRaw, candRaw) {
    const stripNumDate = (s) =>
      stripCitationLocators(s)
        .replace(/\bот\s+\d{1,2}[./-]\d{1,2}[./-]\d{2,4}(?:\s*г\.?)?/giu, " ")
        .replace(/№\s*[0-9А-Яа-яЁёA-Za-z./-]+/gu, " ");
    const wFull = normQ(stripCitationLocators(wantRaw));
    const cFull = normQ(stripCitationLocators(candRaw));
    if (!wFull || !cFull) return 0;
    if (wFull === cFull) return 9;

    const wCore = normQ(stripNumDate(wantRaw)) || wFull;
    const cCore = normQ(stripNumDate(candRaw)) || cFull;
    if (wCore && cCore && wCore === cCore) return 8.5;

    let sc = 0;
    if (wCore.length >= 6 && cCore.length >= 6 && (cCore.includes(wCore) || wCore.includes(cCore))) {
      sc += 5.5;
    } else if (
      wCore.length >= 16 &&
      cCore.length >= 16 &&
      (cCore.includes(wCore.slice(0, 24)) || wCore.includes(cCore.slice(0, 24)))
    ) {
      sc += 3.5;
    }

    const wTokens = extractTitleTokens(wCore);
    const cTokens = extractTitleTokens(cCore);
    if (wTokens.length && cTokens.length) {
      const cSet = new Set(cTokens);
      for (const wt of wTokens) {
        const weight = GENERIC_DOC_WORDS.has(wt) ? 0.5 : 2.0;
        if (cSet.has(wt)) {
          sc += weight;
        } else if (wt.length >= 5) {
          const stem = wt.slice(0, 5);
          if (cTokens.some((ct) => ct.length >= 5 && ct.slice(0, 5) === stem)) {
            sc += weight * 0.8;
          }
        }
      }
    }
    return sc;
  }

  function matchBestSource(q, sources, labelText) {
    const rawLabel = labelText || (q && (q.source_label || q.source_title)) || "";
    const qDocId = (q && (q.doc_id || q.matched_doc_id)) || "";
    const rawDocNum = cleanDocNo((q && q.doc_number) || parseDocNumberFromLabel(rawLabel));
    const wantNum = normQ(rawDocNum);
    const wantTitleRaw = (q && q.source_title) || rawLabel;
    const wantClause = String((q && q.clause) || parseClauseFromLabel(rawLabel) || "")
      .replace(/^(?:пп?\.|пункт[а-я]*)\s*/iu, "")
      .replace(/\.$/, "")
      .trim();

    let best = null;
    let bestSc = 0;
    for (const s of sources || []) {
      let sc = 0;
      if (qDocId && s.docId && s.docId === qDocId) sc += 12;
      const sNum = normQ(s.docNumber);
      if (wantNum && sNum) {
        if (sNum === wantNum) sc += 8;
        else if (wantNum.length >= 3 && (sNum.includes(wantNum) || wantNum.includes(sNum))) sc += 5;
      }
      sc += scoreTitleSimilarity(wantTitleRaw, s.name);

      if (sc > 0 && wantClause) {
        const sCl = String(s.rawClause || s.clause || "")
          .replace(/^(?:пп?\.|пункт[а-я]*)\s*/iu, "")
          .replace(/\.$/, "")
          .trim();
        if (sCl && sCl === wantClause) sc += 2.5;
      }
      if (sc > bestSc) {
        best = s;
        bestSc = sc;
      }
    }
    return bestSc >= 2 ? best : null;
  }

  function resolveQuoteViewerTarget(q, sources, labelText, cardQuoteText) {
    const rawLabel = labelText || (q && q.source_label) || "";
    const bestSrc = matchBestSource(q, sources, rawLabel);
    const docId =
      (q && (q.doc_id || q.matched_doc_id)) || (bestSrc && bestSrc.docId) || "";
    const page =
      (q && parseFirstPageInt(q.page_num)) ||
      (q && parseFirstPageInt(q.page)) ||
      parsePageFromCitationLabel(rawLabel) ||
      (bestSrc && Array.isArray(bestSrc.pages) && bestSrc.pages[0]) ||
      null;
    const sheetName =
      (q && q.sheet_name) ||
      parseSheetFromLabel(rawLabel) ||
      parseSheetFromLabel(q && q.clause) ||
      (bestSrc && bestSrc.sheetName) ||
      "";
    const clause =
      (q && q.clause) ||
      parseClauseFromLabel(rawLabel) ||
      (sheetName ? `Лист: ${sheetName}` : "") ||
      (bestSrc && bestSrc.rawClause) ||
      "";
    const quote = (q && q.text) || cardQuoteText || "";
    const docNumber = cleanDocNo(
      (q && q.doc_number) ||
        parseDocNumberFromLabel(rawLabel) ||
        (bestSrc && bestSrc.docNumber) ||
        ""
    );
    const fallbackTitle =
      stripCitationLocators((q && q.source_title) || "") ||
      stripCitationLocators(rawLabel) ||
      (bestSrc && bestSrc.name) ||
      cleanSource(rawLabel) ||
      "";
    return { docId, page, quote, clause, sheetName, docNumber, fallbackTitle };
  }

  /* ---------------- Clickable Step Source Chips (### Порядок действий) ---------------- */
  const STEP_CITE_STRIP_PREFIX_RE =
    /^(?:см\.?\s+|согласно\s+|в\s+соответствии\s+с\s+|на\s+основании\s+|источник\s*:\s*)/iu;

  function isCitationExpression(rawInner) {
    const s = cleanSource(String(rawInner || "").replace(STEP_CITE_STRIP_PREFIX_RE, "")).trim();
    if (!s || s.length < 3 || s.length > 240) return false;
    if (/https?:\/\/|mailto:|www\./i.test(s)) return false;

    const hasDocNum = /№\s*[0-9А-Яа-яЁёA-Za-z][0-9А-Яа-яЁёA-Za-z./-]*/u.test(s);
    const hasClause =
      /(?:^|[\s,;«"„(])(?:пп?\.|пункт[а-я]*|подпункт[а-я]*)\s*\d+(?:\.\d+)*/iu.test(s);
    const hasPage =
      /(?:^|[\s,;«"„(])(?:стр\.|страниц[а-я]*|с\.)\s*\d+/iu.test(s);
    const hasSheet =
      /(?:^|[\s,;«"„(])(?:лист\s*:\s*[^\s,;()[\]]|лист\s+\d+(?:\s*(?:,|$)))/iu.test(s);
    const hasStructPart =
      /(?:^|[\s,;«"„(])(?:разд(?:ел[а-я]*)?\.?|глав[а-я]*|табл(?:иц[а-я]*)?\.?|прил(?:ожени[еяюий]*)?\.?)\s*(?:№\s*)?(?:\d+|[IVXLCDM]+\b)/iu.test(
        s
      );
    const hasDocDate =
      /(?:^|[^A-Za-zА-Яа-яЁё0-9_])(?:приказ[а-я]*|распоряжени[еяю]|регламент[а-я]*|положени[еяю]|поряд[оа-я]*|инструкци[яию]|устав[а-я]*|правил[а-я]*)(?=$|[^A-Za-zА-Яа-яЁё0-9_])/iu.test(
        s
      ) &&
      (/(?:^|[^A-Za-zА-Яа-яЁё0-9_])от\s+\d{1,2}[./-]\d{1,2}[./-]\d{2,4}/iu.test(s) ||
        hasDocNum ||
        hasClause ||
        hasPage ||
        hasSheet ||
        hasStructPart);

    if (!hasDocNum && !hasClause && !hasPage && !hasSheet && !hasStructPart && !hasDocDate) {
      return false;
    }
    if (
      !hasDocNum &&
      /^(?:например|т\.?\s*е\.|если|когда|кроме)(?=$|[^A-Za-zА-Яа-яЁё0-9_])/iu.test(s) &&
      s.length > 55
    ) {
      return false;
    }
    return true;
  }

  function findBalancedCitationSpans(str) {
    const spans = [];
    const s = String(str || "");
    let i = 0;
    while (i < s.length) {
      const ch = s[i];
      if (ch !== "(" && ch !== "[") {
        i++;
        continue;
      }
      const openCh = ch;
      const closeCh = openCh === "(" ? ")" : "]";
      let depth = 1;
      let j = i + 1;
      while (j < s.length && j - i <= 260 && depth > 0) {
        if (s[j] === openCh) depth++;
        else if (s[j] === closeCh) depth--;
        j++;
      }
      if (depth === 0) {
        const closeIdx = j - 1;
        const inner = s.slice(i + 1, closeIdx);
        if (isCitationExpression(inner)) {
          spans.push({ start: i, end: closeIdx + 1, inner });
          i = closeIdx + 1;
          continue;
        }
        if (/[(\[]/.test(inner)) {
          i++;
          continue;
        }
        i = closeIdx + 1;
      } else {
        i++;
      }
    }
    return spans;
  }

  function resolveStepCitationTarget(rawCitation, { quotes, sources, hits, quoteText = "" } = {}) {
    const cleaned = cleanSource(
      String(rawCitation || "").replace(STEP_CITE_STRIP_PREFIX_RE, "")
    ).trim();
    const docNumber = cleanDocNo(parseDocNumberFromLabel(cleaned));
    const clause = parseClauseFromLabel(cleaned);
    const sheetName = parseSheetFromLabel(cleaned);
    const page = parsePageFromCitationLabel(cleaned);
    const titleQuery = stripCitationLocators(cleaned);

    const normClause = String(clause || "")
      .replace(/^(?:пп?\.|пункт[а-я]*)\s*/iu, "")
      .replace(/\.$/, "")
      .trim();
    const normSheet = normQ(sheetName);
    const wantNum = normQ(docNumber);
    const wantQuote = normQ(quoteText);

    const candidates = [];
    for (const q of quotes || []) {
      const qPages = [];
      const p1 = parseFirstPageInt(q.page_num);
      const p2 = parseFirstPageInt(q.page);
      if (p1) qPages.push(p1);
      if (p2 && !qPages.includes(p2)) qPages.push(p2);
      candidates.push({
        docId: q.doc_id || q.matched_doc_id || "",
        docNumber: cleanDocNo(q.doc_number || parseDocNumberFromLabel(q.source_label || "")),
        title: stripCitationLocators(q.source_title || q.source_label || ""),
        clause: String(q.clause || parseClauseFromLabel(q.source_label || "") || "").trim(),
        sheetName: q.sheet_name || parseSheetFromLabel(q.source_label || q.clause || "") || "",
        pages: qPages,
        quoteText: q.text || "",
      });
    }
    for (const s of sources || []) {
      candidates.push({
        docId: s.docId || "",
        docNumber: cleanDocNo(s.docNumber || ""),
        title: s.name || "",
        clause: String(s.rawClause || s.clause || "").trim(),
        sheetName: s.sheetName || parseSheetFromLabel(s.rawClause || s.clause || "") || "",
        pages: Array.isArray(s.pages) ? s.pages : [],
        quoteText: "",
      });
    }
    for (const h of hits || []) {
      const hTitle = String(h.title || h.file_name || "").replace(/\.(pdf|docx?|xlsx?)$/i, "");
      const hSheet = String(h.sheet_name || (h.metadata && h.metadata.sheet_name) || "").trim();
      candidates.push({
        docId: h.doc_id || "",
        docNumber: cleanDocNo(h.doc_number || ""),
        title: hTitle,
        clause: String(h.clause_no || "").trim(),
        sheetName: hSheet,
        pages: Array.isArray(h.page_numbers) ? h.page_numbers : [],
        quoteText: h.clean_text || "",
      });
    }

    let best = null;
    let bestSc = 0;
    for (const c of candidates) {
      let sc = 0;
      const cNum = normQ(c.docNumber);
      if (wantNum && cNum) {
        if (cNum === wantNum) sc += 8;
        else if (wantNum.length >= 3 && (cNum.includes(wantNum) || wantNum.includes(cNum))) sc += 5;
      }
      if (titleQuery) {
        sc += scoreTitleSimilarity(titleQuery, c.title);
      }
      const cCl = String(c.clause || "")
        .replace(/^(?:пп?\.|пункт[а-я]*)\s*/iu, "")
        .replace(/\.$/, "")
        .trim();
      if (normClause && cCl) {
        if (cCl === normClause) sc += 3.5;
        else if (cCl.startsWith(normClause + ".") || normClause.startsWith(cCl + ".")) sc += 2.0;
      }
      const cSheet = normQ(c.sheetName);
      if (normSheet && cSheet) {
        if (cSheet === normSheet) sc += 3.5;
        else if (cSheet.includes(normSheet) || normSheet.includes(cSheet)) sc += 2.0;
      }
      if (page && Array.isArray(c.pages) && c.pages.includes(page)) {
        sc += 1.5;
      }
      if (wantQuote && c.quoteText) {
        const cQ = normQ(c.quoteText);
        if (cQ && (cQ === wantQuote || cQ.includes(wantQuote.slice(0, 36)))) sc += 6;
      }
      if (sc > bestSc) {
        bestSc = sc;
        best = c;
      }
    }

    const hasDocSpec = Boolean(wantNum || (titleQuery && titleQuery.length >= 4));
    const minScore = hasDocSpec ? 2.0 : 1.5;
    if (bestSc < minScore) {
      best = null;
    }
    if (!best && !hasDocSpec) {
      const uniqueIds = [...new Set(candidates.map((c) => c.docId).filter(Boolean))];
      if (uniqueIds.length === 1) {
        best = candidates.find((c) => c.docId === uniqueIds[0]) || null;
      }
    }

    const resDocId = (best && best.docId) || "";
    const resPage = page || (best && best.pages && best.pages[0]) || null;
    const resSheet = sheetName || (best && best.sheetName) || "";
    const resClause =
      clause ||
      (resSheet ? `Лист: ${resSheet}` : "") ||
      (best && best.clause) ||
      "";
    const resDocNumber = docNumber || (best && best.docNumber) || "";
    const resTitle =
      (best && best.title) ||
      titleQuery ||
      "";
    const resQuote = quoteText || (best && best.quoteText) || "";

    return {
      docId: resDocId,
      page: resPage,
      clause: resClause,
      sheetName: resSheet,
      docNumber: resDocNumber,
      fallbackTitle: resTitle,
      quote: !resPage ? resQuote : quoteText || "",
    };
  }

  function compactStepChipLabel(raw) {
    const clean = cleanSource(String(raw || "").replace(STEP_CITE_STRIP_PREFIX_RE, "")).trim();
    if (!clean) return "Открыть источник";
    let display = clean;
    if (display.length > 56) {
      const locMatch = display.match(
        /(?:,\s*|\s+)(?:№\s*[0-9А-Яа-яЁёA-Za-z]|от\s+\d{1,2}[./-]\d|пп?\.|пункт|лист\s*:|разд\.|табл\.|прил\.|стр\.|с\.\s*\d)/iu
      );
      if (locMatch && locMatch.index >= 18) {
        const head = display.slice(0, locMatch.index).trim();
        const tail = display.slice(locMatch.index).replace(/^\s*,\s*/, "").trim();
        if (head.length > 30) {
          display = `${head.slice(0, 28).trimEnd()}…, ${tail}`;
        }
      } else if (display.length > 64) {
        display = display.slice(0, 62).trimEnd() + "…";
      }
    }
    return display.replace(/(№|пп?\.|стр\.|с\.|разд\.|табл\.|прил\.)\s+(?=\S)/giu, "$1\u00a0");
  }

  function stepTargetKey(t) {
    if (!t) return "";
    return `${t.docId || ""}|${normQ(t.docNumber || "")}|${String(t.clause || "").trim()}|${t.page || ""}`;
  }

  function readStepChipTarget(chip) {
    if (!chip) return {};
    const d = chip.dataset || {};
    const hasDatasetTarget = Boolean(
      d.docId || d.page || d.clause || d.sheetName || d.docNumber || d.fallbackTitle || d.quote
    );
    if (hasDatasetTarget) {
      return {
        docId: d.docId || "",
        page: d.page ? Number(d.page) : null,
        clause: d.clause || "",
        sheetName: d.sheetName || "",
        docNumber: d.docNumber || "",
        fallbackTitle: d.fallbackTitle || "",
        quote: d.quote || "",
      };
    }
    return chip._viewerTarget || {};
  }

  function createStepChipElement(rawLabel, target, { pending = false } = {}) {
    const fullText = cleanSource(
      String(rawLabel || "").replace(STEP_CITE_STRIP_PREFIX_RE, "")
    ).trim();
    const displayText = compactStepChipLabel(fullText || "Открыть источник");
    const btn = el("button", "step-src-chip" + (pending ? " is-pending" : ""));
    btn.type = "button";
    btn.innerHTML = ICONS.openDoc;
    btn.appendChild(el("span", "step-src-label", displayText));

    const canOpen = Boolean(
      !pending && target && (target.docId || target.docNumber || target.fallbackTitle)
    );
    btn.disabled = !canOpen;
    if (target) {
      btn._viewerTarget = target;
      if (target.docId) btn.dataset.docId = target.docId;
      if (target.page) btn.dataset.page = String(target.page);
      if (target.clause) btn.dataset.clause = target.clause;
      if (target.sheetName) btn.dataset.sheetName = target.sheetName;
      if (target.docNumber) btn.dataset.docNumber = target.docNumber;
      if (target.fallbackTitle) btn.dataset.fallbackTitle = target.fallbackTitle;
      if (target.quote) btn.dataset.quote = String(target.quote).slice(0, 500);
    }
    const pageHint = target && target.page ? ` (стр. ${target.page})` : "";
    const ariaTitle = fullText ? `Открыть источник: ${fullText}${pageHint}` : "Открыть документ";
    btn.title = ariaTitle;
    btn.setAttribute("aria-label", ariaTitle);
    return btn;
  }

  function decorateActionStepSources(box, { quotes, sources, hits, isStreaming = false } = {}) {
    // 1. Convert any inline parenthetical/bracketed citations in text nodes into .step-src-chip
    const walker = document.createTreeWalker(box, NodeFilter.SHOW_TEXT);
    const textNodes = [];
    let cur;
    while ((cur = walker.nextNode())) {
      const parent = cur.parentElement;
      if (!parent || parent.closest("code, pre, a, button, blockquote, .step-src-chip, .step-src-chips")) {
        continue;
      }
      if (cur.nodeValue && /[(\[]/.test(cur.nodeValue)) {
        textNodes.push(cur);
      }
    }

    for (const tNode of textNodes) {
      const str = tNode.nodeValue || "";
      const spans = findBalancedCitationSpans(str);
      if (!spans.length) continue;

      const resolvedSpans = [];
      for (const sp of spans) {
        const target = resolveStepCitationTarget(sp.inner, { quotes, sources, hits });
        if (target && (target.docId || target.docNumber || target.fallbackTitle)) {
          resolvedSpans.push({ ...sp, target });
        }
      }
      if (!resolvedSpans.length) continue;

      const frag = document.createDocumentFragment();
      let cursor = 0;
      for (const sp of resolvedSpans) {
        if (sp.start > cursor) {
          let before = str.slice(cursor, sp.start);
          if (before && !/[\s(«"„]$/.test(before)) before += " ";
          frag.appendChild(document.createTextNode(before));
        } else if (cursor === 0 && sp.start === 0) {
          /* starts right at citation */
        }
        frag.appendChild(createStepChipElement(sp.inner, sp.target));
        cursor = sp.end;
      }
      if (cursor < str.length) {
        frag.appendChild(document.createTextNode(str.slice(cursor)));
      }
      tNode.replaceWith(frag);
    }

    // 2. Convert any <blockquote> inside ### Порядок действий into a compact .step-src-chip on its step <li>
    const blockquotes = [...box.querySelectorAll("blockquote")];
    for (const bq of blockquotes) {
      if (bq.parentElement && bq.parentElement.closest("blockquote")) continue;
      const raw = bq.textContent.replace(/^\s+/, "");
      const joined = Array.from(bq.querySelectorAll("p"))
        .map((p) => p.textContent)
        .join("\n");
      const { text, source } = splitQuote(joined || raw);
      const looksLikeQuote = Boolean((raw && OPEN_Q.includes(raw[0])) || source);
      if (!looksLikeQuote) continue;

      const target = resolveStepCitationTarget(source, {
        quotes,
        sources,
        hits,
        quoteText: text,
      });
      const fallbackLabel =
        source ||
        [
          target.fallbackTitle,
          target.docNumber ? `№ ${target.docNumber}` : "",
          target.clause
            ? /^лист\s*:/iu.test(target.clause)
              ? target.clause
              : `п. ${target.clause}`
            : "",
          target.page ? `стр. ${target.page}` : "",
        ]
          .filter(Boolean)
          .join(", ") ||
        (isStreaming ? "Сверка источника…" : "Открыть источник");

      const canOpen = Boolean(target.docId || target.docNumber || target.fallbackTitle);
      const pending = Boolean(isStreaming && !canOpen);
      const chip = createStepChipElement(fallbackLabel, target, { pending });

      const findDirectChipsWrap = (li) =>
        [...li.querySelectorAll(".step-src-chips")].find(
          (w) => w.parentElement === li || (w.parentElement && w.parentElement.tagName === "P" && w.parentElement.parentElement === li)
        ) || null;
      const findLastDirectP = (li) => {
        const ps = [...li.children].filter((c) => c.tagName === "P");
        return ps.length ? ps[ps.length - 1] : null;
      };

      let hostLi = bq.closest("li");
      if (hostLi) {
        const key = stepTargetKey(target);
        const alreadyHas =
          key &&
          [...hostLi.querySelectorAll(".step-src-chip")].some(
            (existing) => stepTargetKey(readStepChipTarget(existing)) === key
          );
        if (alreadyHas) {
          bq.remove();
          continue;
        }
        let wrap = findDirectChipsWrap(hostLi);
        if (!wrap) {
          wrap = el("span", "step-src-chips");
          const lastP = findLastDirectP(hostLi);
          if (lastP && bq.previousElementSibling === lastP) {
            lastP.append(" ", wrap);
            bq.remove();
          } else {
            bq.replaceWith(wrap);
          }
        } else {
          bq.remove();
        }
        wrap.appendChild(chip);
      } else {
        const prev = bq.previousElementSibling;
        const next = bq.nextElementSibling;
        if (prev && (prev.tagName === "OL" || prev.tagName === "UL") && prev.lastElementChild) {
          hostLi = prev.lastElementChild;
          const key = stepTargetKey(target);
          const alreadyHas =
            key &&
            [...hostLi.querySelectorAll(".step-src-chip")].some(
              (existing) => stepTargetKey(readStepChipTarget(existing)) === key
            );
          if (!alreadyHas) {
            let wrap = findDirectChipsWrap(hostLi);
            if (!wrap) {
              wrap = el("span", "step-src-chips");
              const lastP = findLastDirectP(hostLi);
              if (lastP) lastP.append(" ", wrap);
              else hostLi.append(" ", wrap);
            }
            wrap.appendChild(chip);
          }
          bq.remove();
          if (next && next.tagName === "OL" && prev.tagName === "OL") {
            while (next.firstElementChild) prev.appendChild(next.firstElementChild);
            next.remove();
          }
        } else {
          const wrap = el("div", "step-src-chips");
          wrap.appendChild(chip);
          bq.replaceWith(wrap);
        }
      }
    }
  }

  /** Attach verification seals, normalized source metadata, full-clause disclosure & Document Viewer trigger. */
  function applyQuoteMeta(box, quotes, sources) {
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
            btn.innerHTML = ICONS.openDoc;
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
            ? /^лист\s*:/iu.test(String(q.clause))
              ? `Показать ${q.clause} целиком`
              : `Показать пункт ${q.clause} целиком`
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
      card._viewerTarget = target;
      const canOpen = Boolean(target.docId || target.docNumber || target.fallbackTitle);

      if (!actionsWrap.querySelector(".dq-open-doc") && canOpen) {
        const openBtn = el("button", "dq-open-doc");
        openBtn.type = "button";
        openBtn.innerHTML = ICONS.openDoc;
        openBtn.appendChild(el("span", null, "Открыть документ"));
        if (target.page) {
          openBtn.appendChild(el("span", "dq-open-page", `стр.\u00a0${target.page}`));
        }
        openBtn.title = target.page
          ? `Открыть документ на стр. ${target.page}`
          : "Открыть документ";
        actionsWrap.appendChild(openBtn);
      }

      if (btn) {
        if (canOpen) {
          btn.disabled = false;
          btn.title = target.page
            ? `Открыть документ на стр. ${target.page}`
            : "Открыть документ";
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

  /* ---------------- Preload Primary Font (Minsk) ---------------- */
  delete document.documentElement.dataset.font;
  if (document.fonts && typeof document.fonts.load === "function") {
    document.fonts.load('400 16px "Minsk"').catch(() => {});
  }

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
  let currentStreamHits = [];
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
      sheet_name: q.sheet_name || "",
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
    currentStreamHits = [];
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
          versionAlerts: Array.isArray(turn.versionAlerts) ? turn.versionAlerts : [],
          discrepancyAlerts: Array.isArray(turn.discrepancyAlerts) ? turn.discrepancyAlerts : [],
          qualityAlerts: Array.isArray(turn.qualityAlerts) ? turn.qualityAlerts : [],
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
      currentStreamHits = [];
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

  function scrollToViewerPage(pageNum, { instant = false, focusAnchor = false } = {}) {
    const targetNum = Math.max(1, Math.min(viewerState.totalPages, Number(pageNum) || 1));
    syncViewerNavControls(targetNum);

    const pageEl = dvBody.querySelector(`[data-page="${targetNum}"]`);
    if (!pageEl) return;

    const pageTop = yWithinBody(pageEl);
    let desiredTop = Math.max(0, pageTop - 10);

    if (focusAnchor) {
      const anchor = pageEl.querySelector("[data-scroll-anchor]");
      if (anchor) {
        const anchorY = yWithinBody(anchor);
        const anchorBottom = anchorY + (anchor.getBoundingClientRect().height || 24);
        if (anchorBottom - desiredTop > dvBody.clientHeight * 0.78) {
          desiredTop = Math.max(0, anchorY - dvBody.clientHeight * 0.25);
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

  /* ---- Normalized Substring Anchor Finder for Long Word / Excel Sheets ---- */
  function buildQuoteNeedles(quote) {
    const raw = String(quote || "").trim();
    if (!raw) return [];
    const parts = raw
      .split(/\s*(?:\.{3,}|…|\[\s*(?:\.{3}|…)\s*\])\s*/)
      .map((s) => normMatchString(s))
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

  function normMatchString(str) {
    return String(str || "")
      .replace(/\u00ad/g, "")
      .toLowerCase()
      .replace(/ё/g, "е")
      .replace(/[^0-9a-zа-я]+/g, " ")
      .trim();
  }

  function textMatchesNeedles(text, needles) {
    if (!text || !needles || !needles.length) return false;
    const norm = normMatchString(text);
    if (!norm) return false;
    return needles.some((needle) => norm.includes(needle));
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
    const effNum = cleanDocNo(docNumber || parseDocNumberFromLabel(fallbackTitle));
    const wantNum = normQ(effNum);
    let best = null;
    let bestSc = 0;
    for (const d of documents) {
      let sc = 0;
      const dNum = normQ(cleanDocNo(d.doc_number));
      if (wantNum && dNum) {
        if (dNum === wantNum) sc += 10;
        else if (wantNum.length >= 3 && (dNum.includes(wantNum) || wantNum.includes(dNum))) sc += 6;
      }
      const titleSc = Math.max(
        scoreTitleSimilarity(fallbackTitle, d.title || ""),
        scoreTitleSimilarity(fallbackTitle, d.file_name || "")
      );
      sc += titleSc;
      if (sc > 0 && String(d.status || "active") === "active") {
        sc += 0.15;
      }
      if (sc > bestSc) {
        best = d;
        bestSc = sc;
      }
    }
    return best && bestSc >= 2 ? best.doc_id : "";
  }

  async function openDocumentViewer({
    docId,
    page,
    quote,
    clause,
    sheetName,
    docNumber,
    fallbackTitle,
  } = {}) {
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

    dvTitle.textContent =
      stripCitationLocators(fallbackTitle) || cleanSource(fallbackTitle) || "Загрузка документа…";
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
      renderViewerPayload(data, { requestedPage: page, quote, clause, sheetName });
    } catch (err) {
      if (reqId !== viewerState.reqId) return;
      dvBody.innerHTML = "";
      dvBody.appendChild(
        el(
          "div",
          "dv-status-box",
          sanitizeUserErrorMessage(err, "Не удалось загрузить предпросмотр документа.")
        )
      );
    }
  }

  function renderViewerPayload(data, { requestedPage, quote, clause, sheetName }) {
    const fmt = String(data.format || data.file_format || "pdf").toLowerCase();
    const known = ["pdf", "docx", "doc", "xlsx", "xls"].includes(fmt);
    const isExcel = fmt === "xlsx" || fmt === "xls";
    const pages = Array.isArray(data.pages) ? data.pages : [];
    const hasPageImages =
      !isExcel &&
      (fmt === "pdf" ||
        Boolean(data.has_page_images) ||
        pages.some((p) => Boolean(p && p.image_url)));
    const total = hasPageImages
      ? Math.max(
          1,
          Number(data.preview_pages_total) || Number(data.total_pages) || pages.length || 1
        )
      : Math.max(1, Number(data.total_pages) || pages.length || 1);
    let targetPage = Math.min(
      total,
      Math.max(1, Number(data.target_page) || Number(requestedPage) || 1)
    );
    if (isExcel) {
      const normSheetLabel = (val) =>
        normQ(String(val || "").replace(/^лист\s*:\s*/iu, "").trim());
      const wantSheetRaw =
        parseSheetFromLabel(sheetName) ||
        String(sheetName || "")
          .replace(/^лист\s*:\s*/iu, "")
          .split(/[,;]/)[0]
          .trim() ||
        parseSheetFromLabel(clause) ||
        (String(clause || "").match(/^лист\s*:\s*(.+)$/iu) || [])[1] ||
        "";
      const wantSheet = normSheetLabel(wantSheetRaw);
      if (wantSheet) {
        let matchedSheet = pages.find((p) => normSheetLabel(p.label) === wantSheet);
        if (!matchedSheet) {
          let bestLen = 0;
          for (const p of pages) {
            const lbl = normSheetLabel(p.label);
            if (!lbl) continue;
            if (lbl.includes(wantSheet) || wantSheet.includes(lbl)) {
              const score = Math.min(lbl.length, wantSheet.length);
              if (score > bestLen) {
                bestLen = score;
                matchedSheet = p;
              }
            }
          }
        }
        if (matchedSheet && Number(matchedSheet.page_number) >= 1) {
          targetPage = Math.min(total, Math.max(1, Number(matchedSheet.page_number)));
        }
      }
    }

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
    if (!pages.length && !hasPageImages) {
      dvBody.appendChild(el("div", "dv-status-box", "Содержимое документа пусто."));
      syncViewerNavControls(1);
      return;
    }

    const frag = document.createDocumentFragment();

    if (hasPageImages) {
      const pageMap = new Map();
      pages.forEach((p, idx) => {
        const num = Number(p && p.page_number) || idx + 1;
        pageMap.set(num, p);
      });

      for (let pNum = 1; pNum <= total; pNum++) {
        const p = pageMap.get(pNum) || {};
        const card = el("div", "dv-page-card dv-pdf-page");
        card.dataset.page = String(pNum);

        const bar = el("div", "dv-page-bar");
        bar.appendChild(el("span", null, p.label || `Стр. ${pNum}`));
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
        const rawImgUrl =
          p.image_url ||
          `/api/documents/${encodeURIComponent(viewerState.docId)}/pages/${pNum}.png`;
        img.src = rawImgUrl.includes("?") ? rawImgUrl : `${rawImgUrl}?dpi=150`;
        img.addEventListener(
          "load",
          () => {
            if (img.naturalWidth > 0 && img.naturalHeight > 0) {
              stage.style.aspectRatio = `${img.naturalWidth} / ${img.naturalHeight}`;
            }
          },
          { once: true }
        );
        stage.appendChild(img);

        card.appendChild(stage);
        frag.appendChild(card);
      }
    } else {
      const needles = buildQuoteNeedles(quote);
      const cleanClause = String(clause || "")
        .replace(/^(?:пп?\.|пункт)\s*/i, "")
        .replace(/\.$/, "")
        .trim();

      pages.forEach((p) => {
        const pNum = Number(p.page_number) || 1;
        const sheet = el("article", "dv-folio-sheet");
        sheet.dataset.page = String(pNum);

        const head = el("div", "dv-folio-head");
        head.appendChild(
          el("span", null, p.label || (isExcel ? `Лист ${pNum}` : `Стр. ${pNum}`))
        );
        sheet.appendChild(head);

        let anchorAssigned = false;
        const markAnchor = (node) => {
          if (!anchorAssigned && pNum === targetPage) {
            node.dataset.scrollAnchor = "1";
            anchorAssigned = true;
          }
        };

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
            const tbl = el("table", isExcel ? "dv-table dv-xlsx-table" : "dv-table");
            if (Array.isArray(b.headers) && b.headers.length) {
              const thead = el("thead");
              const tr = el("tr");
              b.headers.forEach((cell) => {
                const cellText = String(cell ?? "");
                const th = el("th", null, cellText);
                if (textMatchesNeedles(cellText, needles)) markAnchor(tr);
                tr.appendChild(th);
              });
              thead.appendChild(tr);
              tbl.appendChild(thead);
            }
            if (Array.isArray(b.rows) && b.rows.length) {
              const tbody = el("tbody");
              const rowNeedles = isExcel && quote
                ? [
                    ...needles,
                    ...String(quote)
                      .split(/[|;\n]+/)
                      .map((part) => normMatchString(part.includes(":") ? part.split(":").slice(1).join(":") : part))
                      .filter((s) => s.length >= 4),
                  ]
                : needles;
              b.rows.forEach((row) => {
                const tr = el("tr");
                let rowMatched = false;
                const rowCells = Array.isArray(row) ? row : [];
                rowCells.forEach((cell) => {
                  const cellText = String(cell ?? "");
                  const td = el("td", null, cellText);
                  if (textMatchesNeedles(cellText, rowNeedles)) rowMatched = true;
                  tr.appendChild(td);
                });
                if (!rowMatched && rowNeedles.length && rowCells.length) {
                  if (textMatchesNeedles(rowCells.map((c) => String(c ?? "")).join(" "), rowNeedles)) {
                    rowMatched = true;
                  }
                }
                if (rowMatched || clauseMatched) markAnchor(tr);
                tbody.appendChild(tr);
              });
              tbl.appendChild(tbody);
            }
            wrap.appendChild(tbl);
            sheet.appendChild(wrap);
          } else {
            const tag = bType === "heading" ? "h3" : "p";
            const cls = bType === "heading" ? "dv-block-heading" : "dv-block-para";
            const blk = el(tag, cls);
            const rawText = String(b.text || "");
            if (bClause && !rawText.trim().startsWith(bClause)) {
              blk.appendChild(el("span", "dv-clause-tag", `${bClause}.`));
            }
            if (rawText) blk.appendChild(document.createTextNode(rawText));
            if (clauseMatched || textMatchesNeedles(rawText, needles)) {
              markAnchor(blk);
            }
            if (blk.textContent) sheet.appendChild(blk);
          }
        });

        frag.appendChild(sheet);
      });
    }

    dvBody.appendChild(frag);
    syncViewerNavControls(targetPage);

    // Jump immediately on mount and reinforce on next animation frame for exact scrollTop
    scrollToViewerPage(targetPage, { instant: true, focusAnchor: !hasPageImages });
    requestAnimationFrame(() => {
      scrollToViewerPage(targetPage, { instant: true, focusAnchor: !hasPageImages });
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
    currentStreamHits = [];
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

  /* ---- Delegated Quote & Step Source Chip Click (Works Mid-Stream & Post-Stream) ---- */
  messages.addEventListener("click", (e) => {
    const stepChip = e.target.closest(".step-src-chip");
    if (stepChip && !stepChip.disabled && messages.contains(stepChip)) {
      openDocumentViewer(readStepChipTarget(stepChip));
      return;
    }

    const trigger = e.target.closest(".dq-src, .dq-open-doc");
    if (!trigger || trigger.disabled || !messages.contains(trigger)) return;
    const card = trigger.closest(".doc-quote");
    if (!card) return;
    if (card._viewerTarget) {
      openDocumentViewer(card._viewerTarget);
      return;
    }
    const srcBtn = card.querySelector(".dq-src");
    const labelText = (srcBtn && srcBtn.textContent) || "";
    const cardQuoteText = (card.querySelector(".dq-text") || {}).textContent || "";
    const liveSources = currentStreamHits.length ? buildSources(currentStreamHits) : [];
    const target = resolveQuoteViewerTarget(null, liveSources, labelText, cardQuoteText);
    openDocumentViewer(target);
  });

  /* ---- Anchor at Top of Query (No Downward Auto-Scroll While Streaming) ---- */
  function scrollToQueryTop(userEl) {
    requestAnimationFrame(() => {
      if (!userEl || queryCount <= 1) {
        chatScroll.scrollTop = 0;
        return;
      }
      const scrollRect = chatScroll.getBoundingClientRect();
      const elRect = userEl.getBoundingClientRect();
      const targetTop = Math.max(0, elRect.top - scrollRect.top + chatScroll.scrollTop - 16);
      chatScroll.scrollTo({ top: targetTop, behavior: "smooth" });
    });
  }

  function addUserMessage(text, idx) {
    const li = el("li", "msg msg-user");
    const meta = el("div", "query-meta", `Запрос ${String(idx).padStart(2, "0")}`);
    const heading = el("h2", "query-heading", text);
    li.append(meta, heading);
    messages.appendChild(li);
    return li;
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
      const sheetName = cleanSource(h.sheet_name || (h.metadata && h.metadata.sheet_name) || "");
      let rawClause = String(h.clause_no || "").trim();
      if (rawClause && /б\/н|\[_*\]/i.test(rawClause)) rawClause = "";
      let clause = rawClause;
      if (clause && !/^(п\.|пп\.|раздел|глава|ст\.|лист\s*:|табл)/i.test(clause)) {
        clause = "п. " + clause;
      }
      if (!clause && sheetName) {
        clause = `Лист: ${sheetName}`;
        rawClause = rawClause || clause;
      }
      const docNumber = cleanDocNo(h.doc_number);
      const key = `${h.doc_id || name}|${clause || sheetName}`;
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
        sheetName,
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
        sheetName: s.sheetName || "",
        pages: uniquePages,
      };
    });
  }

  function renderAnswer(
    body,
    {
      answer,
      hits,
      sources: prebuiltSources,
      quotes,
      versionAlerts,
      discrepancyAlerts,
      qualityAlerts,
      latencyMs,
      note,
      noteKind,
      mdBox: liveBox,
    }
  ) {
    const sources = Array.isArray(prebuiltSources) ? prebuiltSources : buildSources(hits);
    let mdBox = liveBox;
    const effectiveAnswer = answer || (note ? "" : "_Текст справки не получен._");
    const structOpts = {
      isStreaming: false,
      quotes,
      sources,
      hits,
      versionAlerts,
      discrepancyAlerts,
      qualityAlerts,
    };
    if (mdBox) {
      updateStructuredAnswerDom(mdBox, effectiveAnswer, structOpts);
      mdBox.classList.remove("streaming");
    } else {
      mdBox = el("div", "md ans-structured");
      updateStructuredAnswerDom(mdBox, effectiveAnswer, structOpts);
      body.appendChild(mdBox);
    }

    if (note) {
      const n = el("div", `gen-note ${noteKind === "warn" ? "warn" : ""}`);
      n.innerHTML = noteKind === "warn" ? ICONS.warn : ICONS.stop;
      n.appendChild(el("span", null, note));
      body.appendChild(n);
    }

    const actions = el("div", "msg-actions");
    let panel = null;

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
            sheetName: s.sheetName || "",
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
    applyQuoteMeta(mdBox, quotes, sources);
    if (mdBox && mdBox._parts && mdBox._parts.detailsBody) {
      const domQuotes = mdBox._parts.detailsBody.querySelectorAll(".doc-quote").length;
      mdBox._quoteCount = domQuotes || (Array.isArray(quotes) ? quotes.length : 0);
      syncDetailsDisclosure(mdBox);
    }
    return sources;
  }

  function renderError(li, body, err, question) {
    li.classList.add("msg-error");
    const cleanMsg = sanitizeUserErrorMessage(err, TECH_ERROR_STUB_RU);
    const errFrame = el("div", "ans-error-frame");
    errFrame.setAttribute("role", "alert");
    const iconWrap = el("span", "ans-error-icon");
    iconWrap.innerHTML = ICONS.alert;
    errFrame.appendChild(iconWrap);
    errFrame.appendChild(el("p", "ans-error-text", cleanMsg));
    body.appendChild(errFrame);

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
    toast(cleanMsg, "error");
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
    const userLi = addUserMessage(question, queryCount);

    const { li, body } = addBotShell();
    const typing = showTyping(body);
    scrollToQueryTop(userLi);
    const epoch = chatEpoch;

    const state = {
      text: "",
      hits: [],
      quotes: [],
      versionAlerts: [],
      discrepancyAlerts: [],
      qualityAlerts: [],
      latencyMs: 0,
      stopped: false,
      error: null,
    };
    currentStreamHits = [];
    let live = null;
    let raf = 0;

    const paint = () => {
      raf = 0;
      if (epoch !== chatEpoch) return;
      if (!live) {
        const probe = parseStructuredAnswer(state.text, { isStreaming: true });
        if (probe.waitingFirstHeading) return;
        typing.stop();
        live = el("div", "md ans-structured streaming");
        body.appendChild(live);
      }
      updateStructuredAnswerDom(live, state.text, {
        isStreaming: true,
        quotes: state.quotes,
        hits: state.hits,
        versionAlerts: state.versionAlerts,
        discrepancyAlerts: state.discrepancyAlerts,
        qualityAlerts: state.qualityAlerts,
      });
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
        const sr = (resp && resp.search_response) || {};
        state.text = (resp && resp.answer) || "";
        state.hits = (sr && sr.hits) || [];
        currentStreamHits = state.hits;
        state.latencyMs = (resp && resp.latency_ms) || 0;
        state.quotes = (resp && resp.quotes) || [];
        state.versionAlerts =
          (resp && Array.isArray(resp.version_alerts) && resp.version_alerts.length
            ? resp.version_alerts
            : sr.version_alerts) || [];
        state.discrepancyAlerts =
          (resp && Array.isArray(resp.discrepancy_alerts) && resp.discrepancy_alerts.length
            ? resp.discrepancy_alerts
            : sr.discrepancy_alerts) || [];
        state.qualityAlerts =
          (resp && Array.isArray(resp.quality_alerts) && resp.quality_alerts.length
            ? resp.quality_alerts
            : sr.quality_alerts) || [];
      } else if (!res.ok) {
        let detail = null;
        let errType = null;
        try {
          const d = await res.json();
          detail = typeof d.detail === "string" ? d.detail : typeof d.error === "string" ? d.error : null;
          errType = d && d.error_type;
        } catch {
          /* ignore */
        }
        if (res.status >= 500 || errType === "technical_error") {
          const techErr = new Error(TECH_ERROR_STUB_RU);
          techErr.status = res.status;
          techErr.isTechnical = true;
          throw techErr;
        }
        const httpErr = new Error(sanitizeUserErrorMessage(detail || "", TECH_ERROR_STUB_RU));
        httpErr.status = res.status;
        throw httpErr;
      } else {
        await readSSE(res, (event, data) => {
          if (epoch !== chatEpoch) return;
          if (event === "sources") {
            state.hits = (data && data.hits) || [];
            state.versionAlerts = (data && Array.isArray(data.version_alerts) && data.version_alerts) || [];
            state.discrepancyAlerts =
              (data && Array.isArray(data.discrepancy_alerts) && data.discrepancy_alerts) || [];
            state.qualityAlerts = (data && Array.isArray(data.quality_alerts) && data.quality_alerts) || [];
            currentStreamHits = state.hits;
            typing.set("Подготовка текста справки…");
          } else if (event === "status") {
            if (data && data.text) typing.set(data.text);
          } else if (event === "token") {
            state.text += (data && data.text) || "";
            schedule();
          } else if (event === "quotes") {
            state.quotes = (data && data.quotes) || [];
            if (live) schedule();
          } else if (event === "done") {
            state.latencyMs = (data && data.latency_ms) || 0;
          } else if (event === "error") {
            const rawErr =
              (data && (data.error || data.detail || data.message)) || TECH_ERROR_STUB_RU;
            const sseErr = new Error(sanitizeUserErrorMessage(rawErr, TECH_ERROR_STUB_RU));
            if (data && data.error_type === "technical_error") sseErr.isTechnical = true;
            state.error = sseErr;
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
        note = sanitizeUserErrorMessage(state.error, TECH_ERROR_STUB_RU);
        noteKind = "warn";
      }
      const builtSources = renderAnswer(body, {
        answer: state.text,
        hits: state.hits,
        quotes: state.quotes,
        versionAlerts: state.versionAlerts,
        discrepancyAlerts: state.discrepancyAlerts,
        qualityAlerts: state.qualityAlerts,
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
          versionAlerts: sanitizeAlertList(state.versionAlerts),
          discrepancyAlerts: sanitizeAlertList(state.discrepancyAlerts),
          qualityAlerts: sanitizeAlertList(state.qualityAlerts),
          latencySec:
            state.stopped || !state.latencyMs ? 0 : Number((state.latencyMs / 1000).toFixed(1)),
          stopped: Boolean(state.stopped),
        });
      } else {
        discardEmptyThread(threadId);
      }
    }

    if (controller === ctrl) controller = null;
    currentStreamHits = [];
    setBusy(false);
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
  const urlUploadForm = $("#urlUploadForm");
  const urlInput = $("#urlInput");
  const urlUploadBtn = $("#urlUploadBtn");
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
      toast(sanitizeUserErrorMessage(err), "error");
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

      // Column 1: Format cell (60px)
      const cellFmt = el("div", "doc-cell-fmt");
      cellFmt.appendChild(el("span", `doc-fmt ${known ? `${fmt} ext-${fmt}` : "other"}`, fmtLabel));
      li.appendChild(cellFmt);

      // Column 2: Title & Note cell (minmax(200px, 1fr)) — mobile meta lives inside .doc-main
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

      const numClean = cleanDocNo(d.doc_number);
      const dateClean = fmtDateNumeric(d.doc_date);
      const isExcel = fmt === "xlsx" || fmt === "xls";
      const pagesText = d.pages_total
        ? `${d.pages_total} ${isExcel ? plural(d.pages_total, "лист", "листа", "листов") : "стр."}`
        : "—";
      const status = String(d.status || "active");
      const statusText = STATUS_LABEL[status] || status;

      // Compact mobile metadata row (hidden on desktop; nested inside .doc-main so .doc has exact 6 grid cells)
      const mobMeta = el("div", "doc-mobile-meta");
      mobMeta.appendChild(el("span", `badge ${status}`, statusText));
      if (numClean) mobMeta.appendChild(el("span", null, `№ ${numClean}`));
      if (dateClean) mobMeta.appendChild(el("span", null, dateClean));
      if (d.pages_total) mobMeta.appendChild(el("span", null, pagesText));
      main.appendChild(mobMeta);

      li.appendChild(main);

      // Column 3: Tabular act number & date (152px)
      const req = el("div", "doc-req");
      req.appendChild(el("span", null, numClean ? `№ ${numClean}` : "Без номера"));
      if (dateClean) req.appendChild(el("span", "doc-req-sub", `от ${dateClean}`));
      li.appendChild(req);

      // Column 4: Tabular page count (76px)
      li.appendChild(el("div", "doc-pages", pagesText));

      // Column 5: Status badge cell (116px)
      const cellStatus = el("div", "doc-cell-status");
      cellStatus.appendChild(el("span", `badge ${status}`, statusText));
      li.appendChild(cellStatus);

      // Column 6: Preview & Delete actions (156px)
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
      toast(sanitizeUserErrorMessage(err), "error");
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

  function formatDuplicateMessage(payload) {
    if (!payload) return "Документ уже присутствует в базе знаний.";
    if (typeof payload === "string" && payload.trim()) return payload.trim();
    const direct = payload.detail || payload.message || payload.error_message;
    if (typeof direct === "string" && direct.trim()) return direct.trim();
    const existing = payload.existing_title || payload.existing_file_name || payload.duplicate_of;
    if (existing) {
      return `Дубликат документа: в базе уже загружен «${existing}».`;
    }
    return "Дубликат документа: файл уже присутствует в активном реестре.";
  }

  function formatUrlDisplayLabel(rawUrl) {
    try {
      const u = new URL(rawUrl);
      const segs = u.pathname.split("/").filter(Boolean);
      const last = segs.length ? decodeURIComponent(segs[segs.length - 1]) : "";
      if (last && ALLOWED.test(last)) return last;
      const compactPath =
        segs.length > 2
          ? `/${segs[0]}/…/${last}`
          : u.pathname === "/"
            ? ""
            : decodeURIComponent(u.pathname);
      const full = `${u.hostname}${compactPath}`;
      return full.length > 68 ? full.slice(0, 67) + "…" : full;
    } catch {
      const s = String(rawUrl || "").trim();
      return s.length > 68 ? s.slice(0, 67) + "…" : s;
    }
  }

  if (urlInput && urlUploadBtn) {
    urlInput.addEventListener("input", () => {
      urlUploadBtn.disabled = !urlInput.value.trim();
    });
  }

  if (urlUploadForm && urlInput) {
    urlUploadForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const raw = urlInput.value.trim();
      if (!raw) return;
      let validHttpUrl = false;
      if (/^https?:\/\/\S+/i.test(raw)) {
        try {
          const parsed = new URL(raw);
          validHttpUrl =
            (parsed.protocol === "http:" || parsed.protocol === "https:") &&
            Boolean(parsed.hostname);
        } catch {
          validHttpUrl = false;
        }
      }
      if (!validHttpUrl) {
        toast("Укажите корректную ссылку, начинающуюся с http:// или https://", "error");
        urlInput.focus();
        return;
      }
      urlInput.value = "";
      if (urlUploadBtn) urlUploadBtn.disabled = true;
      const label = formatUrlDisplayLabel(raw);
      queue.push({ url: raw, label, ui: createUploadRow(label) });
      pump();
    });
  }

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
    const nameEl = el("span", "upload-name", name);
    row.appendChild(nameEl);
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
      setName(newName) {
        if (newName != null && String(newName).trim()) {
          nameEl.textContent = String(newName).trim();
        }
      },
      set(p, text) {
        if (p != null) {
          fill.style.width = `${Math.max(0, Math.min(100, p))}%`;
          pct.textContent = `${Math.round(p)}%`;
        }
        if (text != null) msg.textContent = text;
      },
      warnDuplicate(text) {
        li.classList.remove("done", "failed");
        li.classList.add("duplicate", "warn");
        fill.style.width = "100%";
        pct.textContent = "Дубликат отклонён";
        msg.textContent = text;
      },
      finish(ok, text) {
        li.classList.remove("duplicate", "warn");
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
    try {
      while (queue.length) {
        const job = queue.shift();
        if (job.url) {
          await uploadUrlOne(job.url, job.label, job.ui);
        } else {
          await uploadOne(job.file, job.ui);
        }
        loadDocuments();
        checkHealth();
      }
    } finally {
      uploading = false;
    }
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
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(data || {});
          return;
        }
        if (xhr.status === 409 || (data && data.status === "duplicate")) {
          const dupMsg = formatDuplicateMessage(data);
          const err = new Error(dupMsg);
          err.status = 409;
          err.isDuplicate = true;
          err.data = data;
          reject(err);
          return;
        }
        if (xhr.status >= 500 || (data && data.error_type === "technical_error")) {
          const err = new Error(TECH_ERROR_STUB_RU);
          err.status = xhr.status;
          err.isTechnical = true;
          reject(err);
          return;
        }
        const detail = data && typeof data.detail === "string" ? data.detail : "";
        const err = new Error(sanitizeUserErrorMessage(detail, TECH_ERROR_STUB_RU));
        err.status = xhr.status;
        reject(err);
      };
      xhr.onerror = () => {
        const err = new Error(TECH_ERROR_STUB_RU);
        err.isTechnical = true;
        reject(err);
      };
      xhr.send(fd);
    });
  }

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  async function pollUploadTask(taskId, ui, onFileName) {
    ui.set(10, "Разбор структуры и индексация…");
    for (let i = 0; i < 3600; i++) {
      await sleep(i < 5 ? 600 : 1200);
      let t;
      try {
        t = await api(`/api/tasks/${encodeURIComponent(taskId)}`);
      } catch {
        continue;
      }
      if (onFileName) {
        const resolvedName =
          t.file_name ||
          (Array.isArray(t.files) && t.files[0]) ||
          (t.result &&
            Array.isArray(t.result.files) &&
            t.result.files[0] &&
            (t.result.files[0].file_name || t.result.files[0].name));
        if (resolvedName) onFileName(resolvedName);
      }
      if (t.status === "duplicate") {
        const stDup = (t.result && t.result.files && t.result.files[0]) || {};
        const warnText = formatDuplicateMessage(
          t.message || t.detail ? t : Object.keys(stDup).length ? stDup : t
        );
        ui.warnDuplicate(warnText);
        showToast(warnText, "warn", 5600);
        return;
      }
      const msg = String(t.message || t.stage_label || "Индексация…").replace(
        /^\[\d+\/\d+\]\s*/,
        ""
      );
      ui.set(10 + (Number(t.percent) || 0) * 0.9, msg);
      if (t.status === "completed") {
        const filesList = (t.result && Array.isArray(t.result.files) && t.result.files) || [];
        const st =
          filesList.find(
            (f) => f && (f.status === "failed" || f.status === "error" || f.error_message || f.error)
          ) ||
          filesList[0] ||
          {};
        if (onFileName && st.file_name) onFileName(st.file_name);
        if (st.status === "duplicate") {
          const warnText = formatDuplicateMessage(
            st.detail || st.message || st.existing_title || st.duplicate_of ? st : t
          );
          ui.warnDuplicate(warnText);
          showToast(warnText, "warn", 5600);
        } else if (st.status === "failed" || st.status === "error" || st.error_message || st.error) {
          const errMsg = sanitizeUserErrorMessage(
            st.error_message || st.error || t.error || "Не удалось проиндексировать документ"
          );
          ui.finish(false, errMsg);
          toast(errMsg, "error");
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
      if (t.status === "failed" || t.status === "error") {
        const stFail = (t.result && Array.isArray(t.result.files) && t.result.files[0]) || {};
        const errMsg = sanitizeUserErrorMessage(
          t.error ||
            stFail.error_message ||
            stFail.error ||
            t.message ||
            "Не удалось проиндексировать документ"
        );
        ui.finish(false, errMsg);
        toast(errMsg, "error");
        return;
      }
    }
    ui.finish(false, "Превышено время ожидания индексации");
  }

  async function uploadOne(file, ui) {
    try {
      ui.set(0, "Передача файла на сервер…");
      const res = await sendFile(file, (f) => ui.set(f * 10, "Передача файла на сервер…"));
      if (res && res.status === "duplicate") {
        const warnText = formatDuplicateMessage(res);
        ui.warnDuplicate(warnText);
        showToast(warnText, "warn", 5600);
        return;
      }
      if (!res.task_id) {
        ui.finish(true, "Файл принят");
        return;
      }
      await pollUploadTask(res.task_id, ui);
    } catch (err) {
      if (err && (err.status === 409 || err.isDuplicate)) {
        const warnText = formatDuplicateMessage((err && err.data) || (err && err.message));
        ui.warnDuplicate(warnText);
        showToast(warnText, "warn", 5600);
        return;
      }
      const cleanErr = sanitizeUserErrorMessage(err);
      ui.finish(false, cleanErr);
      toast(`${file.name}: ${cleanErr}`, "error");
    }
  }

  async function uploadUrlOne(url, displayLabel, ui) {
    let currentName = displayLabel;
    const syncName = (candidate) => {
      if (candidate && typeof candidate === "string" && candidate.trim()) {
        currentName = candidate.trim();
        ui.setName(currentName);
      }
    };
    try {
      ui.set(5, "Скачивание документа по ссылке…");
      const res = await api("/api/documents/upload-url", {
        method: "POST",
        body: JSON.stringify({ url }),
      });
      if (res && Array.isArray(res.files) && res.files[0]) {
        syncName(res.files[0]);
      } else if (res && res.file_name) {
        syncName(res.file_name);
      }
      if (res && res.status === "duplicate") {
        const warnText = formatDuplicateMessage(res);
        ui.warnDuplicate(warnText);
        showToast(warnText, "warn", 5600);
        return;
      }
      if (!res || !res.task_id) {
        ui.finish(true, (res && res.message) || "Документ принят");
        return;
      }
      await pollUploadTask(res.task_id, ui, syncName);
    } catch (err) {
      if (err && (err.status === 409 || err.isDuplicate)) {
        const warnText = formatDuplicateMessage((err && err.data) || (err && err.message));
        ui.warnDuplicate(warnText);
        showToast(warnText, "warn", 5600);
        return;
      }
      const cleanErr = sanitizeUserErrorMessage(err);
      ui.finish(false, cleanErr);
      toast(`${currentName}: ${cleanErr}`, "error");
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

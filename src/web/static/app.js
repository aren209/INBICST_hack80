/**
 * MIPT Normative RAG V3 — Frontend SPA Controller (Vanilla JS, zero build step)
 * Connects to FastAPI REST & SSE endpoints:
 *   - /api/health, /api/documents, /api/documents/upload, /api/documents/sync-folder
 *   - /api/search, /api/ask (Autonomous Table-Aware RAG Agent with Tool Calling)
 *   - /api/tables, /api/tables/{table_id}, /api/tables/query (Interactive SQLite Table Explorer)
 */

(function () {
  'use strict';

  // ---------------------------------------------------------------------------
  // Showcase Presets for 3-Tier Adaptive Table Architecture & Tool Calling
  // ---------------------------------------------------------------------------
  const TABLE_SHOWCASE_PRESETS = [
    {
      id: 'showcase_xl_cfo_frkt',
      badge: '📊 XL Tool Query',
      badgeCls: 'bg-emerald-100 text-emerald-900 border-emerald-300',
      btnCls: 'bg-white hover:bg-emerald-50 text-slate-800 border-emerald-300 shadow-sm',
      title: 'ЦФО 2026: Подразделения ФРКТ',
      query: 'Сколько подразделений подчиняется финансодержателю ФРКТ в финструктуре 2026?',
      hint: 'Автономный вызов query_structured_table (filter_rows / aggregate) по таблице Фин. структуры 2026 в SQLite',
      mode: 'ask',
    },
    {
      id: 'showcase_xl_smeta_formulas',
      badge: '🧮 XL Формулы',
      badgeCls: 'bg-indigo-100 text-indigo-900 border-indigo-300',
      btnCls: 'bg-white hover:bg-indigo-50 text-slate-800 border-indigo-300 shadow-sm',
      title: 'Смета: Формулы накладных и НДС',
      query: 'Покажи все строки и формулы расчета накладных расходов и НДС в смете',
      hint: 'Извлечение реальных формул Excel [формула: =...] и ставок 20% через Tool Calling + Приказ 794-1',
      mode: 'ask',
    },
    {
      id: 'showcase_pdf_inline_kz',
      badge: '📄 PDF Inline vs XL',
      badgeCls: 'bg-sky-100 text-sky-900 border-sky-300',
      btnCls: 'bg-white hover:bg-sky-50 text-slate-800 border-sky-300 shadow-sm',
      title: 'Посвящение в КЗ: Дни репетиций (Inline Parent)',
      query: 'Сколько дней репетиций дается физтех-школам на Посвящение в КЗ?',
      hint: 'Контраст архитектуры: компактная таблица из скана PDF (< 25 строк) отдается целиком через ParentSectionRecord без лишнего SQL-вызова',
      mode: 'ask',
    },
  ];

  // ---------------------------------------------------------------------------
  // Application State
  // ---------------------------------------------------------------------------
  const state = {
    activeTab: 'upload',
    playgroundMode: 'ask', // 'search' | 'ask'
    selectedFiles: [],
    activeEventSource: null,
    activePollTimer: null,
    documents: [],
    reportSummary: null,
    presets: [],
    linkedTablesById: new Map(),
    inspector: {
      doc: null,
      sections: [],
      chunks: [],
      viewMode: 'raw_text', // 'raw_text' | 'embed_text' | 'clean_text'
      selectedUnitId: null,
      selectedSectionId: null,
    },
    tableExplorer: {
      tables: [],
      docIdFilter: null,
      activeTableId: null,
      activePassport: null,
      formulaSummary: null,
      operation: 'filter_rows', // 'filter_rows' | 'aggregate' | 'get_formulas'
      offset: 0,
      limit: 25,
      lastResponse: null,
    },
  };

  // ---------------------------------------------------------------------------
  // Safe Formatting, Markdown & Badge Helpers
  // ---------------------------------------------------------------------------
  function escapeHtml(value) {
    if (value === null || value === undefined) return '';
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function formatBytes(bytes) {
    if (!bytes || bytes <= 0) return '0 Б';
    const units = ['Б', 'КБ', 'МБ', 'ГБ'];
    const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
    return (bytes / Math.pow(1024, i)).toFixed(i === 0 ? 0 : 1) + ' ' + units[i];
  }

  function formatStatusBadge(status) {
    const map = {
      active: {
        label: 'Действует (active)',
        cls: 'bg-emerald-50 text-emerald-700 border border-emerald-200',
      },
      superseded_template_reference: {
        label: 'Справочник формул (отменен)',
        cls: 'bg-amber-50 text-amber-800 border border-amber-200',
      },
      superseded: {
        label: 'Отменен (superseded)',
        cls: 'bg-slate-100 text-slate-600 border border-slate-300',
      },
      draft: {
        label: 'Проект (draft)',
        cls: 'bg-sky-50 text-sky-700 border border-sky-200',
      },
      error: {
        label: 'Ошибка (error)',
        cls: 'bg-red-50 text-red-700 border border-red-200',
      },
    };
    const info = map[status] || {
      label: status || 'active',
      cls: 'bg-slate-100 text-slate-700 border border-slate-200',
    };
    return `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-semibold ${info.cls}">${escapeHtml(info.label)}</span>`;
  }

  function formatZoneBadge(zone) {
    const map = {
      normative: 'bg-blue-50 text-blue-700 border-blue-200',
      table: 'bg-indigo-50 text-indigo-700 border-indigo-200',
      form: 'bg-amber-50 text-amber-800 border-amber-200',
      approval: 'bg-slate-100 text-slate-500 border-slate-300',
      meta: 'bg-purple-50 text-purple-700 border-purple-200',
    };
    const cls = map[zone] || 'bg-slate-100 text-slate-700 border-slate-200';
    return `<span class="inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-mono-code font-semibold border ${cls}">zone: ${escapeHtml(zone || 'normative')}</span>`;
  }

  function formatFileBadge(fmt) {
    const ext = (fmt || '').toLowerCase().replace(/^\./, '');
    const map = {
      pdf: 'bg-rose-50 text-rose-700 border-rose-200',
      docx: 'bg-blue-50 text-blue-700 border-blue-200',
      xlsx: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    };
    const cls = map[ext] || 'bg-slate-100 text-slate-700 border-slate-200';
    return `<span class="inline-block px-2 py-0.5 rounded text-[11px] font-mono-code font-bold uppercase border ${cls}">${escapeHtml(ext || 'file')}</span>`;
  }

  /**
   * Formats cell text highlighting `[формула: =...]` tags cleanly.
   */
  function formatCellWithFormulaHighlight(rawVal) {
    if (rawVal === null || rawVal === undefined || rawVal === '') {
      return '<span class="text-slate-300">—</span>';
    }
    const str = String(rawVal);
    const formulaRegex = /\[формула:\s*([^\]]+)\]/gi;
    if (!formulaRegex.test(str)) {
      return escapeHtml(str);
    }
    formulaRegex.lastIndex = 0;
    let result = '';
    let lastIdx = 0;
    let match;
    while ((match = formulaRegex.exec(str)) !== null) {
      const before = str.slice(lastIdx, match.index);
      if (before) result += escapeHtml(before);
      result += `<span class="formula-cell-badge">ƒx ${escapeHtml(match[1].trim())}</span>`;
      lastIdx = match.index + match[0].length;
    }
    if (lastIdx < str.length) {
      result += escapeHtml(str.slice(lastIdx));
    }
    return result;
  }

  /**
   * Inline Markdown formatter (input is raw string -> escaped safely first).
   */
  function formatInlineMarkdown(text) {
    let safe = escapeHtml(text || '');
    // Highlight [формула: =...]
    safe = safe.replace(
      /\[формула:\s*([^\]]+)\]/gi,
      '<span class="formula-cell-badge">ƒx $1</span>'
    );
    // Inline code `...`
    safe = safe.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Bold **...**
    safe = safe.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    return safe;
  }

  /**
   * Safe lightweight Markdown renderer supporting headings, lists, code blocks,
   * blockquotes, bold/inline-code, and Markdown pipe tables.
   */
  function renderMarkdownSafe(markdownText) {
    if (!markdownText) return '';
    const lines = String(markdownText).replace(/\r\n/g, '\n').split('\n');
    const htmlParts = [];
    let i = 0;

    while (i < lines.length) {
      const line = lines[i];
      const trimmed = line.trim();

      // Empty line
      if (!trimmed) {
        i++;
        continue;
      }

      // Fenced code block ```
      if (trimmed.startsWith('```')) {
        const codeLines = [];
        i++;
        while (i < lines.length && !lines[i].trim().startsWith('```')) {
          codeLines.push(lines[i]);
          i++;
        }
        if (i < lines.length) i++; // skip closing ```
        htmlParts.push(`<pre><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`);
        continue;
      }

      // Pipe table detection: current line has | and next line is separator |---|---| or pipe row
      if (
        trimmed.includes('|') &&
        i + 1 < lines.length &&
        /^\s*\|?[\s:-]+\|[\s|:-]*$/.test(lines[i + 1])
      ) {
        const parsePipeRow = (r) => {
          let s = r.trim();
          if (s.startsWith('|')) s = s.slice(1);
          if (s.endsWith('|')) s = s.slice(0, -1);
          return s.split('|').map((c) => c.trim());
        };
        const headers = parsePipeRow(trimmed);
        i += 2; // skip header + separator
        const bodyRows = [];
        while (i < lines.length && lines[i].trim().includes('|') && lines[i].trim() !== '') {
          bodyRows.push(parsePipeRow(lines[i]));
          i++;
        }
        const thHtml = headers.map((h) => `<th>${formatInlineMarkdown(h)}</th>`).join('');
        const trsHtml = bodyRows
          .map(
            (r) =>
              `<tr>${headers
                .map((_, cIdx) => `<td>${formatInlineMarkdown(r[cIdx] ?? '')}</td>`)
                .join('')}</tr>`
          )
          .join('');
        htmlParts.push(
          `<div class="ai-md-table-wrap"><table class="ai-md-table"><thead><tr>${thHtml}</tr></thead><tbody>${trsHtml}</tbody></table></div>`
        );
        continue;
      }

      // Headings
      const hMatch = trimmed.match(/^(#{1,4})\s+(.+)$/);
      if (hMatch) {
        const lvl = Math.min(4, hMatch[1].length);
        htmlParts.push(`<h${lvl}>${formatInlineMarkdown(hMatch[2])}</h${lvl}>`);
        i++;
        continue;
      }

      // Blockquote
      if (trimmed.startsWith('>')) {
        const qLines = [];
        while (i < lines.length && lines[i].trim().startsWith('>')) {
          qLines.push(lines[i].trim().replace(/^>\s?/, ''));
          i++;
        }
        htmlParts.push(`<blockquote>${qLines.map(formatInlineMarkdown).join('<br/>')}</blockquote>`);
        continue;
      }

      // Unordered list
      if (/^[-*•]\s+/.test(trimmed)) {
        const items = [];
        while (i < lines.length && /^[-*•]\s+/.test(lines[i].trim())) {
          items.push(lines[i].trim().replace(/^[-*•]\s+/, ''));
          i++;
        }
        htmlParts.push(`<ul>${items.map((it) => `<li>${formatInlineMarkdown(it)}</li>`).join('')}</ul>`);
        continue;
      }

      // Ordered list
      if (/^\d+[.)]\s+/.test(trimmed)) {
        const items = [];
        while (i < lines.length && /^\d+[.)]\s+/.test(lines[i].trim())) {
          items.push(lines[i].trim().replace(/^\d+[.)]\s+/, ''));
          i++;
        }
        htmlParts.push(`<ol>${items.map((it) => `<li>${formatInlineMarkdown(it)}</li>`).join('')}</ol>`);
        continue;
      }

      // Regular paragraph (group consecutive plain lines)
      const pLines = [];
      while (
        i < lines.length &&
        lines[i].trim() !== '' &&
        !lines[i].trim().startsWith('```') &&
        !/^(#{1,4})\s+/.test(lines[i].trim()) &&
        !lines[i].trim().startsWith('>') &&
        !/^[-*•]\s+/.test(lines[i].trim()) &&
        !/^\d+[.)]\s+/.test(lines[i].trim()) &&
        !(
          lines[i].trim().includes('|') &&
          i + 1 < lines.length &&
          /^\s*\|?[\s:-]+\|[\s|:-]*$/.test(lines[i + 1])
        )
      ) {
        pLines.push(lines[i].trim());
        i++;
      }
      htmlParts.push(`<p>${pLines.map(formatInlineMarkdown).join('<br/>')}</p>`);
    }

    return htmlParts.join('');
  }

  /**
   * Renders raw_text visually highlighting top `...overlap` and bottom `overlap...`
   * in subtle grey/italic (.overlap-segment), while keeping core clean_text crisp.
   */
  function renderTextWithOverlapHighlight(item) {
    const raw = item.raw_text || '';
    const clean = item.clean_text || '';
    const explicitPrefix = item.overlap_prefix || '';
    const explicitSuffix = item.overlap_suffix || '';

    if ((explicitPrefix || explicitSuffix) && clean) {
      let html = '';
      if (explicitPrefix) {
        html += `<div class="mb-1.5"><span class="overlap-segment"><span class="overlap-marker-tag">↑ overlap 12%</span>${escapeHtml(explicitPrefix)}</span></div>`;
      }
      html += `<div>${escapeHtml(clean)}</div>`;
      if (explicitSuffix) {
        html += `<div class="mt-1.5"><span class="overlap-segment"><span class="overlap-marker-tag">↓ overlap 12%</span>${escapeHtml(explicitSuffix)}</span></div>`;
      }
      return html;
    }

    if (raw && clean && raw !== clean) {
      const idx = raw.indexOf(clean);
      if (idx !== -1) {
        const before = raw.slice(0, idx).trim();
        const after = raw.slice(idx + clean.length).trim();
        let html = '';
        if (before) {
          html += `<div class="mb-1.5"><span class="overlap-segment"><span class="overlap-marker-tag">↑ overlap 12%</span>${escapeHtml(before)}</span></div>`;
        }
        html += `<div>${escapeHtml(clean)}</div>`;
        if (after) {
          html += `<div class="mt-1.5"><span class="overlap-segment"><span class="overlap-marker-tag">↓ overlap 12%</span>${escapeHtml(after)}</span></div>`;
        }
        return html;
      }
    }

    return escapeHtml(raw || clean || '');
  }

  function renderRelationBadges(doc) {
    const badges = [];
    (doc.amends || []).forEach((num) => {
      badges.push(`<span class="inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-medium bg-purple-50 text-purple-700 border border-purple-200">Изменяет № ${escapeHtml(num)}</span>`);
    });
    (doc.amended_by || []).forEach((num) => {
      badges.push(`<span class="inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-medium bg-fuchsia-50 text-fuchsia-800 border border-fuchsia-200">Изменен № ${escapeHtml(num)}</span>`);
    });
    (doc.supersedes || []).forEach((num) => {
      badges.push(`<span class="inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-medium bg-blue-50 text-blue-700 border border-blue-200">Отменяет № ${escapeHtml(num)}</span>`);
    });
    (doc.superseded_by || []).forEach((num) => {
      badges.push(`<span class="inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-medium bg-rose-50 text-rose-700 border border-rose-200">Отменен № ${escapeHtml(num)}</span>`);
    });
    (doc.also_in_files || []).forEach((fname) => {
      badges.push(`<span class="inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-medium bg-indigo-50 text-indigo-700 border border-indigo-200" title="Кросс-файловая дедупликация по content_hash / table_hash">Дублируется в: ${escapeHtml(fname)}</span>`);
    });
    if (doc.applicable_years && doc.applicable_years.length > 0) {
      badges.push(`<span class="inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-mono-code bg-slate-100 text-slate-700 border border-slate-200">Годы: ${escapeHtml(doc.applicable_years.join(', '))}</span>`);
    }
    return badges.length ? `<div class="flex flex-wrap gap-1">${badges.join('')}</div>` : '<span class="text-slate-400">—</span>';
  }

  // ---------------------------------------------------------------------------
  // Toast Notifications
  // ---------------------------------------------------------------------------
  function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const styles = {
      success: 'bg-emerald-600 text-white border-emerald-700',
      error: 'bg-red-600 text-white border-red-700',
      warning: 'bg-amber-500 text-white border-amber-600',
      info: 'bg-slate-900 text-white border-slate-800',
    };

    const el = document.createElement('div');
    el.className = `toast-item pointer-events-auto px-4 py-3 rounded-xl shadow-lg border text-xs font-medium flex items-start justify-between gap-3 ${styles[type] || styles.info}`;
    el.innerHTML = `
      <span class="leading-relaxed">${escapeHtml(message)}</span>
      <button type="button" class="opacity-80 hover:opacity-100 font-bold text-sm leading-none">&times;</button>
    `;
    el.querySelector('button').addEventListener('click', () => el.remove());
    container.appendChild(el);

    setTimeout(() => {
      if (el.parentNode) el.remove();
    }, 5000);
  }

  // ---------------------------------------------------------------------------
  // System Health Polling (/api/health)
  // ---------------------------------------------------------------------------
  async function refreshHealth() {
    try {
      const res = await fetch('/api/health');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      const qDot = document.getElementById('health-qdrant-dot');
      const qText = document.getElementById('health-qdrant-text');
      const qConnected = data.qdrant && data.qdrant.connected;
      const pts = data.qdrant && typeof data.qdrant.points_count === 'number' ? data.qdrant.points_count : 0;
      qDot.className = `w-2 h-2 rounded-full ${qConnected ? 'bg-emerald-500' : 'bg-red-500'}`;
      qText.textContent = qConnected ? `OK (${pts} точек)` : 'недоступен';

      const pDot = document.getElementById('health-polza-dot');
      const pText = document.getElementById('health-polza-text');
      const pConf = data.polza_api && data.polza_api.configured;
      const model = (data.polza_api && data.polza_api.embedding_model) || 'bge-m3';
      pDot.className = `w-2 h-2 rounded-full ${pConf ? 'bg-emerald-500' : 'bg-amber-500'}`;
      pText.textContent = pConf ? model : 'не настроен ключ';

      document.getElementById('health-docs-count').textContent = data.registry_docs_count ?? 0;
      document.getElementById('health-sections-count').textContent = data.sections_count ?? 0;

      const tblEl = document.getElementById('health-tables-count');
      if (tblEl) {
        tblEl.textContent = String(data.structured_tables_count ?? data.tables_count ?? 0);
      }
    } catch (err) {
      document.getElementById('health-qdrant-dot').className = 'w-2 h-2 rounded-full bg-red-500';
      document.getElementById('health-qdrant-text').textContent = 'ошибка связи';
    }
  }

  // ---------------------------------------------------------------------------
  // Tab & Playground Mode Navigation
  // ---------------------------------------------------------------------------
  function switchTab(tabName) {
    state.activeTab = tabName;
    const tabs = ['upload', 'registry', 'search'];
    tabs.forEach((t) => {
      const panel = document.getElementById(`panel-${t}`);
      const btn = document.getElementById(`tab-btn-${t}`);
      if (!panel || !btn) return;

      if (t === tabName) {
        panel.classList.remove('hidden');
        btn.className = 'tab-nav-btn inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold bg-blue-600 text-white shadow-sm transition';
      } else {
        panel.classList.add('hidden');
        btn.className = 'tab-nav-btn inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium text-slate-600 hover:bg-slate-100 hover:text-slate-900 transition';
      }
    });

    if (tabName === 'registry') {
      loadDocumentsRegistry();
    }
  }

  function setPlaygroundMode(mode) {
    state.playgroundMode = mode === 'search' ? 'search' : 'ask';
    const btnSearch = document.getElementById('btn-mode-search');
    const btnAsk = document.getElementById('btn-mode-ask');
    if (!btnSearch || !btnAsk) return;

    if (state.playgroundMode === 'ask') {
      btnAsk.className =
        'px-3.5 py-2 rounded-lg bg-gradient-to-r from-indigo-600 to-emerald-600 text-white shadow-sm transition cursor-pointer flex items-center gap-1.5';
      btnSearch.className =
        'px-3 py-2 rounded-lg text-slate-600 hover:text-slate-900 hover:bg-white/60 transition cursor-pointer';
    } else {
      btnSearch.className =
        'px-3 py-2 rounded-lg bg-white text-slate-900 shadow-sm transition cursor-pointer';
      btnAsk.className =
        'px-3.5 py-2 rounded-lg text-indigo-700 hover:bg-indigo-50 transition cursor-pointer flex items-center gap-1.5';
    }
  }

  // ---------------------------------------------------------------------------
  // Tab 1: Drag & Drop Upload + Sync Folder + Live 6-stage Progress
  // ---------------------------------------------------------------------------
  function addFilesToSelection(fileList) {
    const allowed = ['.pdf', '.docx', '.xlsx'];
    let added = 0;
    Array.from(fileList || []).forEach((file) => {
      const lower = file.name.toLowerCase();
      const valid = allowed.some((ext) => lower.endsWith(ext));
      if (!valid) {
        showToast(`Формат файла "${file.name}" не поддерживается (разрешены .pdf, .docx, .xlsx)`, 'warning');
        return;
      }
      const exists = state.selectedFiles.some((f) => f.name === file.name && f.size === file.size);
      if (!exists) {
        state.selectedFiles.push(file);
        added++;
      }
    });
    if (added > 0) renderSelectedFiles();
  }

  function renderSelectedFiles() {
    const container = document.getElementById('upload-files-container');
    const listEl = document.getElementById('upload-files-list');
    const countEl = document.getElementById('upload-files-count');

    if (state.selectedFiles.length === 0) {
      container.classList.add('hidden');
      listEl.innerHTML = '';
      countEl.textContent = '0';
      return;
    }

    container.classList.remove('hidden');
    countEl.textContent = String(state.selectedFiles.length);
    listEl.innerHTML = state.selectedFiles
      .map((file, idx) => {
        const ext = file.name.includes('.') ? file.name.split('.').pop() : '';
        return `
          <li class="px-3 py-2 flex items-center justify-between gap-2">
            <div class="flex items-center gap-2 min-w-0">
              ${formatFileBadge(ext)}
              <span class="font-medium text-slate-800 truncate" title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</span>
              <span class="text-slate-400 shrink-0">(${formatBytes(file.size)})</span>
            </div>
            <button type="button" data-remove-idx="${idx}" class="text-slate-400 hover:text-red-600 font-bold px-1.5">&times;</button>
          </li>
        `;
      })
      .join('');

    listEl.querySelectorAll('[data-remove-idx]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const idx = parseInt(btn.getAttribute('data-remove-idx'), 10);
        state.selectedFiles.splice(idx, 1);
        renderSelectedFiles();
      });
    });
  }

  function appendEventLog(message, stage = 'info', timestamp = null) {
    const logEl = document.getElementById('prog-event-log');
    if (!logEl) return;
    const ts = timestamp
      ? new Date(timestamp).toLocaleTimeString('ru-RU', { hour12: false })
      : new Date().toLocaleTimeString('ru-RU', { hour12: false });

    let colorCls = 'text-slate-200';
    if (stage === 'error') colorCls = 'text-red-400 font-semibold';
    else if (stage === 'done') colorCls = 'text-emerald-400 font-semibold';
    else if (stage === 'ocr') colorCls = 'text-amber-300';
    else if (stage === 'embedding' || stage === 'qdrant') colorCls = 'text-sky-300';

    const row = document.createElement('div');
    row.className = `${colorCls} leading-relaxed`;
    row.textContent = `[${ts}] ${message}`;
    logEl.appendChild(row);
    logEl.scrollTop = logEl.scrollHeight;
  }

  function updateStageStepper(stageIndex, status) {
    const cards = document.querySelectorAll('#pipeline-stages-grid .stage-step-card');
    cards.forEach((card) => {
      const idx = parseInt(card.getAttribute('data-stage-idx'), 10);
      const dot = card.querySelector('.stage-dot');
      card.classList.remove('border-blue-500', 'bg-blue-50', 'border-emerald-300', 'bg-emerald-50/70', 'border-red-300', 'bg-red-50');
      dot.classList.remove('stage-active-circle', 'bg-blue-600', 'text-white', 'bg-emerald-600', 'bg-red-600', 'bg-slate-200', 'text-slate-700');

      if (status === 'completed' || status === 'done' || (stageIndex > idx && status !== 'failed')) {
        card.classList.add('border-emerald-300', 'bg-emerald-50/70');
        dot.classList.add('bg-emerald-600', 'text-white');
        dot.textContent = '✓';
      } else if (idx === stageIndex && (status === 'failed' || status === 'error')) {
        card.classList.add('border-red-300', 'bg-red-50');
        dot.classList.add('bg-red-600', 'text-white');
        dot.textContent = '!';
      } else if (idx === stageIndex) {
        card.classList.add('border-blue-500', 'bg-blue-50');
        dot.classList.add('bg-blue-600', 'text-white', 'stage-active-circle');
        dot.textContent = String(idx);
      } else {
        dot.classList.add('bg-slate-200', 'text-slate-700');
        dot.textContent = String(idx);
      }
    });
  }

  function applyProgressEvent(ev) {
    const pct = Math.max(0, Math.min(100, Number(ev.percent ?? 0)));
    const stageIdx = Number(ev.stage_index ?? 0);
    const stage = ev.stage || 'running';

    const barFill = document.getElementById('prog-bar-fill');
    barFill.style.width = `${pct}%`;
    if (stage === 'done' || ev.status === 'completed') {
      barFill.className = 'h-full bg-emerald-600 rounded-full transition-all duration-300';
    } else if (stage === 'error' || ev.status === 'failed') {
      barFill.className = 'h-full bg-red-600 rounded-full transition-all duration-300';
    } else {
      barFill.className = 'h-full bg-blue-600 progress-bar-animated rounded-full transition-all duration-300';
    }

    document.getElementById('prog-percent-label').textContent = `${pct}%`;
    if (ev.file_name) {
      document.getElementById('prog-current-file').textContent = `Файл: ${ev.file_name}`;
    }
    if (typeof ev.parent_count === 'number' && ev.parent_count >= 0) {
      document.getElementById('prog-parent-count').textContent = String(ev.parent_count);
    }
    if (typeof ev.chunk_count === 'number' && ev.chunk_count >= 0) {
      document.getElementById('prog-chunk-count').textContent = String(ev.chunk_count);
    }
    if (typeof ev.ocr_pages === 'number' && ev.ocr_pages >= 0) {
      document.getElementById('prog-ocr-count').textContent = String(ev.ocr_pages);
    }

    const subtitle = document.getElementById('prog-status-subtitle');
    const badge = document.getElementById('prog-task-badge');

    if (stage === 'done' || ev.status === 'completed') {
      updateStageStepper(6, 'completed');
      subtitle.textContent = ev.message || 'Индексация успешно завершена';
      badge.className = 'px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800 border border-emerald-200';
      badge.textContent = 'Готово';
    } else if (stage === 'error' || ev.status === 'failed') {
      updateStageStepper(stageIdx || 1, 'failed');
      subtitle.textContent = ev.message || ev.error || 'Ошибка выполнения задачи';
      badge.className = 'px-2.5 py-1 rounded-full text-xs font-semibold bg-red-100 text-red-800 border border-red-200';
      badge.textContent = 'Ошибка';
    } else {
      updateStageStepper(stageIdx || 1, 'running');
      subtitle.textContent = `${ev.stage_label || 'Обработка'}: ${ev.message || ''}`;
      badge.className = 'px-2.5 py-1 rounded-full text-xs font-semibold bg-blue-100 text-blue-800 border border-blue-200';
      badge.textContent = ev.stage_label || 'Выполняется...';
    }
  }

  function stopActiveTaskTracking() {
    if (state.activeEventSource) {
      state.activeEventSource.close();
      state.activeEventSource = null;
    }
    if (state.activePollTimer) {
      clearInterval(state.activePollTimer);
      state.activePollTimer = null;
    }
  }

  function trackTaskProgress(taskId) {
    stopActiveTaskTracking();
    let seenEventCount = 0;

    document.getElementById('prog-parent-count').textContent = '0';
    document.getElementById('prog-chunk-count').textContent = '0';
    document.getElementById('prog-ocr-count').textContent = '0';
    updateStageStepper(1, 'running');

    const onTaskFinished = (isError, msg) => {
      stopActiveTaskTracking();
      refreshHealth();
      loadDocumentsRegistry();
      if (isError) {
        showToast(msg || 'Индексация завершилась с ошибкой', 'error');
      } else {
        showToast(msg || 'Индексация успешно завершена!', 'success');
      }
    };

    try {
      const es = new EventSource(`/api/tasks/${encodeURIComponent(taskId)}/stream`);
      state.activeEventSource = es;

      es.onmessage = (e) => {
        if (!e.data) return;
        try {
          const payload = JSON.parse(e.data);
          if (Array.isArray(payload.events)) {
            payload.events.forEach((ev) => {
              appendEventLog(ev.message || ev.stage_label || '', ev.stage, ev.timestamp);
            });
            seenEventCount = payload.events.length;
            applyProgressEvent(payload);
            if (payload.status === 'completed' || payload.status === 'failed') {
              if (payload.result && typeof payload.result.total_chunks === 'number') {
                document.getElementById('prog-parent-count').textContent = String(payload.result.total_parent_sections || 0);
                document.getElementById('prog-chunk-count').textContent = String(payload.result.total_chunks || 0);
                document.getElementById('prog-ocr-count').textContent = String(payload.result.total_ocr_pages || 0);
              }
              onTaskFinished(payload.status === 'failed', payload.message);
            }
          } else {
            seenEventCount++;
            if (payload.message) {
              appendEventLog(payload.message, payload.stage, payload.timestamp);
            }
            applyProgressEvent(payload);
            if (payload.stage === 'done' || payload.stage === 'error') {
              fetch(`/api/tasks/${encodeURIComponent(taskId)}`)
                .then((r) => r.json())
                .then((snap) => {
                  if (snap.result && typeof snap.result.total_chunks === 'number') {
                    document.getElementById('prog-parent-count').textContent = String(snap.result.total_parent_sections || 0);
                    document.getElementById('prog-chunk-count').textContent = String(snap.result.total_chunks || 0);
                    document.getElementById('prog-ocr-count').textContent = String(snap.result.total_ocr_pages || 0);
                  }
                })
                .catch(() => {});
              onTaskFinished(payload.stage === 'error', payload.message);
            }
          }
        } catch (parseErr) {
          // ignore malformed SSE frame
        }
      };

      es.onerror = () => {
        if (state.activeEventSource) {
          state.activeEventSource.close();
          state.activeEventSource = null;
        }
        startPollingFallback(taskId, seenEventCount, onTaskFinished);
      };
    } catch (err) {
      startPollingFallback(taskId, seenEventCount, onTaskFinished);
    }
  }

  function startPollingFallback(taskId, initialSeen, onTaskFinished) {
    let seen = initialSeen;
    state.activePollTimer = setInterval(async () => {
      try {
        const res = await fetch(`/api/tasks/${encodeURIComponent(taskId)}`);
        if (!res.ok) return;
        const snap = await res.json();
        const evs = snap.events || [];
        for (let i = seen; i < evs.length; i++) {
          appendEventLog(evs[i].message || '', evs[i].stage, evs[i].timestamp);
        }
        seen = evs.length;
        applyProgressEvent(snap);

        if (snap.status === 'completed' || snap.status === 'failed') {
          if (snap.result && typeof snap.result.total_chunks === 'number') {
            document.getElementById('prog-parent-count').textContent = String(snap.result.total_parent_sections || 0);
            document.getElementById('prog-chunk-count').textContent = String(snap.result.total_chunks || 0);
            document.getElementById('prog-ocr-count').textContent = String(snap.result.total_ocr_pages || 0);
          }
          onTaskFinished(snap.status === 'failed', snap.message);
        }
      } catch (e) {
        // retry next tick
      }
    }, 1000);
  }

  async function handleUploadSubmit() {
    if (state.selectedFiles.length === 0) {
      showToast('Сначала выберите или перетащите хотя бы один файл (.pdf, .docx, .xlsx)', 'warning');
      return;
    }

    const fd = new FormData();
    state.selectedFiles.forEach((f) => fd.append('files', f));
    fd.append('status', document.getElementById('upload-status').value);
    fd.append('ocr_backend', document.getElementById('upload-ocr-backend').value);
    fd.append('supersedes', document.getElementById('upload-supersedes').value.trim());
    fd.append('amends', document.getElementById('upload-amends').value.trim());
    fd.append('applicable_years', document.getElementById('upload-years').value.trim());
    fd.append('version_note', document.getElementById('upload-version-note').value.trim());
    fd.append('force_reindex', document.getElementById('upload-force-reindex').checked ? 'true' : 'false');
    fd.append('async_task', 'true');

    const btn = document.getElementById('btn-upload-submit');
    btn.disabled = true;
    try {
      appendEventLog(`Отправка ${state.selectedFiles.length} файл(ов) на сервер...`, 'info');
      const res = await fetch('/api/documents/upload', { method: 'POST', body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);

      state.selectedFiles = [];
      renderSelectedFiles();
      showToast(data.message || 'Загрузка запущена', 'info');
      trackTaskProgress(data.task_id);
    } catch (err) {
      showToast(`Ошибка загрузки: ${err.message}`, 'error');
      appendEventLog(`Ошибка запуска загрузки: ${err.message}`, 'error');
    } finally {
      btn.disabled = false;
    }
  }

  async function handleSyncFolder() {
    const ocrBackend = document.getElementById('sync-ocr-backend').value;
    const force = document.getElementById('sync-force-reindex').checked;
    const btn = document.getElementById('btn-sync-folder');

    switchTab('upload');
    btn.disabled = true;
    try {
      appendEventLog(`Запуск синхронизации папки доки/ (OCR=${ocrBackend}, force=${force})...`, 'info');
      const res = await fetch('/api/documents/sync-folder', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          force: force,
          ocr_backend: ocrBackend,
          async_task: true,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      showToast(data.message || 'Синхронизация папки доки/ запущена', 'info');
      trackTaskProgress(data.task_id);
    } catch (err) {
      showToast(`Ошибка синхронизации: ${err.message}`, 'error');
      appendEventLog(`Ошибка синхронизации: ${err.message}`, 'error');
    } finally {
      btn.disabled = false;
    }
  }

  // ---------------------------------------------------------------------------
  // Tab 2: Document Registry, Metadata Editor & Reindex/Delete
  // ---------------------------------------------------------------------------
  async function loadDocumentsRegistry() {
    try {
      const res = await fetch('/api/documents');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      state.documents = data.documents || [];
      state.reportSummary = data.report_summary || null;

      document.getElementById('reg-stat-total').textContent = String(data.total_documents ?? state.documents.length);
      const regTblEl = document.getElementById('reg-stat-tables');
      if (regTblEl) {
        const sumTbl = state.documents.reduce((acc, d) => acc + Number(d.tables_count || 0), 0);
        regTblEl.textContent = String(data.structured_tables_count ?? data.tables_count ?? sumTbl);
      }

      if (state.reportSummary && state.reportSummary.finished_at) {
        const dt = new Date(state.reportSummary.finished_at).toLocaleString('ru-RU');
        const dur = state.reportSummary.total_duration_sec ? ` (${state.reportSummary.total_duration_sec.toFixed(1)} сек)` : '';
        document.getElementById('reg-stat-last-run').textContent = dt + dur;
        document.getElementById('reg-stat-counts').textContent =
          `${state.reportSummary.indexed_files || 0} / ${state.reportSummary.skipped_files || 0} / ${state.reportSummary.failed_files || 0}`;
      }

      renderRegistryTable();
    } catch (err) {
      showToast(`Не удалось загрузить реестр: ${err.message}`, 'error');
    }
  }

  function renderRegistryTable() {
    const tbody = document.getElementById('registry-table-body');
    const q = (document.getElementById('reg-filter-search').value || '').trim().toLowerCase();
    const statusFilter = document.getElementById('reg-filter-status').value;
    const fmtFilter = document.getElementById('reg-filter-format').value;

    const filtered = state.documents.filter((doc) => {
      if (statusFilter && doc.status !== statusFilter) return false;
      const fmt = (doc.file_format || '').toLowerCase().replace(/^\./, '');
      if (fmtFilter && fmt !== fmtFilter) return false;
      if (q) {
        const hay = `${doc.title || ''} ${doc.file_name || ''} ${doc.doc_number || ''} ${doc.doc_date || ''} ${doc.doc_id || ''} ${doc.version_note || ''}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });

    if (filtered.length === 0) {
      tbody.innerHTML = `
        <tr>
          <td colspan="8" class="px-4 py-10 text-center text-slate-500">
            ${state.documents.length === 0
              ? 'База знаний пока пуста. Нажмите <strong>«Синхронизировать папку доки/»</strong> на первой вкладке для индексации документов МФТИ.'
              : 'По заданному фильтру документы не найдены.'}
          </td>
        </tr>
      `;
      return;
    }

    tbody.innerHTML = filtered
      .map((doc) => {
        const reqStr = doc.doc_number
          ? `<div class="font-mono-code font-semibold text-slate-900">№ ${escapeHtml(doc.doc_number)}</div>`
          : '<div class="text-slate-400">б/н</div>';
        const dateStr = doc.doc_date
          ? `<div class="text-[11px] font-mono-code text-slate-500">от ${escapeHtml(doc.doc_date)}</div>`
          : '';
        const noteTip = doc.version_note
          ? `<div class="mt-1 text-[11px] text-amber-800 bg-amber-50 border border-amber-200 rounded px-2 py-1 leading-snug">${escapeHtml(doc.version_note)}</div>`
          : '';
        const warnTip = doc.quality_warning
          ? `<div class="mt-1 text-[11px] text-rose-700 bg-rose-50 border border-rose-200 rounded px-2 py-1 leading-snug">⚠ ${escapeHtml(doc.quality_warning)}</div>`
          : '';
        const tablesCount = Number(doc.tables_count || 0);
        const tablesBadgeHtml = tablesCount > 0
          ? `<button type="button" data-action="explore-tables" data-doc-id="${escapeHtml(doc.doc_id)}" class="mt-1 inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-emerald-50 hover:bg-emerald-100 text-emerald-800 border border-emerald-300 font-semibold text-[11px] transition cursor-pointer" title="Открыть таблицы документа в конструкторе SQLite">
               📊 ${tablesCount} табл. SQLite
             </button>`
          : '';

        return `
          <tr class="hover:bg-slate-50/90 transition align-top">
            <td class="px-4 py-3 max-w-md">
              <div class="font-semibold text-slate-900 leading-snug">${escapeHtml(doc.title || doc.file_name)}</div>
              <div class="text-[11px] text-slate-500 font-mono-code mt-0.5 break-all">${escapeHtml(doc.file_name)}</div>
              ${noteTip}
              ${warnTip}
            </td>
            <td class="px-3 py-3 whitespace-nowrap">
              ${reqStr}
              ${dateStr}
            </td>
            <td class="px-3 py-3 text-center whitespace-nowrap">
              ${formatFileBadge(doc.file_format || doc.file_name.split('.').pop())}
            </td>
            <td class="px-3 py-3 text-center font-mono-code whitespace-nowrap">
              <span class="font-semibold text-slate-800">${doc.pages_total || 0}</span>
              <span class="text-slate-400">/</span>
              <span class="${(doc.ocr_pages || 0) > 0 ? 'text-indigo-700 font-bold' : 'text-slate-400'}" title="Распознано страниц OCR">${doc.ocr_pages || 0} OCR</span>
            </td>
            <td class="px-3 py-3 text-center font-mono-code whitespace-nowrap">
              <div>
                <span class="px-1.5 py-0.5 rounded bg-slate-100 text-slate-800 font-semibold" title="ParentSectionRecord">${doc.parent_count || 0} P</span>
                <span class="px-1.5 py-0.5 rounded bg-blue-50 text-blue-700 font-bold ml-1" title="ChunkRecord">${doc.chunk_count || 0} C</span>
              </div>
              ${tablesBadgeHtml}
            </td>
            <td class="px-3 py-3 whitespace-nowrap">
              ${formatStatusBadge(doc.status)}
            </td>
            <td class="px-3 py-3 max-w-xs">
              ${renderRelationBadges(doc)}
            </td>
            <td class="px-4 py-3 text-right whitespace-nowrap">
              <div class="inline-flex flex-wrap justify-end gap-1.5">
                ${tablesCount > 0 ? `
                  <button type="button" data-action="explore-tables" data-doc-id="${escapeHtml(doc.doc_id)}" class="px-2.5 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white font-semibold text-[11px] shadow-sm transition cursor-pointer">
                    📊 Таблицы (${tablesCount})
                  </button>
                ` : ''}
                <button type="button" data-action="inspect" data-doc-id="${escapeHtml(doc.doc_id)}" class="px-2.5 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-semibold text-[11px] shadow-sm transition cursor-pointer">
                  Открыть чанки
                </button>
                <button type="button" data-action="edit-meta" data-doc-id="${escapeHtml(doc.doc_id)}" class="px-2.5 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 font-medium text-[11px] transition cursor-pointer">
                  Метаданные
                </button>
                <button type="button" data-action="reindex" data-doc-id="${escapeHtml(doc.doc_id)}" class="px-2.5 py-1.5 rounded-lg bg-indigo-50 hover:bg-indigo-100 text-indigo-700 border border-indigo-200 font-medium text-[11px] transition cursor-pointer">
                  Переиндексировать
                </button>
                <button type="button" data-action="delete" data-doc-id="${escapeHtml(doc.doc_id)}" class="px-2.5 py-1.5 rounded-lg bg-red-50 hover:bg-red-100 text-red-700 border border-red-200 font-medium text-[11px] transition cursor-pointer">
                  Удалить из базы
                </button>
              </div>
            </td>
          </tr>
        `;
      })
      .join('');

    tbody.querySelectorAll('[data-action]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const action = btn.getAttribute('data-action');
        const docId = btn.getAttribute('data-doc-id');
        if (action === 'explore-tables') openTableExplorer({ docId: docId });
        else if (action === 'inspect') openDocumentInspector(docId);
        else if (action === 'edit-meta') openMetadataModal(docId);
        else if (action === 'reindex') reindexSingleDocument(docId);
        else if (action === 'delete') deleteSingleDocument(docId);
      });
    });
  }

  async function reindexSingleDocument(docId) {
    const doc = state.documents.find((d) => d.doc_id === docId);
    const label = doc ? doc.file_name : docId;
    try {
      switchTab('upload');
      appendEventLog(`Запрос принудительной переиндексации документа "${label}"...`, 'info');
      const res = await fetch(`/api/documents/${encodeURIComponent(docId)}/reindex`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ async_task: true }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      showToast(data.message || 'Переиндексация запущена', 'info');
      trackTaskProgress(data.task_id);
    } catch (err) {
      showToast(`Ошибка переиндексации: ${err.message}`, 'error');
    }
  }

  async function deleteSingleDocument(docId) {
    const doc = state.documents.find((d) => d.doc_id === docId);
    const name = doc ? doc.title || doc.file_name : docId;
    if (!window.confirm(`Удалить документ «${name}» из Qdrant, SQLite-таблиц, хранилища разделов и реестра?`)) {
      return;
    }
    try {
      const res = await fetch(`/api/documents/${encodeURIComponent(docId)}`, { method: 'DELETE' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      showToast(data.message || 'Документ удален из базы', 'success');
      await refreshHealth();
      await loadDocumentsRegistry();
    } catch (err) {
      showToast(`Ошибка удаления: ${err.message}`, 'error');
    }
  }

  // ---------------------------------------------------------------------------
  // Metadata Edit Modal (PATCH /api/documents/{doc_id}/metadata)
  // ---------------------------------------------------------------------------
  function openMetadataModal(docId) {
    const doc = state.documents.find((d) => d.doc_id === docId);
    if (!doc) return;

    document.getElementById('meta-edit-doc-id').value = doc.doc_id;
    document.getElementById('meta-modal-doc-id').textContent = `doc_id: ${doc.doc_id} | файл: ${doc.file_name}`;
    document.getElementById('meta-edit-title').value = doc.title || '';
    document.getElementById('meta-edit-number').value = doc.doc_number || '';
    document.getElementById('meta-edit-date').value = doc.doc_date || '';
    document.getElementById('meta-edit-status').value = doc.status || 'active';
    document.getElementById('meta-edit-supersedes').value = (doc.supersedes || []).join(', ');
    document.getElementById('meta-edit-amends').value = (doc.amends || []).join(', ');
    document.getElementById('meta-edit-years').value = (doc.applicable_years || []).join(', ');
    document.getElementById('meta-edit-version-note').value = doc.version_note || '';
    document.getElementById('meta-edit-quality-warning').value = doc.quality_warning || '';

    document.getElementById('metadata-modal').classList.remove('hidden');
  }

  function closeMetadataModal() {
    document.getElementById('metadata-modal').classList.add('hidden');
  }

  async function submitMetadataPatch(e) {
    e.preventDefault();
    const docId = document.getElementById('meta-edit-doc-id').value;
    if (!docId) return;

    const payload = {
      title: document.getElementById('meta-edit-title').value.trim(),
      doc_number: document.getElementById('meta-edit-number').value.trim(),
      doc_date: document.getElementById('meta-edit-date').value.trim(),
      status: document.getElementById('meta-edit-status').value,
      supersedes: document.getElementById('meta-edit-supersedes').value.trim(),
      amends: document.getElementById('meta-edit-amends').value.trim(),
      applicable_years: document.getElementById('meta-edit-years').value.trim(),
      version_note: document.getElementById('meta-edit-version-note').value.trim(),
      quality_warning: document.getElementById('meta-edit-quality-warning').value.trim(),
    };

    try {
      const res = await fetch(`/api/documents/${encodeURIComponent(docId)}/metadata`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);

      closeMetadataModal();
      showToast('Метаданные, граф связей, u00_meta и payload в Qdrant обновлены!', 'success');
      await refreshHealth();
      await loadDocumentsRegistry();
    } catch (err) {
      showToast(`Ошибка сохранения метаданных: ${err.message}`, 'error');
    }
  }

  // ---------------------------------------------------------------------------
  // Document Inspector Modal (unit_id -> ParentSectionRecord -> ChunkRecord)
  // ---------------------------------------------------------------------------
  async function openDocumentInspector(docId, initialSectionId = null) {
    const modal = document.getElementById('inspector-modal');
    const treeEl = document.getElementById('insp-tree-container');
    const chunksEl = document.getElementById('insp-chunks-container');

    modal.classList.remove('hidden');
    treeEl.innerHTML = '<div class="text-slate-400 p-3">Загрузка дерева разделов...</div>';
    chunksEl.innerHTML = '<div class="text-slate-400 p-6 text-center">Загрузка Parent-разделов и Child-чанков из Qdrant...</div>';

    try {
      const res = await fetch(`/api/documents/${encodeURIComponent(docId)}/chunks`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);

      state.inspector.doc = data.document;
      state.inspector.sections = data.sections || [];
      state.inspector.chunks = (data.chunks || []).sort((a, b) =>
        String(a.chunk_id || '').localeCompare(String(b.chunk_id || ''), undefined, { numeric: true })
      );
      state.inspector.selectedUnitId = null;
      state.inspector.selectedSectionId = initialSectionId;

      const doc = state.inspector.doc;
      document.getElementById('insp-doc-title').textContent = doc.title || doc.file_name;
      document.getElementById('insp-doc-status-badge').outerHTML = `<span id="insp-doc-status-badge">${formatStatusBadge(doc.status)}</span>`;
      document.getElementById('insp-doc-subtitle').textContent =
        `${doc.file_name} • ${doc.doc_number ? '№ ' + doc.doc_number : 'б/н'} ${doc.doc_date ? 'от ' + doc.doc_date : ''} • Parent-разделов: ${state.inspector.sections.length} • Child-чанков: ${state.inspector.chunks.length}`;

      renderInspectorTree();
      renderInspectorChunks();
    } catch (err) {
      chunksEl.innerHTML = `<div class="p-6 text-red-600 font-medium">Ошибка загрузки чанков: ${escapeHtml(err.message)}</div>`;
    }
  }

  function renderInspectorTree() {
    const treeEl = document.getElementById('insp-tree-container');
    const { sections, chunks, selectedUnitId, selectedSectionId } = state.inspector;

    const unitsMap = new Map();
    sections.forEach((sec) => {
      const uid = sec.unit_id || 'u01_main';
      if (!unitsMap.has(uid)) {
        unitsMap.set(uid, { unit_id: uid, unit_title: sec.unit_title || uid, sections: [], orphanChunks: [] });
      }
      unitsMap.get(uid).sections.push(sec);
    });

    const sectionIdsSet = new Set(sections.map((s) => s.section_id));
    chunks.forEach((ch) => {
      const uid = ch.unit_id || 'u01_main';
      if (!unitsMap.has(uid)) {
        unitsMap.set(uid, { unit_id: uid, unit_title: ch.unit_title || uid, sections: [], orphanChunks: [] });
      }
      if (!ch.section_id || !sectionIdsSet.has(ch.section_id)) {
        unitsMap.get(uid).orphanChunks.push(ch);
      }
    });

    const units = Array.from(unitsMap.values()).sort((a, b) => a.unit_id.localeCompare(b.unit_id));

    treeEl.innerHTML = units
      .map((u) => {
        const isUnitSelected = selectedUnitId === u.unit_id && !selectedSectionId;
        const unitChunksCount = chunks.filter((c) => (c.unit_id || 'u01_main') === u.unit_id).length;

        const secItems = u.sections
          .map((sec) => {
            const isSecSelected = selectedSectionId === sec.section_id;
            const secLabel = sec.section_num
              ? `Раздел ${sec.section_num}. ${sec.section_title || ''}`
              : sec.section_title || sec.section_id;
            return `
              <button type="button" data-select-sec="${escapeHtml(sec.section_id)}" class="w-full text-left px-2.5 py-2 rounded-lg transition flex items-start justify-between gap-2 ${
                isSecSelected
                  ? 'bg-blue-600 text-white font-semibold shadow-sm'
                  : 'hover:bg-slate-200/70 text-slate-700'
              }">
                <span class="leading-snug">${escapeHtml(secLabel)}</span>
                <span class="shrink-0 font-mono-code text-[10px] px-1.5 py-0.5 rounded ${
                  isSecSelected ? 'bg-blue-500 text-white' : 'bg-slate-200 text-slate-600'
                }">${(sec.chunk_ids || []).length} c</span>
              </button>
            `;
          })
          .join('');

        return `
          <div class="rounded-xl border border-slate-200 bg-white overflow-hidden">
            <button type="button" data-select-unit="${escapeHtml(u.unit_id)}" class="w-full px-3 py-2.5 text-left flex items-center justify-between gap-2 border-b border-slate-100 ${
              isUnitSelected ? 'bg-indigo-50 text-indigo-900 font-bold' : 'bg-slate-50 hover:bg-slate-100 font-semibold text-slate-800'
            }">
              <div class="min-w-0">
                <span class="font-mono-code text-[11px] text-indigo-600 mr-1">[${escapeHtml(u.unit_id)}]</span>
                <span class="truncate">${escapeHtml(u.unit_title)}</span>
              </div>
              <span class="text-[11px] font-mono-code text-slate-500 shrink-0">${unitChunksCount} чанков</span>
            </button>
            ${secItems ? `<div class="p-1.5 space-y-1">${secItems}</div>` : ''}
          </div>
        `;
      })
      .join('');

    treeEl.querySelectorAll('[data-select-unit]').forEach((btn) => {
      btn.addEventListener('click', () => {
        state.inspector.selectedUnitId = btn.getAttribute('data-select-unit');
        state.inspector.selectedSectionId = null;
        renderInspectorTree();
        renderInspectorChunks();
      });
    });

    treeEl.querySelectorAll('[data-select-sec]').forEach((btn) => {
      btn.addEventListener('click', () => {
        state.inspector.selectedSectionId = btn.getAttribute('data-select-sec');
        state.inspector.selectedUnitId = null;
        renderInspectorTree();
        renderInspectorChunks();
      });
    });
  }

  function renderInspectorChunks() {
    const parentBox = document.getElementById('insp-selected-parent-box');
    const chunksContainer = document.getElementById('insp-chunks-container');
    const { sections, chunks, viewMode, selectedUnitId, selectedSectionId } = state.inspector;

    if (selectedSectionId) {
      const sec = sections.find((s) => s.section_id === selectedSectionId);
      if (sec) {
        parentBox.classList.remove('hidden');
        parentBox.innerHTML = `
          <details open class="rounded-xl border border-indigo-200 bg-indigo-50/50 overflow-hidden">
            <summary class="px-4 py-3 cursor-pointer font-semibold text-xs text-indigo-900 flex items-center justify-between select-none">
              <span>Родительский раздел (ParentSectionRecord): ${escapeHtml(sec.section_num ? 'Раздел ' + sec.section_num + '. ' : '')}${escapeHtml(sec.section_title || sec.section_id)}</span>
              <span class="font-mono-code text-[11px] text-indigo-700">${sec.char_count || (sec.clean_text || '').length} симв. (чистый контекст без overlap)</span>
            </summary>
            <div class="px-4 py-3 bg-white border-t border-indigo-100 text-xs text-slate-800 chunk-text-body max-h-72 overflow-y-auto">${escapeHtml(sec.clean_text)}</div>
          </details>
        `;
      } else {
        parentBox.classList.add('hidden');
      }
    } else {
      parentBox.classList.add('hidden');
    }

    const filteredChunks = chunks.filter((ch) => {
      if (selectedSectionId) return ch.section_id === selectedSectionId;
      if (selectedUnitId) return (ch.unit_id || 'u01_main') === selectedUnitId;
      return true;
    });

    if (filteredChunks.length === 0) {
      chunksContainer.innerHTML = '<div class="p-6 text-center text-slate-400 text-xs">Нет чанков в выбранном узле дерева.</div>';
      return;
    }

    chunksContainer.innerHTML = filteredChunks
      .map((ch) => {
        let bodyHtml = '';
        if (viewMode === 'raw_text') {
          bodyHtml = renderTextWithOverlapHighlight(ch);
        } else if (viewMode === 'embed_text') {
          bodyHtml = `<span class="font-mono-code text-[11.5px] text-slate-800">${escapeHtml(ch.embed_text || ch.raw_text)}</span>`;
        } else {
          bodyHtml = escapeHtml(ch.clean_text || ch.raw_text);
        }

        const roleTags = (ch.roles || [])
          .map((r) => `<span class="px-2 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-200 text-[11px]">👤 ${escapeHtml(r)}</span>`)
          .join('');
        const termTags = (ch.terms || [])
          .map((t) => `<span class="px-2 py-0.5 rounded bg-amber-50 text-amber-800 border border-amber-200 text-[11px] font-medium">📘 ${escapeHtml(t)}</span>`)
          .join('');
        const pagesStr = (ch.page_numbers || []).length ? `стр. ${ch.page_numbers.join(', ')}` : '';
        const dedupBadge = (ch.also_in_files || []).length
          ? `<span class="px-2 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-200 text-[11px]">Дублируется в: ${escapeHtml(ch.also_in_files.join(', '))}</span>`
          : '';
        const tableId = ch.table_id || (ch.metadata && ch.metadata.table_id) || '';
        const tableBtn = tableId
          ? `<button type="button" data-open-table-id="${escapeHtml(tableId)}" class="px-2 py-0.5 rounded bg-emerald-50 hover:bg-emerald-100 text-emerald-800 border border-emerald-300 font-semibold text-[11px] cursor-pointer">
               📊 SQLite: ${escapeHtml(tableId)} →
             </button>`
          : '';

        return `
          <article class="rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden">
            <div class="px-4 py-2.5 bg-slate-50 border-b border-slate-200 flex flex-wrap items-center justify-between gap-2 text-xs">
              <div class="flex flex-wrap items-center gap-2">
                <span class="font-mono-code font-bold text-slate-900 bg-white px-2 py-0.5 rounded border border-slate-200">
                  ${ch.clause_no ? 'п. ' + escapeHtml(ch.clause_no) : 'Без номера п.'}
                </span>
                ${formatZoneBadge(ch.zone)}
                ${tableBtn}
                <span class="font-mono-code text-[11px] text-slate-500">${escapeHtml(ch.chunk_id)}</span>
                ${pagesStr ? `<span class="text-[11px] text-slate-500 font-mono-code">• ${escapeHtml(pagesStr)}</span>` : ''}
              </div>
              <div class="flex items-center gap-2 text-[11px] font-mono-code text-slate-500">
                <span>${ch.char_count || (ch.clean_text || '').length} симв.</span>
                ${ch.prev_id ? `<span title="prev_id: ${escapeHtml(ch.prev_id)}">← prev</span>` : ''}
                ${ch.next_id ? `<span title="next_id: ${escapeHtml(ch.next_id)}">next →</span>` : ''}
              </div>
            </div>

            <div class="p-4 text-xs text-slate-800 chunk-text-body">${bodyHtml}</div>

            ${(roleTags || termTags || dedupBadge) ? `
              <div class="px-4 py-2 bg-slate-50/70 border-t border-slate-100 flex flex-wrap gap-1.5">
                ${roleTags}
                ${termTags}
                ${dedupBadge}
              </div>
            ` : ''}
          </article>
        `;
      })
      .join('');

    chunksContainer.querySelectorAll('[data-open-table-id]').forEach((btn) => {
      btn.addEventListener('click', () => {
        openTableExplorer({ tableId: btn.getAttribute('data-open-table-id') });
      });
    });
  }

  // ---------------------------------------------------------------------------
  // Tab 3: Hybrid Search & Autonomous Table-Aware AI Agent (/api/search & /api/ask)
  // ---------------------------------------------------------------------------
  function renderTableShowcasePresets() {
    const container = document.getElementById('table-presets-container');
    if (!container) return;

    container.innerHTML = TABLE_SHOWCASE_PRESETS.map(
      (p, idx) => `
        <button type="button" data-showcase-idx="${idx}" title="${escapeHtml(p.hint)}" class="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl border text-xs font-medium transition cursor-pointer ${p.btnCls}">
          <span class="px-1.5 py-0.5 rounded-md text-[10px] font-bold border ${p.badgeCls}">${escapeHtml(p.badge)}</span>
          <span class="font-semibold">${escapeHtml(p.title)}</span>
        </button>
      `
    ).join('');

    container.querySelectorAll('[data-showcase-idx]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const idx = parseInt(btn.getAttribute('data-showcase-idx'), 10);
        const preset = TABLE_SHOWCASE_PRESETS[idx];
        if (!preset) return;
        document.getElementById('search-query-input').value = preset.query;
        setPlaygroundMode('ask');
        executeAgentAsk();
      });
    });
  }

  async function loadSearchPresets() {
    renderTableShowcasePresets();

    const container = document.getElementById('search-presets-container');
    try {
      const res = await fetch('/api/search/presets');
      if (res.ok) {
        state.presets = await res.json();
      }
    } catch (e) {
      state.presets = [];
    }

    container.innerHTML = state.presets
      .map((p, idx) => {
        const isOos = (p.id || '').includes('9') || (p.id || '').includes('10');
        const isTablePreset = p.preset_mode === 'tool_query' || (p.id || '').startsWith('table_');
        let cls = 'bg-slate-100 hover:bg-blue-50 text-slate-700 hover:text-blue-700 border-slate-200 hover:border-blue-300';
        if (isOos) {
          cls = 'bg-rose-50 hover:bg-rose-100 text-rose-700 border-rose-200';
        } else if (isTablePreset) {
          cls = 'bg-emerald-50 hover:bg-emerald-100 text-emerald-800 border-emerald-200';
        }
        return `
          <button type="button" data-preset-idx="${idx}" title="${escapeHtml(p.expected || p.query)}" class="px-2.5 py-1 rounded-lg border text-xs font-medium transition cursor-pointer ${cls}">
            ${idx + 1}. ${escapeHtml(p.title)}
          </button>
        `;
      })
      .join('');

    container.querySelectorAll('[data-preset-idx]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const idx = parseInt(btn.getAttribute('data-preset-idx'), 10);
        const preset = state.presets[idx];
        if (!preset) return;
        document.getElementById('search-query-input').value = preset.query;
        if (state.playgroundMode === 'ask') {
          executeAgentAsk();
        } else {
          executeHybridSearch();
        }
      });
    });
  }

  /**
   * Renders a TablePassport card (for XL sheets and large >=25-row Word/PDF tables).
   */
  function renderTablePassportCard(tp) {
    const title = tp.table_title || tp.sheet_name || tp.section_title || tp.file_name || tp.table_id;
    const cols = tp.columns || [];
    const colCount = tp.col_count || cols.length;
    const rowCount = tp.row_count || 0;

    const colPills = cols.slice(0, 8).map((c) => {
      const icon = c.has_formulas ? 'ƒx ' : c.inferred_type === 'number' || c.inferred_type === 'numeric' ? '# ' : c.is_hierarchy ? '🌳 ' : '';
      const cls = c.has_formulas
        ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
        : c.is_hierarchy
        ? 'bg-indigo-50 text-indigo-800 border-indigo-200'
        : 'bg-slate-100 text-slate-700 border-slate-200';
      return `<span class="inline-flex items-center px-2 py-0.5 rounded text-[10.5px] font-medium border ${cls}" title="Тип: ${escapeHtml(c.inferred_type || 'text')}">${escapeHtml(icon + c.name)}</span>`;
    });
    if (cols.length > 8) {
      colPills.push(`<span class="px-2 py-0.5 rounded text-[10.5px] font-mono-code bg-slate-50 text-slate-500 border border-slate-200">+${cols.length - 8} ещё</span>`);
    }

    const formulaBadge = tp.has_formulas
      ? `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-semibold bg-amber-50 text-amber-900 border border-amber-200">🧮 Формул: ${tp.formula_count || 'есть'}</span>`
      : '';

    const alsoBadge = (tp.also_in_files || []).length
      ? `<div class="text-[11px] text-indigo-700 font-medium">🔗 Дедупликация (также в): ${escapeHtml(tp.also_in_files.join(', '))}</div>`
      : '';

    return `
      <div class="rounded-2xl border border-emerald-300 bg-gradient-to-br from-white via-emerald-50/20 to-slate-50 p-4 shadow-sm flex flex-col justify-between gap-3">
        <div class="space-y-2">
          <div class="flex flex-wrap items-center justify-between gap-2">
            <span class="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-emerald-100 text-emerald-900 border border-emerald-300">
              📊 XL / Большая таблица · SQLite + Tool Query (${rowCount} строк)
            </span>
            <div class="flex items-center gap-1.5">
              ${formatFileBadge(tp.file_format || (tp.file_name || '').split('.').pop())}
              ${formulaBadge}
            </div>
          </div>

          <div>
            <div class="font-bold text-sm text-slate-900 leading-snug">${escapeHtml(title)}</div>
            <div class="text-[11px] text-slate-600 font-mono-code mt-0.5">
              ${escapeHtml(tp.file_name || '')}
              ${tp.sheet_name ? ` • лист: «${escapeHtml(tp.sheet_name)}»` : ''}
              • ${rowCount} стр. × ${colCount} кол.
            </div>
            ${alsoBadge}
          </div>

          ${colPills.length ? `<div class="flex flex-wrap gap-1 pt-0.5">${colPills.join('')}</div>` : ''}
        </div>

        <div class="pt-2 border-t border-emerald-100 flex items-center justify-between gap-2">
          <code class="text-[10.5px] font-mono-code text-slate-500 truncate">${escapeHtml(tp.table_id)}</code>
          <button type="button" data-open-table-id="${escapeHtml(tp.table_id)}" class="shrink-0 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white font-semibold text-xs shadow-sm transition cursor-pointer">
            <span>Открыть конструктор запросов к таблице</span>
          </button>
        </div>
      </div>
    `;
  }

  /**
   * Renders architecture delivery badge inside individual SearchHit cards:
   *  - `📊 XL / Большая таблица · SQLite + Tool Query (N строк)` + Explorer button
   *  - vs `📄 Компактная таблица Word/PDF · Встроена целиком в Parent`
   */
  function renderHitTableArchitectureBar(hit) {
    const meta = hit.metadata || {};
    const deliveryMode = meta.table_delivery_mode || '';
    const tableId = hit.table_id || meta.table_id || '';
    const linkedTp = tableId ? state.linkedTablesById.get(tableId) : null;

    if (deliveryMode === 'tool_query' || linkedTp || (tableId && deliveryMode !== 'inline_parent')) {
      const rowsLabel = linkedTp && linkedTp.row_count ? `${linkedTp.row_count} строк` : 'SQLite';
      const targetId = tableId || (linkedTp ? linkedTp.table_id : '');
      return `
        <div class="px-4 py-2 bg-emerald-50/80 border-b border-emerald-200 flex flex-wrap items-center justify-between gap-2 text-xs">
          <div class="flex flex-wrap items-center gap-2">
            <span class="inline-flex items-center px-2.5 py-0.5 rounded-full font-bold text-[11px] bg-emerald-100 text-emerald-900 border border-emerald-300">
              📊 XL / Большая таблица · SQLite + Tool Query (${escapeHtml(rowsLabel)})
            </span>
            <span class="text-[11px] text-emerald-900">В Qdrant хранится 8-строчный поисковый срез, а полная таблица доступна через SQL-фильтрацию и агрегацию</span>
          </div>
          ${targetId ? `
            <button type="button" data-open-table-id="${escapeHtml(targetId)}" class="px-2.5 py-1 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white font-semibold text-[11px] shadow-sm transition cursor-pointer">
              Открыть конструктор запросов к таблице
            </button>
          ` : ''}
        </div>
      `;
    }

    if (deliveryMode === 'inline_parent' || hit.zone === 'table') {
      return `
        <div class="px-4 py-2 bg-sky-50/80 border-b border-sky-200 flex flex-wrap items-center justify-between gap-2 text-xs">
          <div class="flex flex-wrap items-center gap-2">
            <span class="inline-flex items-center px-2.5 py-0.5 rounded-full font-bold text-[11px] bg-sky-100 text-sky-900 border border-sky-300">
              📄 Компактная таблица Word/PDF · Встроена целиком в Parent
            </span>
            <span class="text-[11px] text-sky-900">Таблица (&lt; 25 строк) передается в контекст целиком через ParentSectionRecord без лишних SQL-раундов</span>
          </div>
        </div>
      `;
    }

    return '';
  }

  function renderSearchHitCard(hit, rankLabel, isAmendmentCard = false) {
    const includeParent = document.getElementById('search-include-parent').checked;
    const borderCls = isAmendmentCard
      ? 'border-purple-300 bg-purple-50/20'
      : 'border-slate-200 bg-white';

    const docHeader = [
      hit.doc_type || 'Документ',
      hit.doc_number ? `№ ${hit.doc_number}` : '',
      hit.doc_date ? `от ${hit.doc_date}` : '',
    ]
      .filter(Boolean)
      .join(' ');

    const locParts = [];
    if (hit.unit_title || hit.unit_id) locParts.push(`${hit.unit_title || hit.unit_id}`);
    if (hit.section_num || hit.section_title) {
      locParts.push(`Раздел ${hit.section_num ? hit.section_num + '. ' : ''}${hit.section_title || ''}`.trim());
    }
    if (hit.clause_no) locParts.push(`п. ${hit.clause_no}`);
    if (hit.page_numbers && hit.page_numbers.length) locParts.push(`стр. ${hit.page_numbers.join(', ')}`);

    const roleTags = (hit.roles || [])
      .map((r) => `<span class="px-2 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-200 text-[11px]">👤 ${escapeHtml(r)}</span>`)
      .join('');
    const termTags = (hit.terms || [])
      .map((t) => `<span class="px-2 py-0.5 rounded bg-amber-50 text-amber-800 border border-amber-200 text-[11px]">📘 ${escapeHtml(t)}</span>`)
      .join('');

    const parentText = (hit.parent_section && hit.parent_section.clean_text) || hit.parent_text || '';
    const parentHtml = parentText
      ? `
        <details ${includeParent ? 'open' : ''} class="mt-3 rounded-xl border border-slate-200 bg-slate-50 overflow-hidden">
          <summary class="px-3.5 py-2 cursor-pointer text-xs font-semibold text-slate-700 hover:bg-slate-100 flex items-center justify-between select-none">
            <span>Полный чистый родительский раздел (ParentSectionRecord)</span>
            <span class="font-mono-code text-[11px] text-slate-500">${parentText.length} симв.</span>
          </summary>
          <div class="px-3.5 py-3 bg-white border-t border-slate-200 text-xs text-slate-700 chunk-text-body max-h-80 overflow-y-auto">${escapeHtml(parentText)}</div>
        </details>
      `
      : '';

    const noteCallout = hit.version_note
      ? `<div class="mt-2 px-3 py-1.5 rounded-lg bg-amber-50 border border-amber-200 text-[11px] text-amber-900"><strong>Примечание редакции:</strong> ${escapeHtml(hit.version_note)}</div>`
      : '';

    const tableArchBar = renderHitTableArchitectureBar(hit);

    return `
      <article class="rounded-2xl border ${borderCls} shadow-sm overflow-hidden">
        <!-- Hit Top Metadata Bar -->
        <div class="px-4 py-3 bg-slate-50/90 border-b border-slate-200 flex flex-wrap items-center justify-between gap-2 text-xs">
          <div class="flex flex-wrap items-center gap-2">
            <span class="px-2.5 py-0.5 rounded-lg font-mono-code font-bold ${
              isAmendmentCard ? 'bg-purple-600 text-white' : 'bg-slate-900 text-white'
            }">${escapeHtml(rankLabel)}</span>
            <span class="font-bold text-slate-900">${escapeHtml(hit.title || hit.file_name)}</span>
            <span class="font-mono-code text-slate-600">(${escapeHtml(docHeader)})</span>
            ${formatStatusBadge(hit.status)}
            ${formatZoneBadge(hit.zone)}
          </div>

          <div class="flex flex-wrap items-center gap-1.5 font-mono-code text-[11px]">
            <span class="px-2 py-0.5 rounded bg-blue-50 text-blue-800 border border-blue-200 font-semibold" title="Reciprocal Rank Fusion Score">
              RRF: ${Number(hit.rrf_score || hit.score || 0).toFixed(4)}
            </span>
            <span class="px-2 py-0.5 rounded bg-indigo-50 text-indigo-800 border border-indigo-200 font-semibold" title="Dense Cosine Similarity (baai/bge-m3)">
              Dense cos: ${Number(hit.dense_score || 0).toFixed(4)}
            </span>
            ${hit.doc_id ? `
              <button type="button" data-open-insp-doc="${escapeHtml(hit.doc_id)}" data-open-insp-sec="${escapeHtml(hit.section_id || '')}" class="px-2 py-0.5 rounded bg-white hover:bg-slate-100 text-slate-700 border border-slate-300 font-sans font-medium cursor-pointer">
                Открыть в инспекторе →
              </button>
            ` : ''}
          </div>
        </div>

        ${tableArchBar}

        <!-- Exact Source Citation Breadcrumb -->
        <div class="px-4 py-2 bg-white border-b border-slate-100 text-[11px] text-slate-600 flex flex-wrap items-center justify-between gap-2">
          <div>
            <span class="font-semibold text-slate-700">Источник:</span>
            <code class="font-mono-code text-slate-800">${escapeHtml(hit.file_name)}</code>
            ${locParts.length ? ` <span class="text-slate-400">→</span> <span class="font-medium text-slate-800">${escapeHtml(locParts.join(' • '))}</span>` : ''}
          </div>
          ${(hit.also_in_files || []).length ? `<span class="text-indigo-700 font-medium">Также в: ${escapeHtml(hit.also_in_files.join(', '))}</span>` : ''}
        </div>

        <!-- Child Chunk Text + Optional Hydrated Parent Section -->
        <div class="p-4">
          <div class="text-xs sm:text-[13px] text-slate-900 chunk-text-body">${renderTextWithOverlapHighlight(hit)}</div>
          ${noteCallout}
          ${parentHtml}
        </div>

        ${(roleTags || termTags) ? `
          <div class="px-4 py-2 bg-slate-50/60 border-t border-slate-100 flex flex-wrap gap-1.5">
            ${roleTags}
            ${termTags}
          </div>
        ` : ''}
      </article>
    `;
  }

  /**
   * Renders the AI Answer card + expandable visual ToolCallTrace block.
   */
  function renderAgentAnswerCard(askResp) {
    const container = document.getElementById('agent-answer-container');
    if (!container) return;

    container.classList.remove('hidden');
    const toolCalls = askResp.tool_calls || [];
    const usedTools = Boolean(askResp.used_table_tools || toolCalls.length > 0);
    const isOos = Boolean(askResp.search_response && askResp.search_response.is_out_of_scope);

    let modePill = '';
    if (usedTools) {
      modePill = `<span class="px-2.5 py-1 rounded-full text-[11px] font-bold bg-emerald-100 text-emerald-900 border border-emerald-300">⚡ SQLite Tool Calling: ${toolCalls.length} выз.</span>`;
    } else if (isOos) {
      modePill = `<span class="px-2.5 py-1 rounded-full text-[11px] font-bold bg-amber-100 text-amber-900 border border-amber-300">🛡️ Out-of-Scope (0 вызовов инструмента)</span>`;
    } else {
      modePill = `<span class="px-2.5 py-1 rounded-full text-[11px] font-bold bg-sky-100 text-sky-900 border border-sky-300">📄 Контекст ParentSection (0 SQL-дозапросов)</span>`;
    }

    const tracesHtml = toolCalls.length > 0
      ? `
        <details open class="border-t border-indigo-100 bg-slate-50/80">
          <summary class="px-5 py-3 cursor-pointer text-xs font-bold text-indigo-950 hover:bg-indigo-50/60 flex flex-wrap items-center justify-between gap-2 select-none">
            <span>🔧 Трассировка вызовов инструмента таблиц (ToolCallTrace · ${toolCalls.length} шаг${toolCalls.length === 1 ? '' : toolCalls.length < 5 ? 'а' : 'ов'})</span>
            <span class="text-[11px] font-normal text-slate-500">Детерминированные запросы ИИ-агента к SQLite tables_store.db</span>
          </summary>
          <div class="px-5 pb-4 pt-1 space-y-3">
            ${toolCalls
              .map((tc, idx) => {
                const args = tc.arguments || {};
                const op = args.operation || (tc.tool_name === 'get_table_schema' ? 'get_schema' : 'query');
                const tblId = args.table_id || '';
                const argsPretty = JSON.stringify(args, null, 2);
                const previewHtml = tc.result_preview
                  ? renderMarkdownSafe(tc.result_preview)
                  : '<span class="text-slate-400">Нет превью строк</span>';

                return `
                  <div class="rounded-xl border border-indigo-200 bg-white shadow-sm overflow-hidden">
                    <div class="px-3.5 py-2 bg-indigo-50/70 border-b border-indigo-100 flex flex-wrap items-center justify-between gap-2 text-xs">
                      <div class="flex flex-wrap items-center gap-2">
                        <span class="px-2 py-0.5 rounded bg-indigo-600 text-white font-mono-code font-bold text-[11px]">Шаг #${tc.step || idx + 1}</span>
                        <code class="font-mono-code font-bold text-indigo-950">${escapeHtml(tc.tool_name || 'query_structured_table')}</code>
                        <span class="px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-900 border border-emerald-300 font-mono-code font-semibold text-[11px]">op: ${escapeHtml(op)}</span>
                        <span class="px-2 py-0.5 rounded bg-white border border-slate-200 font-mono-code text-[11px] text-slate-700">Найдено строк: <strong>${tc.matched_count ?? 0}</strong></span>
                      </div>
                      <div class="flex items-center gap-2 font-mono-code text-[11px] text-slate-500">
                        <span>${Number(tc.latency_ms || 0).toFixed(1)} мс</span>
                        ${tblId ? `
                          <button type="button" data-open-table-id="${escapeHtml(tblId)}" class="px-2 py-0.5 rounded bg-emerald-600 hover:bg-emerald-700 text-white font-sans font-semibold cursor-pointer">
                            Открыть таблицу →
                          </button>
                        ` : ''}
                      </div>
                    </div>

                    <div class="p-3.5 space-y-2.5 text-xs">
                      ${tc.summary ? `<div class="font-medium text-slate-800">${escapeHtml(tc.summary)}</div>` : ''}
                      <div class="grid grid-cols-1 lg:grid-cols-12 gap-3">
                        <div class="lg:col-span-4">
                          <div class="text-[10.5px] font-semibold uppercase tracking-wider text-slate-500 mb-1">Аргументы вызова (JSON):</div>
                          <pre class="bg-slate-900 text-emerald-300 font-mono-code text-[11px] p-2.5 rounded-lg overflow-x-auto max-h-48">${escapeHtml(argsPretty)}</pre>
                        </div>
                        <div class="lg:col-span-8 min-w-0">
                          <div class="text-[10.5px] font-semibold uppercase tracking-wider text-slate-500 mb-1">Результат выборки / агрегации из SQLite:</div>
                          <div class="ai-markdown-body text-xs bg-slate-50 border border-slate-200 rounded-lg p-2.5 max-h-56 overflow-auto">${previewHtml}</div>
                        </div>
                      </div>
                    </div>
                  </div>
                `;
              })
              .join('')}
          </div>
        </details>
      `
      : `
        <div class="px-5 py-2.5 border-t border-slate-100 bg-slate-50/70 text-[11px] text-slate-600 flex flex-wrap items-center justify-between gap-2">
          <span>
            ${isOos
              ? '🛡️ Детектор Out-of-Scope заблокировал нерелевантный запрос до вызова табличных инструментов (0 лишних токенов).'
              : '📄 Трассировка ToolCallTrace: 0 вызовов SQLite — полный ответ получен напрямую из найденных нормативных пунктов и компактных ParentSectionRecord.'}
          </span>
        </div>
      `;

    container.innerHTML = `
      <div class="rounded-2xl border-2 border-indigo-200 bg-white shadow-md overflow-hidden">
        <!-- AI Answer Header -->
        <div class="px-5 py-3.5 bg-gradient-to-r from-indigo-50 via-emerald-50/60 to-white border-b border-indigo-100 flex flex-wrap items-center justify-between gap-2">
          <div class="flex flex-wrap items-center gap-2">
            <span class="text-sm font-bold text-slate-900">🤖 ИИ-ответ по нормативной базе и таблицам МФТИ</span>
            ${modePill}
          </div>
          <div class="flex flex-wrap items-center gap-2 font-mono-code text-xs">
            ${askResp.model_used ? `<span class="px-2.5 py-0.5 rounded-lg bg-white border border-indigo-200 text-indigo-900 font-semibold">Модель: ${escapeHtml(askResp.model_used)}</span>` : ''}
            <span class="px-2.5 py-0.5 rounded-lg bg-white border border-slate-200 text-slate-700">${Number(askResp.latency_ms || 0).toFixed(0)} мс</span>
          </div>
        </div>

        <!-- Formatted Grounded Answer -->
        <div class="p-5 text-xs sm:text-sm ai-markdown-body">
          ${renderMarkdownSafe(askResp.answer || 'Ответ не сформирован.')}
        </div>

        <!-- ToolCallTrace Section -->
        ${tracesHtml}
      </div>
    `;

    container.querySelectorAll('[data-open-table-id]').forEach((btn) => {
      btn.addEventListener('click', () => {
        openTableExplorer({ tableId: btn.getAttribute('data-open-table-id') });
      });
    });
  }

  async function executeHybridSearch(e) {
    if (e) e.preventDefault();
    const query = document.getElementById('search-query-input').value.trim();
    if (!query) {
      showToast('Введите текст поискового запроса', 'warning');
      return;
    }

    setPlaygroundMode('search');
    const topK = parseInt(document.getElementById('search-top-k').value, 10) || 5;
    const rawStatus = document.getElementById('search-status-filter').value;
    const rawYear = document.getElementById('search-year-filter').value.trim();
    const includeParent = document.getElementById('search-include-parent').checked;
    const includeAmendments = document.getElementById('search-include-amendments').checked;

    const payload = {
      query: query,
      top_k: topK,
      status_filter: rawStatus ? rawStatus.split(',') : null,
      year_filter: rawYear ? parseInt(rawYear, 10) : null,
      include_parent: includeParent,
      include_amendments: includeAmendments,
      include_superseded: true,
    };

    const btn = document.getElementById('btn-run-search');
    btn.disabled = true;

    // Hide previous AI answer card when running pure vector/BM25 search
    const agentContainer = document.getElementById('agent-answer-container');
    if (agentContainer) {
      agentContainer.classList.add('hidden');
      agentContainer.innerHTML = '';
    }

    try {
      const res = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);

      renderSearchResults(data, data.linked_tables || []);
    } catch (err) {
      showToast(`Ошибка поиска: ${err.message}`, 'error');
    } finally {
      btn.disabled = false;
    }
  }

  async function executeAgentAsk(e) {
    if (e) e.preventDefault();
    const query = document.getElementById('search-query-input').value.trim();
    if (!query) {
      showToast('Введите вопрос для ИИ-агента', 'warning');
      return;
    }

    setPlaygroundMode('ask');
    const topK = parseInt(document.getElementById('search-top-k').value, 10) || 5;
    const rawStatus = document.getElementById('search-status-filter').value;
    const rawYear = document.getElementById('search-year-filter').value.trim();
    const enableTools = document.getElementById('ask-enable-table-tools').checked;

    const payload = {
      query: query,
      top_k: topK,
      status_filter: rawStatus ? rawStatus.split(',') : null,
      year_filter: rawYear ? parseInt(rawYear, 10) : null,
      enable_table_tools: enableTools,
      max_tool_rounds: 4,
    };

    const btnAsk = document.getElementById('btn-run-ask');
    const btnSearch = document.getElementById('btn-run-search');
    btnAsk.disabled = true;
    btnSearch.disabled = true;

    const wrapper = document.getElementById('search-results-wrapper');
    const agentContainer = document.getElementById('agent-answer-container');
    wrapper.classList.remove('hidden');
    agentContainer.classList.remove('hidden');
    agentContainer.innerHTML = `
      <div class="rounded-2xl border border-indigo-200 bg-indigo-50/50 p-5 flex items-center gap-3 text-xs text-indigo-950 shadow-sm">
        <div class="w-5 h-5 rounded-full border-2 border-indigo-600 border-t-transparent animate-spin shrink-0"></div>
        <div>
          <div class="font-bold text-sm">Выполняется гибридный поиск RRF и синтез ответа ИИ-агентом...</div>
          <div class="text-indigo-700 mt-0.5">При обнаружении связанных Excel или больших таблиц агент автоматически вызовет <code>query_structured_table</code> к SQLite.</div>
        </div>
      </div>
    `;

    try {
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const askData = await res.json();
      if (!res.ok) throw new Error(askData.detail || `HTTP ${res.status}`);

      const combinedLinked =
        (askData.linked_tables && askData.linked_tables.length > 0)
          ? askData.linked_tables
          : ((askData.search_response && askData.search_response.linked_tables) || []);

      // Index linked tables BEFORE rendering cards so hit badges resolve row counts
      state.linkedTablesById = new Map();
      combinedLinked.forEach((tp) => {
        if (tp && tp.table_id) state.linkedTablesById.set(tp.table_id, tp);
      });

      renderAgentAnswerCard(askData);
      if (askData.search_response) {
        renderSearchResults(askData.search_response, combinedLinked);
      }
    } catch (err) {
      agentContainer.innerHTML = `
        <div class="rounded-2xl border border-red-300 bg-red-50 p-4 text-xs text-red-900">
          <strong>Ошибка генерации ИИ-ответа:</strong> ${escapeHtml(err.message)}
        </div>
      `;
      showToast(`Ошибка ИИ-ответа: ${err.message}`, 'error');
    } finally {
      btnAsk.disabled = false;
      btnSearch.disabled = false;
    }
  }

  function renderSearchResults(data, explicitLinkedTables = null) {
    if (!data) return;
    const wrapper = document.getElementById('search-results-wrapper');
    const banner = document.getElementById('search-scope-banner');
    const alertsEl = document.getElementById('search-alerts-container');
    const linkedSec = document.getElementById('search-linked-tables-section');
    const linkedList = document.getElementById('search-linked-tables-list');
    const amendSec = document.getElementById('search-amendments-section');
    const amendList = document.getElementById('search-amendments-list');
    const hitsList = document.getElementById('search-hits-list');

    wrapper.classList.remove('hidden');

    const linkedTables = explicitLinkedTables || data.linked_tables || [];
    state.linkedTablesById = new Map();
    linkedTables.forEach((tp) => {
      if (tp && tp.table_id) state.linkedTablesById.set(tp.table_id, tp);
    });

    // 1. Out-of-Scope / Confidence Banner
    if (data.is_out_of_scope) {
      banner.className = 'rounded-2xl p-4 border border-amber-300 bg-amber-50 text-amber-950 shadow-sm';
      banner.innerHTML = `
        <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
          <div>
            <div class="flex items-center gap-2 font-bold text-sm text-amber-900">
              <span>⚠️ Запрос вне предметной области базы МФТИ (Out-of-Scope = True)</span>
            </div>
            <p class="text-xs text-amber-800 mt-1">
              Система определила низкое семантическое и лексическое пересечение с внутренними нормативными документами МФТИ. Достоверный ответ по регламентам дать нельзя.
            </p>
          </div>
          <div class="flex flex-wrap gap-2 font-mono-code text-xs shrink-0">
            <span class="px-2.5 py-1 rounded-lg bg-white border border-amber-200">Confidence: ${Number(data.confidence_score || 0).toFixed(3)}</span>
            <span class="px-2.5 py-1 rounded-lg bg-white border border-amber-200">Max Dense: ${Number(data.top_dense_score || 0).toFixed(3)}</span>
            <span class="px-2.5 py-1 rounded-lg bg-white border border-amber-200">Lexical: ${Number(data.lexical_overlap || 0).toFixed(2)}</span>
            <span class="px-2.5 py-1 rounded-lg bg-white border border-amber-200">${Number(data.latency_ms || 0).toFixed(0)} мс</span>
          </div>
        </div>
      `;
    } else {
      banner.className = 'rounded-2xl p-4 border border-emerald-200 bg-emerald-50/80 text-emerald-950 shadow-sm';
      banner.innerHTML = `
        <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
          <div>
            <div class="flex items-center gap-2 font-bold text-sm text-emerald-900">
              <span>✅ Уверенный поиск по нормативной базе МФТИ (In-Scope)</span>
            </div>
            ${data.expanded_query && data.expanded_query !== data.query
              ? `<p class="text-[11px] text-emerald-800 mt-1">Расширение глоссария: <code class="font-mono-code bg-white/80 px-1.5 py-0.5 rounded">${escapeHtml(data.expanded_query)}</code></p>`
              : ''}
          </div>
          <div class="flex flex-wrap gap-2 font-mono-code text-xs shrink-0">
            <span class="px-2.5 py-1 rounded-lg bg-white border border-emerald-200 font-semibold">Confidence: ${Number(data.confidence_score || 0).toFixed(3)}</span>
            <span class="px-2.5 py-1 rounded-lg bg-white border border-emerald-200">Top Dense: ${Number(data.top_dense_score || 0).toFixed(3)}</span>
            <span class="px-2.5 py-1 rounded-lg bg-white border border-emerald-200">Найдено: ${data.total_found || (data.hits || []).length}</span>
            ${linkedTables.length > 0 ? `<span class="px-2.5 py-1 rounded-lg bg-emerald-100 border border-emerald-300 text-emerald-900 font-semibold">Таблиц SQLite: ${linkedTables.length}</span>` : ''}
            <span class="px-2.5 py-1 rounded-lg bg-white border border-emerald-200">${Number(data.latency_ms || 0).toFixed(0)} мс</span>
          </div>
        </div>
      `;
    }

    // 2. Version Alerts & Quality Alerts
    const alertHtml = [];
    (data.version_alerts || []).forEach((al) => {
      alertHtml.push(`
        <div class="rounded-xl border border-purple-300 bg-purple-50 px-4 py-3 text-xs text-purple-950 flex items-start gap-2.5">
          <span class="font-bold text-sm leading-none mt-0.5">⚡</span>
          <div><strong class="font-semibold">Контроль редакции акта:</strong> ${escapeHtml(al)}</div>
        </div>
      `);
    });
    (data.quality_alerts || []).forEach((qa) => {
      alertHtml.push(`
        <div class="rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-xs text-amber-950 flex items-start gap-2.5">
          <span class="font-bold text-sm leading-none mt-0.5">ℹ️</span>
          <div><strong class="font-semibold">Примечание к источнику:</strong> ${escapeHtml(qa)}</div>
        </div>
      `);
    });
    alertsEl.innerHTML = alertHtml.join('');

    // 3. Linked Structured Tables (TablePassport cards)
    if (linkedSec && linkedList) {
      if (linkedTables.length > 0 && !data.is_out_of_scope) {
        linkedSec.classList.remove('hidden');
        linkedList.innerHTML = linkedTables.map((tp) => renderTablePassportCard(tp)).join('');
      } else {
        linkedSec.classList.add('hidden');
        linkedList.innerHTML = '';
      }
    }

    // 4. Amendment Chunks
    const amends = data.amendment_chunks || [];
    if (amends.length > 0) {
      amendSec.classList.remove('hidden');
      amendList.innerHTML = amends
        .map((hit, idx) => renderSearchHitCard(hit, `Поправка #${idx + 1}`, true))
        .join('');
    } else {
      amendSec.classList.add('hidden');
      amendList.innerHTML = '';
    }

    // 5. Ranked Primary Hits
    const hits = data.hits || [];
    if (hits.length === 0) {
      hitsList.innerHTML = `
        <div class="bg-white rounded-2xl border border-slate-200 p-8 text-center text-sm text-slate-500">
          По данному запросу релевантные нормативные пункты не возвращены (проверьте индексацию документов во вкладке «Реестр»).
        </div>
      `;
    } else {
      hitsList.innerHTML = hits.map((hit, i) => renderSearchHitCard(hit, `#${i + 1}`, false)).join('');
    }

    // Wire "Открыть в инспекторе" buttons
    wrapper.querySelectorAll('[data-open-insp-doc]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const dId = btn.getAttribute('data-open-insp-doc');
        const sId = btn.getAttribute('data-open-insp-sec') || null;
        openDocumentInspector(dId, sId);
      });
    });

    // Wire "Открыть конструктор запросов к таблице" buttons on TablePassport & SearchHit cards
    wrapper.querySelectorAll('[data-open-table-id]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const tId = btn.getAttribute('data-open-table-id');
        if (tId) openTableExplorer({ tableId: tId });
      });
    });
  }

  // ---------------------------------------------------------------------------
  // Interactive Table Explorer Modal (/api/tables, /api/tables/{id}, /api/tables/query)
  // ---------------------------------------------------------------------------
  async function openTableExplorer(opts = {}) {
    const { tableId = null, docId = null } = opts;
    const modal = document.getElementById('table-explorer-modal');
    const selectEl = document.getElementById('texp-table-select');
    const tableContainer = document.getElementById('texp-table-container');
    const showAllBtn = document.getElementById('texp-btn-show-all-docs');

    modal.classList.remove('hidden');
    state.tableExplorer.docIdFilter = docId || null;
    state.tableExplorer.offset = 0;

    if (showAllBtn) {
      if (state.tableExplorer.docIdFilter) showAllBtn.classList.remove('hidden');
      else showAllBtn.classList.add('hidden');
    }

    tableContainer.innerHTML = '<div class="p-6 text-center text-slate-400">Загрузка списка структурированных таблиц из SQLite...</div>';

    try {
      const url = state.tableExplorer.docIdFilter
        ? `/api/tables?doc_id=${encodeURIComponent(state.tableExplorer.docIdFilter)}`
        : '/api/tables';
      const res = await fetch(url);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);

      const tables = Array.isArray(data) ? data : (data.tables || []);
      state.tableExplorer.tables = tables;

      if (tables.length === 0) {
        selectEl.innerHTML = '<option value="">Нет таблиц в SQLite</option>';
        document.getElementById('texp-columns-strip').innerHTML = '<span class="text-xs text-slate-400">Таблицы отсутствуют</span>';
        document.getElementById('texp-passport-badges').innerHTML = '';
        document.getElementById('texp-results-meta').textContent = 'Структурированные таблицы не найдены';
        tableContainer.innerHTML = `
          <div class="p-8 text-center text-slate-500 space-y-2">
            <div class="font-semibold text-sm text-slate-700">В хранилище SQLite (tables_store.db) пока нет таблиц${state.tableExplorer.docIdFilter ? ' для выбранного документа' : ''}.</div>
            <div class="text-xs">Синхронизируйте папку <code>доки/</code> (с флагом force при необходимости), чтобы индексировать все листы Excel и большие таблицы Word/PDF (≥ 25 строк).</div>
          </div>
        `;
        return;
      }

      selectEl.innerHTML = tables
        .map((t) => {
          const lbl = `${t.file_name || t.doc_id}${t.sheet_name ? ' [' + t.sheet_name + ']' : ''} — ${t.table_title || t.section_title || t.table_id} (${t.row_count || 0} стр.)`;
          return `<option value="${escapeHtml(t.table_id)}">${escapeHtml(lbl)}</option>`;
        })
        .join('');

      const targetId =
        tableId && tables.some((t) => t.table_id === tableId)
          ? tableId
          : tables[0].table_id;

      selectEl.value = targetId;
      await selectExplorerTable(targetId);
    } catch (err) {
      tableContainer.innerHTML = `<div class="p-6 text-red-600 font-medium">Ошибка загрузки таблиц: ${escapeHtml(err.message)}</div>`;
    }
  }

  function closeTableExplorer() {
    document.getElementById('table-explorer-modal').classList.add('hidden');
  }

  async function selectExplorerTable(tableId) {
    if (!tableId) return;
    state.tableExplorer.activeTableId = tableId;
    state.tableExplorer.offset = 0;

    // Reset filter inputs on table switch
    document.getElementById('texp-search-text').value = '';
    document.getElementById('texp-filter-val').value = '';

    try {
      const res = await fetch(`/api/tables/${encodeURIComponent(tableId)}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);

      const passport = data.passport || data;
      state.tableExplorer.activePassport = passport;
      state.tableExplorer.formulaSummary = data.formula_summary || null;

      renderExplorerPassportHeader(passport);
      populateExplorerColumnSelectors(passport);
      await runExplorerQuery();
    } catch (err) {
      showToast(`Ошибка загрузки схемы таблицы: ${err.message}`, 'error');
    }
  }

  function renderExplorerPassportHeader(tp) {
    const titleEl = document.getElementById('texp-title');
    const badgesEl = document.getElementById('texp-passport-badges');
    const summaryMetaEl = document.getElementById('texp-schema-summary-meta');
    const stripEl = document.getElementById('texp-columns-strip');

    titleEl.textContent = tp.table_title || tp.sheet_name || tp.section_title || tp.file_name || tp.table_id;

    const badges = [];
    badges.push(`<span class="px-2 py-0.5 rounded bg-emerald-100 text-emerald-900 border border-emerald-300 font-bold">${tp.row_count || 0} строк × ${tp.col_count || (tp.columns || []).length} кол.</span>`);
    if (tp.has_formulas) {
      badges.push(`<span class="px-2 py-0.5 rounded bg-amber-100 text-amber-900 border border-amber-300 font-semibold">🧮 формул: ${tp.formula_count || 'есть'}</span>`);
    }
    if ((tp.also_in_files || []).length > 0) {
      badges.push(`<span class="px-2 py-0.5 rounded bg-indigo-50 text-indigo-800 border border-indigo-200 font-sans" title="Кросс-файловая дедупликация по table_hash">🔗 Также в: ${escapeHtml(tp.also_in_files.join(', '))}</span>`);
    }
    badgesEl.innerHTML = badges.join('');

    const cols = tp.columns || [];
    summaryMetaEl.textContent = `table_id: ${tp.table_id} • колонок: ${cols.length}`;

    if (cols.length === 0) {
      stripEl.innerHTML = '<span class="text-xs text-slate-400">Схема колонок пуста</span>';
      return;
    }

    stripEl.innerHTML = cols
      .map((col) => {
        const typeLabel = col.inferred_type || 'text';
        const samples = (col.sample_values || []).slice(0, 3).join(' | ');
        const tip = `Колонка: ${col.name}\nТип: ${typeLabel}\nЗаполнено: ${col.non_empty_count || 0}\nПримеры: ${samples || '—'}`;
        let cls = 'bg-slate-50 hover:bg-emerald-50 text-slate-800 border-slate-200 hover:border-emerald-400';
        if (col.has_formulas) cls = 'bg-amber-50 hover:bg-amber-100 text-amber-950 border-amber-300';
        else if (col.is_hierarchy) cls = 'bg-indigo-50 hover:bg-indigo-100 text-indigo-950 border-indigo-300';
        else if (typeLabel === 'number' || typeLabel === 'numeric' || typeLabel === 'float') {
          cls = 'bg-emerald-50/70 hover:bg-emerald-100 text-emerald-950 border-emerald-300';
        }

        return `
          <button type="button" data-pick-col="${escapeHtml(col.name)}" title="${escapeHtml(tip)}" class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg border text-[11px] font-medium transition cursor-pointer ${cls}">
            <span class="font-semibold">${escapeHtml(col.name)}</span>
            <span class="font-mono-code text-[10px] opacity-75">(${escapeHtml(typeLabel)}${col.has_formulas ? ', ƒx' : ''}${col.is_hierarchy ? ', иерарх.' : ''})</span>
          </button>
        `;
      })
      .join('');

    stripEl.querySelectorAll('[data-pick-col]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const colName = btn.getAttribute('data-pick-col');
        if (state.tableExplorer.operation === 'aggregate') {
          const grpEl = document.getElementById('texp-group-by');
          const aggColEl = document.getElementById('texp-agg-col');
          if (!grpEl.value) {
            grpEl.value = colName;
            showToast(`Колонка группировки: «${colName}»`, 'info');
          } else {
            aggColEl.value = colName;
            showToast(`Целевая колонка агрегации: «${colName}»`, 'info');
          }
        } else {
          document.getElementById('texp-filter-col').value = colName;
          document.getElementById('texp-filter-val').focus();
        }
      });
    });
  }

  function populateExplorerColumnSelectors(tp) {
    const cols = (tp.columns || []).map((c) => c.name);
    const filterColEl = document.getElementById('texp-filter-col');
    const aggColEl = document.getElementById('texp-agg-col');
    const groupByEl = document.getElementById('texp-group-by');

    const colOptions = cols.map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join('');
    filterColEl.innerHTML = `<option value="">— Любая колонка (без фильтра) —</option>${colOptions}`;
    aggColEl.innerHTML = `<option value="">— Все строки (*) —</option>${colOptions}`;
    groupByEl.innerHTML = `<option value="">— Без группировки —</option>${colOptions}`;
  }

  function setExplorerOperation(op) {
    state.tableExplorer.operation = op;
    state.tableExplorer.offset = 0;

    document.querySelectorAll('.texp-op-btn').forEach((btn) => {
      const active = btn.getAttribute('data-texp-op') === op;
      btn.className = active
        ? 'texp-op-btn px-3 py-1.5 rounded-lg bg-white text-slate-900 shadow-sm transition cursor-pointer'
        : 'texp-op-btn px-3 py-1.5 rounded-lg text-slate-600 hover:text-slate-900 transition cursor-pointer';
    });

    const aggControls = document.getElementById('texp-agg-controls');
    if (op === 'aggregate') {
      aggControls.classList.remove('hidden');
    } else {
      aggControls.classList.add('hidden');
    }
  }

  async function runExplorerQuery() {
    const tableId = state.tableExplorer.activeTableId;
    if (!tableId) return;

    const op = state.tableExplorer.operation;
    const limit = parseInt(document.getElementById('texp-limit').value, 10) || 25;
    state.tableExplorer.limit = limit;

    const searchText = document.getElementById('texp-search-text').value.trim();
    const filterCol = document.getElementById('texp-filter-col').value;
    const filterVal = document.getElementById('texp-filter-val').value.trim();

    const filters = {};
    if (filterCol && filterVal) {
      filters[filterCol] = filterVal;
    }

    const payload = {
      table_id: tableId,
      operation: op,
      filters: filters,
      search_text: searchText,
      limit: limit,
      offset: state.tableExplorer.offset,
    };

    if (op === 'aggregate') {
      payload.agg_func = document.getElementById('texp-agg-func').value || 'count';
      payload.agg_column = document.getElementById('texp-agg-col').value || '';
      payload.group_by = document.getElementById('texp-group-by').value || '';
    }

    const runBtn = document.getElementById('texp-btn-run');
    runBtn.disabled = true;

    try {
      const res = await fetch('/api/tables/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const qData = await res.json();
      if (!res.ok) throw new Error(qData.detail || `HTTP ${res.status}`);

      state.tableExplorer.lastResponse = qData;
      renderExplorerResults(qData);
    } catch (err) {
      showToast(`Ошибка запроса к таблице: ${err.message}`, 'error');
    } finally {
      runBtn.disabled = false;
    }
  }

  function renderExplorerResults(qData) {
    const metaEl = document.getElementById('texp-results-meta');
    const aggBox = document.getElementById('texp-agg-result-box');
    const formulasBox = document.getElementById('texp-formulas-summary-box');
    const tableContainer = document.getElementById('texp-table-container');
    const prevBtn = document.getElementById('texp-btn-prev');
    const nextBtn = document.getElementById('texp-btn-next');
    const pageLbl = document.getElementById('texp-page-label');

    const totalRows = qData.total_table_rows ?? 0;
    const matched = qData.matched_count ?? 0;
    const returned = qData.returned_count ?? (qData.rows || []).length;
    const offset = state.tableExplorer.offset;
    const limit = state.tableExplorer.limit;

    metaEl.innerHTML = `
      Операция: <code class="font-mono-code font-bold text-slate-900">${escapeHtml(qData.operation)}</code> •
      Совпало строк: <strong class="text-emerald-700 font-mono-code">${matched}</strong> из <span class="font-mono-code">${totalRows}</span> •
      Показано: <span class="font-mono-code">${returned}</span> •
      Время SQLite: <span class="font-mono-code">${Number(qData.latency_ms || 0).toFixed(1)} мс</span>
    `;

    pageLbl.textContent = `offset: ${offset}`;
    prevBtn.disabled = offset <= 0;
    nextBtn.disabled = offset + limit >= matched;

    // 1. Aggregation Result Box
    if (qData.operation === 'aggregate' && qData.aggregation_result !== null && qData.aggregation_result !== undefined) {
      aggBox.classList.remove('hidden');
      const agg = qData.aggregation_result;

      if (Array.isArray(agg)) {
        // Group-by array or distinct values array
        if (agg.length > 0 && typeof agg[0] === 'object' && agg[0] !== null) {
          const keys = Object.keys(agg[0]);
          aggBox.innerHTML = `
            <div class="rounded-xl border border-indigo-200 bg-indigo-50/40 p-4 space-y-2">
              <div class="text-xs font-bold text-indigo-950">📊 Результат группировки (${agg.length} групп):</div>
              <div class="ai-md-table-wrap bg-white">
                <table class="ai-md-table">
                  <thead><tr>${keys.map((k) => `<th>${escapeHtml(k)}</th>`).join('')}</tr></thead>
                  <tbody>
                    ${agg
                      .map(
                        (row) =>
                          `<tr>${keys
                            .map((k) => `<td class="font-mono-code">${escapeHtml(typeof row[k] === 'object' ? JSON.stringify(row[k]) : row[k])}</td>`)
                            .join('')}</tr>`
                      )
                      .join('')}
                  </tbody>
                </table>
              </div>
            </div>
          `;
        } else {
          aggBox.innerHTML = `
            <div class="rounded-xl border border-indigo-200 bg-indigo-50/40 p-4 space-y-2">
              <div class="text-xs font-bold text-indigo-950">📊 Уникальные значения (${agg.length}):</div>
              <div class="flex flex-wrap gap-1.5">
                ${agg.map((v) => `<span class="px-2.5 py-1 rounded-lg bg-white border border-indigo-200 text-xs font-medium text-slate-800">${escapeHtml(v)}</span>`).join('')}
              </div>
            </div>
          `;
        }
      } else if (typeof agg === 'object') {
        // Object: check if it contains `groups` array or key-value KPIs
        if (Array.isArray(agg.groups)) {
          const keys = agg.groups.length > 0 ? Object.keys(agg.groups[0]) : [];
          aggBox.innerHTML = `
            <div class="rounded-xl border border-indigo-200 bg-indigo-50/40 p-4 space-y-2">
              <div class="flex items-center justify-between text-xs font-bold text-indigo-950">
                <span>📊 Результат агрегации по группам (${agg.groups.length}):</span>
                ${agg.total !== undefined ? `<span class="font-mono-code">Итого: ${escapeHtml(agg.total)}</span>` : ''}
              </div>
              ${keys.length > 0 ? `
                <div class="ai-md-table-wrap bg-white">
                  <table class="ai-md-table">
                    <thead><tr>${keys.map((k) => `<th>${escapeHtml(k)}</th>`).join('')}</tr></thead>
                    <tbody>
                      ${agg.groups
                        .map(
                          (row) =>
                            `<tr>${keys.map((k) => `<td class="font-mono-code">${escapeHtml(row[k])}</td>`).join('')}</tr>`
                        )
                        .join('')}
                    </tbody>
                  </table>
                </div>
              ` : `<pre class="text-xs font-mono-code bg-white p-3 rounded-lg border border-indigo-100">${escapeHtml(JSON.stringify(agg, null, 2))}</pre>`}
            </div>
          `;
        } else if (Array.isArray(agg.values)) {
          aggBox.innerHTML = `
            <div class="rounded-xl border border-indigo-200 bg-indigo-50/40 p-4 space-y-2">
              <div class="text-xs font-bold text-indigo-950">📊 Уникальных значений: ${escapeHtml(agg.count ?? agg.values.length)}</div>
              <div class="flex flex-wrap gap-1.5 max-h-48 overflow-y-auto">
                ${agg.values.map((v) => `<span class="px-2.5 py-1 rounded-lg bg-white border border-indigo-200 text-xs font-medium text-slate-800">${escapeHtml(v)}</span>`).join('')}
              </div>
            </div>
          `;
        } else {
          const entries = Object.entries(agg);
          aggBox.innerHTML = `
            <div class="rounded-xl border border-indigo-200 bg-indigo-50/40 p-4 space-y-2">
              <div class="text-xs font-bold text-indigo-950">📊 Результат агрегации:</div>
              <div class="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                ${entries
                  .map(
                    ([k, v]) => `
                      <div class="p-3 rounded-xl bg-white border border-indigo-200">
                        <div class="text-[11px] text-slate-500 font-medium truncate" title="${escapeHtml(k)}">${escapeHtml(k)}</div>
                        <div class="text-base font-bold text-indigo-900 font-mono-code mt-0.5 break-all">${escapeHtml(typeof v === 'object' ? JSON.stringify(v) : v)}</div>
                      </div>
                    `
                  )
                  .join('')}
              </div>
            </div>
          `;
        }
      } else {
        aggBox.innerHTML = `
          <div class="rounded-xl border border-indigo-200 bg-indigo-50 p-4 flex items-center justify-between">
            <span class="text-xs font-bold text-indigo-950">📊 Вычисленное значение агрегации:</span>
            <span class="text-xl font-bold font-mono-code text-indigo-700">${escapeHtml(agg)}</span>
          </div>
        `;
      }
    } else {
      aggBox.classList.add('hidden');
      aggBox.innerHTML = '';
    }

    // 2. Formulas Summary Box (for get_formulas)
    const formulas = qData.formulas || [];
    if (qData.operation === 'get_formulas' && formulas.length > 0) {
      formulasBox.classList.remove('hidden');
      formulasBox.innerHTML = `
        <div class="rounded-xl border border-amber-300 bg-amber-50/50 p-4 space-y-2.5">
          <div class="text-xs font-bold text-amber-950">🧮 Извлеченные ячейки с формулами Excel (${formulas.length}):</div>
          <div class="ai-md-table-wrap bg-white">
            <table class="ai-md-table">
              <thead>
                <tr>
                  <th>Строка</th>
                  <th>Колонка / Ячейка</th>
                  <th>Формула</th>
                  <th>Значение / Контекст строки</th>
                </tr>
              </thead>
              <tbody>
                ${formulas
                  .map((f) => {
                    const rIdx = f.row_index ?? f.row ?? '—';
                    const col = f.column || f.col_name || f.cell || '—';
                    const expr = f.formula || f.expression || (f.formulas ? JSON.stringify(f.formulas) : '—');
                    let ctx = f.cell_text || f.value || f.cell_value || f.row_label || f.row_preview || '';
                    if (f.row_label && f.cell_text && !f.cell_text.includes(f.row_label)) {
                      ctx = `${f.row_label} → ${f.cell_text}`;
                    } else if (!ctx && f.context) {
                      ctx = typeof f.context === 'object' ? JSON.stringify(f.context) : String(f.context);
                    }
                    return `
                      <tr>
                        <td class="font-mono-code font-semibold">${escapeHtml(rIdx)}</td>
                        <td class="font-semibold">${escapeHtml(col)}</td>
                        <td><span class="formula-cell-badge">ƒx ${escapeHtml(expr)}</span></td>
                        <td>${formatCellWithFormulaHighlight(ctx)}</td>
                      </tr>
                    `;
                  })
                  .join('')}
              </tbody>
            </table>
          </div>
        </div>
      `;
    } else {
      formulasBox.classList.add('hidden');
      formulasBox.innerHTML = '';
    }

    // 3. Rows Data Table
    const rows = qData.rows || [];
    if (rows.length === 0) {
      if (qData.operation === 'aggregate' && qData.aggregation_result !== null) {
        tableContainer.innerHTML = '';
      } else if (qData.markdown_preview) {
        tableContainer.innerHTML = `<div class="ai-markdown-body">${renderMarkdownSafe(qData.markdown_preview)}</div>`;
      } else {
        tableContainer.innerHTML = '<div class="p-6 text-center text-slate-400 text-xs">По заданным условиям строки не найдены.</div>';
      }
      return;
    }

    const cols =
      qData.columns && qData.columns.length > 0
        ? qData.columns
        : Object.keys(rows[0]).filter((k) => k !== '_row_index');

    const ths = cols.map((c) => `<th class="px-3 py-2.5 text-left text-[11px] font-bold text-slate-700 whitespace-nowrap border-b border-slate-200">${escapeHtml(c)}</th>`).join('');
    const trs = rows
      .map((r, idx) => {
        const rowNum = r._row_index ?? r.row_index ?? offset + idx + 1;
        const tds = cols
          .map((c) => `<td class="px-3 py-2 border-b border-slate-100 text-slate-800 align-top">${formatCellWithFormulaHighlight(r[c])}</td>`)
          .join('');
        return `
          <tr class="hover:bg-emerald-50/40 transition">
            <td class="px-2.5 py-2 border-b border-slate-100 font-mono-code text-[11px] text-slate-400 text-right select-none">${escapeHtml(rowNum)}</td>
            ${tds}
          </tr>
        `;
      })
      .join('');

    tableContainer.innerHTML = `
      <div class="border border-slate-200 rounded-xl overflow-auto max-h-[54vh]">
        <table class="min-w-full text-xs explorer-data-table">
          <thead>
            <tr>
              <th class="px-2.5 py-2.5 text-right text-[11px] font-mono-code text-slate-400 border-b border-slate-200 w-12">#</th>
              ${ths}
            </tr>
          </thead>
          <tbody>
            ${trs}
          </tbody>
        </table>
      </div>
    `;
  }

  // ---------------------------------------------------------------------------
  // Event Listeners & Initialization
  // ---------------------------------------------------------------------------
  function initApp() {
    // Tab navigation
    document.querySelectorAll('.tab-nav-btn').forEach((btn) => {
      btn.addEventListener('click', () => switchTab(btn.getAttribute('data-tab')));
    });

    // Playground mode toggle buttons
    const btnModeSearch = document.getElementById('btn-mode-search');
    const btnModeAsk = document.getElementById('btn-mode-ask');
    if (btnModeSearch && btnModeAsk) {
      btnModeSearch.addEventListener('click', () => setPlaygroundMode('search'));
      btnModeAsk.addEventListener('click', () => setPlaygroundMode('ask'));
      setPlaygroundMode(state.playgroundMode);
    }

    // Header & Registry & Playground Table Explorer shortcuts
    const btnHeaderTables = document.getElementById('btn-header-open-tables');
    if (btnHeaderTables) {
      btnHeaderTables.addEventListener('click', () => openTableExplorer());
    }
    const btnRegTables = document.getElementById('btn-registry-open-tables');
    if (btnRegTables) {
      btnRegTables.addEventListener('click', () => openTableExplorer());
    }
    const btnPlayTables = document.getElementById('btn-playground-open-explorer');
    if (btnPlayTables) {
      btnPlayTables.addEventListener('click', () => openTableExplorer());
    }

    // Dropzone & file input
    const dropzone = document.getElementById('upload-dropzone');
    const fileInput = document.getElementById('upload-file-input');

    dropzone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
      addFilesToSelection(e.target.files);
      fileInput.value = '';
    });

    ['dragenter', 'dragover'].forEach((evtName) => {
      dropzone.addEventListener(evtName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add('dropzone-active');
      });
    });

    ['dragleave', 'drop'].forEach((evtName) => {
      dropzone.addEventListener(evtName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove('dropzone-active');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      if (e.dataTransfer && e.dataTransfer.files) {
        addFilesToSelection(e.dataTransfer.files);
      }
    });

    document.getElementById('btn-clear-files').addEventListener('click', () => {
      state.selectedFiles = [];
      renderSelectedFiles();
    });

    document.getElementById('btn-upload-submit').addEventListener('click', handleUploadSubmit);
    document.getElementById('btn-sync-folder').addEventListener('click', handleSyncFolder);
    document.getElementById('btn-registry-sync-shortcut').addEventListener('click', handleSyncFolder);

    document.getElementById('btn-clear-log').addEventListener('click', () => {
      document.getElementById('prog-event-log').innerHTML = '';
    });

    // Registry filters
    document.getElementById('btn-refresh-registry').addEventListener('click', loadDocumentsRegistry);
    document.getElementById('reg-filter-search').addEventListener('input', renderRegistryTable);
    document.getElementById('reg-filter-status').addEventListener('change', renderRegistryTable);
    document.getElementById('reg-filter-format').addEventListener('change', renderRegistryTable);

    // Inspector modal controls
    document.getElementById('btn-close-inspector').addEventListener('click', () => {
      document.getElementById('inspector-modal').classList.add('hidden');
    });
    document.getElementById('btn-insp-show-all').addEventListener('click', () => {
      state.inspector.selectedUnitId = null;
      state.inspector.selectedSectionId = null;
      renderInspectorTree();
      renderInspectorChunks();
    });
    document.querySelectorAll('.insp-mode-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        state.inspector.viewMode = btn.getAttribute('data-view-mode');
        document.querySelectorAll('.insp-mode-btn').forEach((b) => {
          const active = b === btn;
          b.className = active
            ? 'insp-mode-btn px-3 py-1.5 rounded-lg bg-white text-slate-900 shadow-sm transition'
            : 'insp-mode-btn px-3 py-1.5 rounded-lg text-slate-600 hover:text-slate-900 transition';
        });
        renderInspectorChunks();
      });
    });

    // Metadata modal controls
    document.getElementById('btn-close-meta-modal').addEventListener('click', closeMetadataModal);
    document.getElementById('btn-cancel-meta-modal').addEventListener('click', closeMetadataModal);
    document.getElementById('metadata-edit-form').addEventListener('submit', submitMetadataPatch);

    // Table Explorer modal controls
    document.getElementById('btn-close-table-explorer').addEventListener('click', closeTableExplorer);
    document.getElementById('texp-table-select').addEventListener('change', (e) => {
      selectExplorerTable(e.target.value);
    });
    document.getElementById('texp-btn-show-all-docs').addEventListener('click', () => {
      openTableExplorer({ tableId: state.tableExplorer.activeTableId, docId: null });
    });
    document.querySelectorAll('.texp-op-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        setExplorerOperation(btn.getAttribute('data-texp-op'));
        runExplorerQuery();
      });
    });
    document.getElementById('texp-btn-run').addEventListener('click', () => {
      state.tableExplorer.offset = 0;
      runExplorerQuery();
    });
    document.getElementById('texp-btn-reset').addEventListener('click', () => {
      document.getElementById('texp-search-text').value = '';
      document.getElementById('texp-filter-col').value = '';
      document.getElementById('texp-filter-val').value = '';
      document.getElementById('texp-agg-func').value = 'count';
      document.getElementById('texp-agg-col').value = '';
      document.getElementById('texp-group-by').value = '';
      state.tableExplorer.offset = 0;
      runExplorerQuery();
    });
    ['texp-search-text', 'texp-filter-val'].forEach((inpId) => {
      document.getElementById(inpId).addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          state.tableExplorer.offset = 0;
          runExplorerQuery();
        }
      });
    });
    document.getElementById('texp-limit').addEventListener('change', () => {
      state.tableExplorer.offset = 0;
      runExplorerQuery();
    });
    document.getElementById('texp-btn-prev').addEventListener('click', () => {
      state.tableExplorer.offset = Math.max(0, state.tableExplorer.offset - state.tableExplorer.limit);
      runExplorerQuery();
    });
    document.getElementById('texp-btn-next').addEventListener('click', () => {
      state.tableExplorer.offset += state.tableExplorer.limit;
      runExplorerQuery();
    });

    // Close modals on Escape key
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        document.getElementById('inspector-modal').classList.add('hidden');
        closeMetadataModal();
        closeTableExplorer();
      }
    });

    // Search & AI Agent Playground controls
    document.getElementById('search-form').addEventListener('submit', (e) => {
      e.preventDefault();
      if (state.playgroundMode === 'ask') {
        executeAgentAsk();
      } else {
        executeHybridSearch();
      }
    });
    document.getElementById('btn-run-search').addEventListener('click', (e) => {
      e.preventDefault();
      executeHybridSearch();
    });
    document.getElementById('btn-run-ask').addEventListener('click', (e) => {
      e.preventDefault();
      executeAgentAsk();
    });
    document.getElementById('btn-clear-query').addEventListener('click', () => {
      const inp = document.getElementById('search-query-input');
      inp.value = '';
      inp.focus();
    });

    // Initial data bootstrap
    refreshHealth();
    loadDocumentsRegistry();
    loadSearchPresets();
    setInterval(refreshHealth, 25000);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
  } else {
    initApp();
  }
})();

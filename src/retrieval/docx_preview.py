"""High-fidelity A4 visual document renderer for `.docx` and `.doc` files.

Converts Word documents (`.docx` OOXML and `.doc` OLE2/RTF/HTML/XML) into cached,
print-grade paginated A4 PDFs (`data/preview_cache/{sha256}.pdf`) and renders crisp
page PNGs via PyMuPDF (`fitz.Story` + `fitz.Archive` with Cyrillic Liberation/Noto fonts,
or headless LibreOffice when available).
"""

from __future__ import annotations

import hashlib
import html
import io
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
import zipfile
from pathlib import Path
from typing import Any

import docx
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph
import pymupdf as fitz

from src.config import DATA_DIR
from src.ingestion.parsers import WordNumberingResolver, _W_NS_MAP, parse_document
from src.schemas import BlockType, ParsedBlock

logger = logging.getLogger(__name__)

PREVIEW_CACHE_DIR: Path = DATA_DIR / "preview_cache"
PREVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_R_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_A_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"

# Standard A4 dimensions in PDF points (1 pt = 1/72 inch)
A4_PORTRAIT_W: float = 595.28
A4_PORTRAIT_H: float = 841.89
A4_LANDSCAPE_W: float = 841.89
A4_LANDSCAPE_H: float = 595.28

# Cap vertical rowspan in huge tables so MuPDF Story paginates 500+ row tables cleanly across pages
_MAX_HTML_ROWSPAN: int = 16

_RENDER_LOCK = threading.Lock()

_VISIBLE_BORDER_VALS: frozenset[str] = frozenset(
    {
        "single",
        "double",
        "thick",
        "dotted",
        "dashed",
        "dotdash",
        "dotdotdash",
        "triple",
        "thinThickSmallGap",
        "thickThinSmallGap",
        "outset",
        "inset",
        "threeDEngrave",
        "threeDEmboss",
        "wave",
    }
)
_NONE_BORDER_VALS: frozenset[str] = frozenset({"none", "nil"})

_REQUISITE_PAIR_RE = re.compile(
    r"(?:от\s+\d{1,2}[.\s/]\d{1,2}[.\s/]\d{2,4}|(?:№|N)\s*[0-9А-Яа-яA-Za-z/-]+)",
    re.IGNORECASE,
)
_EDO_STAMP_RE = re.compile(
    r"(?:зарегистрирован[оа]?\s+(?:в\s+(?:эдо|сэд|реестре|журнале)|от\b)|"
    r"регистрационн[а-яё]+\s*(?:№|n|номер)|рег\.\s*(?:№|n)\s*\d|документ\s+подписан\s+электронной\s+подписью)",
    re.IGNORECASE,
)
_CENTER_DOC_KIND_RE = re.compile(
    r"^\s*(?:ПРИКАЗ|РАСПОРЯЖЕНИЕ|ПОЛОЖЕНИЕ|РЕГЛАМЕНТ|ПОРЯДОК|ИНСТРУКЦИЯ|"
    r"ЛИСТ\s+СОГЛАСОВАНИЯ|ПРИКАЗЫВАЮ\s*:?|ОБЯЗЫВАЮ\s*:?|СОГЛАСИЕ)\s*$",
    re.IGNORECASE,
)
_MINISTRY_HEADER_RE = re.compile(
    r"^\s*(?:Министерство\s+науки\s+и\s+высшего\s+образования|"
    r"федеральное\s+государственное\s+автономное\s+образовательное\s+учреждение|"
    r"высшего\s+образования\s*$|"
    r"[«\"]?Московский\s+физико-технический\s+институт|"
    r"\(национальный\s+исследовательский\s+университет\))",
    re.IGNORECASE,
)
_CORNER_STAMP_START_RE = re.compile(
    r"^\s*(?:УТВЕРЖДЕН[ОЫА]?|УТВЕРЖДАЮ|Приложение\s*(?:№|N)?\s*\d*[а-яa-z]?\s+к\s+)",
    re.IGNORECASE,
)
_SIGNATURE_ROLE_RE = re.compile(
    r"^\s*(?:И\.?\s*о\.?\s+ректора|Ио\s+ректора|Ректор|Первый\s+проректор|"
    r"Проректор(?:\s+по\s+[а-яё\s]+)?|Директор(?:\s+по\s+[а-яё\s]+)?)\s*$",
    re.IGNORECASE,
)

_FONT_DIRS: tuple[Path, ...] = (
    Path("/usr/share/fonts/liberation"),
    Path("/usr/share/fonts/TTF"),
    Path("/usr/share/fonts/noto"),
    Path("/usr/share/fonts/dejavu"),
    Path("/usr/share/fonts/truetype/liberation"),
    Path("/usr/share/fonts/truetype/dejavu"),
)


def compute_bytes_sha256(data: bytes) -> str:
    """Compute SHA-256 hex digest of in-memory bytes."""
    return hashlib.sha256(data).hexdigest()


def compute_path_sha256(file_path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file on disk."""
    h = hashlib.sha256()
    with Path(file_path).open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_preview_cache_path(sha256: str, cache_dir: Path | None = None) -> Path:
    """Return canonical path `data/preview_cache/{sha256}.pdf`."""
    base = cache_dir if cache_dir is not None else PREVIEW_CACHE_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{sha256}.pdf"


def _discover_fonts() -> tuple[list[str], dict[str, str], Path | None]:
    """
    Locate Cyrillic-capable Serif and Sans TTF font families on the host system.
    Returns `(font_dirs, font_files_map, regular_serif_ttf_path)`.
    """
    existing_dirs = [str(d) for d in _FONT_DIRS if d.is_dir()]

    candidates: dict[str, tuple[str, ...]] = {
        "serif_reg": ("LiberationSerif-Regular.ttf", "DejaVuSerif.ttf", "NotoSerif-Regular.ttf"),
        "serif_bold": ("LiberationSerif-Bold.ttf", "DejaVuSerif-Bold.ttf", "NotoSerif-Bold.ttf"),
        "serif_italic": ("LiberationSerif-Italic.ttf", "DejaVuSerif-Italic.ttf", "NotoSerif-Italic.ttf"),
        "serif_bi": ("LiberationSerif-BoldItalic.ttf", "DejaVuSerif-BoldItalic.ttf", "NotoSerif-BoldItalic.ttf"),
        "sans_reg": ("LiberationSans-Regular.ttf", "DejaVuSans.ttf", "NotoSans-Regular.ttf"),
        "sans_bold": ("LiberationSans-Bold.ttf", "DejaVuSans-Bold.ttf", "NotoSans-Bold.ttf"),
        "sans_italic": ("LiberationSans-Italic.ttf", "DejaVuSans-Oblique.ttf", "NotoSans-Italic.ttf"),
        "sans_bi": ("LiberationSans-BoldItalic.ttf", "DejaVuSans-BoldOblique.ttf", "NotoSans-BoldItalic.ttf"),
    }

    resolved: dict[str, str] = {}
    reg_ttf_path: Path | None = None

    for key, names in candidates.items():
        for d_str in existing_dirs:
            d_path = Path(d_str)
            found = next((nm for nm in names if (d_path / nm).is_file()), None)
            if found:
                resolved[key] = found
                if key == "serif_reg" and reg_ttf_path is None:
                    reg_ttf_path = d_path / found
                break

    if reg_ttf_path is None:
        for d_str in existing_dirs:
            d_path = Path(d_str)
            any_ttf = next(d_path.glob("*.ttf"), None)
            if any_ttf is not None:
                reg_ttf_path = any_ttf
                break

    return existing_dirs, resolved, reg_ttf_path


def _build_print_css(font_map: dict[str, str], is_landscape: bool = False) -> str:
    """Build print-grade A4 CSS stylesheet with embedded Cyrillic `@font-face` declarations."""
    face_rules: list[str] = []
    if "serif_reg" in font_map:
        face_rules.append(f'@font-face {{ font-family: "DocSerif"; src: url("{font_map["serif_reg"]}"); }}')
    if "serif_bold" in font_map:
        face_rules.append(
            f'@font-face {{ font-family: "DocSerif"; src: url("{font_map["serif_bold"]}"); font-weight: bold; }}'
        )
    if "serif_italic" in font_map:
        face_rules.append(
            f'@font-face {{ font-family: "DocSerif"; src: url("{font_map["serif_italic"]}"); font-style: italic; }}'
        )
    if "serif_bi" in font_map:
        face_rules.append(
            f'@font-face {{ font-family: "DocSerif"; src: url("{font_map["serif_bi"]}"); font-weight: bold; font-style: italic; }}'
        )

    if "sans_reg" in font_map:
        face_rules.append(f'@font-face {{ font-family: "DocSans"; src: url("{font_map["sans_reg"]}"); }}')
    if "sans_bold" in font_map:
        face_rules.append(
            f'@font-face {{ font-family: "DocSans"; src: url("{font_map["sans_bold"]}"); font-weight: bold; }}'
        )
    if "sans_italic" in font_map:
        face_rules.append(
            f'@font-face {{ font-family: "DocSans"; src: url("{font_map["sans_italic"]}"); font-style: italic; }}'
        )
    if "sans_bi" in font_map:
        face_rules.append(
            f'@font-face {{ font-family: "DocSans"; src: url("{font_map["sans_bi"]}"); font-weight: bold; font-style: italic; }}'
        )

    base_font = '"DocSerif", serif' if "serif_reg" in font_map else '"DocSans", sans-serif'
    sans_font = '"DocSans", sans-serif' if "sans_reg" in font_map else base_font
    tbl_font_pt = "7.6pt" if is_landscape else "8.8pt"
    tbl_pad = "2pt 3pt" if is_landscape else "3pt 4.5pt"

    return "\n".join(
        [
            *face_rules,
            f"""
body {{
    font-family: {base_font};
    font-size: 11pt;
    line-height: 1.22;
    color: #111418;
    margin: 0;
    padding: 0;
}}
p {{
    margin-top: 0pt;
    margin-bottom: 3.5pt;
    line-height: 1.22;
}}
h1, h2, h3, h4 {{
    font-family: {base_font};
    color: #0d1015;
    margin-top: 7pt;
    margin-bottom: 4pt;
    line-height: 1.2;
}}
h1 {{ font-size: 13.5pt; font-weight: bold; text-align: center; }}
h2 {{ font-size: 12pt; font-weight: bold; }}
h3 {{ font-size: 11pt; font-weight: bold; }}
div.doc-running-header {{
    border-bottom: 0.5pt solid #8a94a6;
    padding-bottom: 4pt;
    margin-bottom: 8pt;
    font-size: 9.5pt;
    color: #2c3440;
}}
div.edo-stamp {{
    border: 0.9pt solid #1e4db7;
    background-color: #f5f8ff;
    color: #163b8c;
    font-family: {sans_font};
    font-size: 8.5pt;
    line-height: 1.25;
    padding: 4.5pt 7pt;
    margin: 6pt 0;
}}
table.req-table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 3pt;
    margin-bottom: 7pt;
}}
table.req-table td {{
    border: none;
    padding: 2pt 1pt 4pt 1pt;
    font-size: 11pt;
    vertical-align: bottom;
}}
table.sig-table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 10pt;
    margin-bottom: 6pt;
}}
table.sig-table td {{
    border: none;
    padding: 3pt 2pt;
    font-size: 11pt;
    vertical-align: bottom;
}}
table.layout-table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 3pt;
    margin-bottom: 4pt;
}}
table.layout-table td {{
    border: none;
    padding: 1.5pt 2pt;
    vertical-align: top;
}}
table.data-table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 5pt;
    margin-bottom: 7pt;
    font-size: {tbl_font_pt};
    line-height: 1.16;
}}
table.data-table th {{
    border: 0.6pt solid #1f242d;
    background-color: #f1f4f8;
    padding: {tbl_pad};
    font-weight: bold;
    vertical-align: middle;
    color: #0d1015;
}}
table.data-table td {{
    border: 0.6pt solid #262c36;
    padding: {tbl_pad};
    vertical-align: top;
    color: #111418;
}}
div.page-break {{
    page-break-before: always;
}}
""",
        ]
    )


def _escape_html_preserving_breaks(text: str) -> str:
    """HTML-escape text while converting internal newlines to `<br/>` and keeping non-breaking spaces."""
    if not text:
        return ""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    # Clean Private Use Area Wingdings/Symbol bullets to standard bullet
    for pua in ("\uf02d", "\uf0b7", "\uf0a7", "\uf076", "\uf0d8", "\uf0fc", "\uf06e", "\uf0a8"):
        cleaned = cleaned.replace(pua, "•")
    escaped = html.escape(cleaned, quote=True)
    return escaped.replace("\n", "<br/>")


def _twips_to_pt(val_str: str | None, default: float = 0.0) -> float:
    if not val_str:
        return default
    try:
        return float(int(val_str)) / 20.0
    except (TypeError, ValueError):
        return default


def _sanitize_hex_color(raw_hex: str | None, is_background: bool = False) -> str | None:
    """Validate 6-digit OOXML RGB hex color; prevent invisible white-on-white ink."""
    if not raw_hex:
        return None
    h = raw_hex.strip().lstrip("#").upper()
    if h in {"AUTO", "NONE", "NIL", ""} or not re.fullmatch(r"[0-9A-F]{6}", h):
        return None
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    if is_background:
        if r >= 252 and g >= 252 and b >= 252:
            return None
        return f"#{h}"
    # Text ink: darken overly bright colors so text never disappears on white A4 paper
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    if luminance > 195:
        return "#22262c"
    return f"#{h}"


def _resolve_style_alignment(para: DocxParagraph) -> str | None:
    """Walk paragraph and its style chain to resolve explicit alignment (`left`, `center`, `right`, `justify`)."""
    p_el = para._p
    jc_vals = p_el.xpath("./w:pPr/w:jc/@w:val")
    if jc_vals:
        raw_jc = str(jc_vals[0]).lower()
    else:
        raw_jc = ""
        st = getattr(para, "style", None)
        visited = 0
        while st is not None and visited < 6:
            visited += 1
            st_el = getattr(st, "_element", None)
            if st_el is not None:
                st_jc = st_el.xpath("./w:pPr/w:jc/@w:val")
                if st_jc:
                    raw_jc = str(st_jc[0]).lower()
                    break
            st = getattr(st, "base_style", None)

    if raw_jc in {"center"}:
        return "center"
    if raw_jc in {"right", "end"}:
        return "right"
    if raw_jc in {"both", "distribute", "thaiDistribute", "justify"}:
        return "justify"
    if raw_jc in {"left", "start"}:
        return "left"
    return None


def _resolve_style_bold_and_size(para: DocxParagraph) -> tuple[bool, float | None]:
    """Check paragraph style chain for default bold and font size (in pt)."""
    is_bold = False
    size_pt: float | None = None
    st = getattr(para, "style", None)
    visited = 0
    while st is not None and visited < 6:
        visited += 1
        st_name = (getattr(st, "name", "") or "").lower()
        if st_name.startswith("heading") or "заголовок" in st_name or st_name == "title":
            is_bold = True
        font = getattr(st, "font", None)
        if font is not None:
            if font.bold is True:
                is_bold = True
            if size_pt is None and font.size is not None:
                try:
                    size_pt = float(font.size.pt)
                except Exception:
                    pass
        st = getattr(st, "base_style", None)
    return is_bold, size_pt


def _render_docx_paragraph_html(
    para: DocxParagraph,
    doc: docx.document.Document,
    num_resolver: WordNumberingResolver | None,
    extra_images: dict[str, bytes],
    in_table_cell: bool = False,
) -> tuple[str, bool]:
    """
    Render a `DocxParagraph` to rich HTML preserving runs (`<b>`, `<i>`, `<u>`, `<sup>`, `<sub>`,
    font sizes, ink colors), inline images, XML list numbering, alignment, indents, and page breaks.
    Returns `(html_fragment, has_page_break_after)`.
    """
    p_el = para._p
    page_break_before = bool(p_el.xpath("./w:pPr/w:pageBreakBefore"))
    explicit_br_count = len(p_el.xpath(".//w:br[@w:type='page']"))

    xml_prefix = ""
    if num_resolver is not None and not in_table_cell:
        try:
            xml_prefix = num_resolver.resolve_paragraph_prefix(p_el) or ""
        except Exception:
            xml_prefix = ""

    style_bold, style_size_pt = _resolve_style_bold_and_size(para)

    # Extract inline runs & hyperlinks in exact XML child order
    run_fragments: list[str] = []

    def _process_run_element(r_el: Any) -> None:
        # Check embedded inline drawing blips (`a:blip/@r:embed`)
        for blip in r_el.iter(f"{_A_NS}blip"):
            r_id = blip.get(f"{_R_NS}embed")
            if r_id and r_id in doc.part.rels:
                try:
                    target_part = doc.part.rels[r_id].target_part
                    blob = getattr(target_part, "blob", None)
                    if isinstance(blob, bytes) and blob:
                        ext = Path(str(getattr(target_part, "partname", "img.png"))).suffix.lower() or ".png"
                        img_key = f"inline_{hashlib.sha1(blob).hexdigest()[:12]}{ext}"
                        extra_images[img_key] = blob
                        run_fragments.append(
                            f'<img src="{img_key}" style="max-width: 100%; height: auto; margin: 2pt 0;"/>'
                        )
                except Exception as img_exc:
                    logger.debug("Skipping inline DOCX image %s: %s", r_id, img_exc)

        # Build run text (handling `<w:t>`, `<w:tab/>`, `<w:br/>`)
        pieces: list[str] = []
        for sub in r_el:
            stag = sub.tag.rsplit("}", 1)[-1] if "}" in sub.tag else sub.tag
            if stag == "t" and sub.text:
                pieces.append(sub.text)
            elif stag == "tab":
                pieces.append("\u00a0\u00a0\u00a0\u00a0")
            elif stag == "br":
                br_t = sub.get(f"{_W_NS}type", "")
                if br_t != "page":
                    pieces.append("\n")
            elif stag == "noBreakHyphen":
                pieces.append("-")

        raw_txt = "".join(pieces)
        if not raw_txt:
            return

        escaped_txt = _escape_html_preserving_breaks(raw_txt)

        # Run formatting (`w:rPr`)
        rpr_list = r_el.xpath("./w:rPr")
        b_on = style_bold
        i_on = False
        u_on = False
        strike_on = False
        vert_align = ""
        r_size_pt: float | None = style_size_pt
        r_color: str | None = None

        if rpr_list:
            rpr = rpr_list[0]
            b_els = rpr.xpath("./w:b")
            if b_els:
                b_val = b_els[0].get(f"{_W_NS}val", "true").lower()
                b_on = b_val not in {"0", "false", "off"}
            i_els = rpr.xpath("./w:i")
            if i_els:
                i_val = i_els[0].get(f"{_W_NS}val", "true").lower()
                i_on = i_val not in {"0", "false", "off"}
            u_els = rpr.xpath("./w:u")
            if u_els:
                u_val = u_els[0].get(f"{_W_NS}val", "single").lower()
                u_on = u_val not in {"none", "0", "false", "off"}
            s_els = rpr.xpath("./w:strike")
            if s_els:
                s_val = s_els[0].get(f"{_W_NS}val", "true").lower()
                strike_on = s_val not in {"0", "false", "off"}
            va_els = rpr.xpath("./w:vertAlign/@w:val")
            if va_els:
                vert_align = str(va_els[0]).lower()
            sz_els = rpr.xpath("./w:sz/@w:val")
            if sz_els:
                try:
                    # Word `w:sz` is in half-points!
                    r_size_pt = float(int(sz_els[0])) / 2.0
                except ValueError:
                    pass
            col_els = rpr.xpath("./w:color/@w:val")
            if col_els:
                r_color = _sanitize_hex_color(str(col_els[0]), is_background=False)

        span_styles: list[str] = []
        if not in_table_cell and r_size_pt is not None and abs(r_size_pt - 11.0) >= 0.8:
            clamped_sz = max(7.5, min(18.0, r_size_pt))
            span_styles.append(f"font-size: {clamped_sz:.1f}pt")
        if r_color and r_color not in {"#000000", "#111111", "#111418"}:
            span_styles.append(f"color: {r_color}")

        frag = escaped_txt
        if vert_align == "superscript":
            frag = f"<sup>{frag}</sup>"
        elif vert_align == "subscript":
            frag = f"<sub>{frag}</sub>"
        if strike_on:
            frag = f"<s>{frag}</s>"
        if u_on:
            frag = f"<u>{frag}</u>"
        if i_on:
            frag = f"<i>{frag}</i>"
        if b_on:
            frag = f"<b>{frag}</b>"
        if span_styles:
            frag = f'<span style="{"; ".join(span_styles)}">{frag}</span>'

        run_fragments.append(frag)

    for child in p_el:
        ctag = child.tag.rsplit("}", 1)[-1] if "}" in child.tag else child.tag
        if ctag == "r":
            _process_run_element(child)
        elif ctag == "hyperlink":
            for sub_r in child.xpath(".//w:r"):
                _process_run_element(sub_r)
        elif ctag in {"ins", "smartTag", "sdt"}:
            for sub_r in child.xpath(".//w:r"):
                _process_run_element(sub_r)

    plain_text = (para.text or "").strip()
    inner_html = "".join(run_fragments).strip()

    if not inner_html and not plain_text:
        if page_break_before or explicit_br_count > 0:
            return '<div class="page-break"></div>', False
        return "", False

    if not inner_html and plain_text:
        inner_html = _escape_html_preserving_breaks(plain_text)
        if style_bold:
            inner_html = f"<b>{inner_html}</b>"

    if xml_prefix and not plain_text.lstrip().startswith(xml_prefix):
        inner_html = f"<b>{html.escape(xml_prefix)}</b>\u00a0{inner_html}"

    # EDO registration stamp block
    if not in_table_cell and _EDO_STAMP_RE.search(plain_text) and len(plain_text) <= 260:
        prefix_br = '<div class="page-break"></div>' if page_break_before else ""
        suffix_br = explicit_br_count > 0
        return f'{prefix_br}<div class="edo-stamp">{inner_html}</div>', suffix_br

    # Paragraph CSS styles (alignment, indents, spacing, shading)
    align = _resolve_style_alignment(para)
    if align is None and not in_table_cell:
        if _CENTER_DOC_KIND_RE.match(plain_text) or _MINISTRY_HEADER_RE.match(plain_text):
            align = "center"
        elif len(plain_text) >= 85:
            align = "justify"

    p_styles: list[str] = []
    if align:
        p_styles.append(f"text-align: {align}")

    if not in_table_cell:
        ind_left = p_el.xpath("./w:pPr/w:ind/@w:left") or p_el.xpath("./w:pPr/w:ind/@w:start")
        ind_right = p_el.xpath("./w:pPr/w:ind/@w:right") or p_el.xpath("./w:pPr/w:ind/@w:end")
        ind_first = p_el.xpath("./w:pPr/w:ind/@w:firstLine")
        ind_hang = p_el.xpath("./w:pPr/w:ind/@w:hanging")

        left_pt = _twips_to_pt(ind_left[0] if ind_left else None, 0.0)
        right_pt = _twips_to_pt(ind_right[0] if ind_right else None, 0.0)
        first_pt = _twips_to_pt(ind_first[0] if ind_first else None, 0.0)
        hang_pt = _twips_to_pt(ind_hang[0] if ind_hang else None, 0.0)

        if left_pt > 1.0:
            p_styles.append(f"margin-left: {min(260.0, left_pt):.1f}pt")
        elif xml_prefix:
            p_styles.append("margin-left: 16pt")

        if right_pt > 1.0:
            p_styles.append(f"margin-right: {min(160.0, right_pt):.1f}pt")

        if first_pt > 1.0 and align != "center":
            p_styles.append(f"text-indent: {min(42.0, first_pt):.1f}pt")
        elif hang_pt > 1.0 and left_pt >= hang_pt:
            p_styles.append(f"text-indent: -{min(32.0, hang_pt):.1f}pt")

        sp_before = p_el.xpath("./w:pPr/w:spacing/@w:before")
        sp_after = p_el.xpath("./w:pPr/w:spacing/@w:after")
        if sp_before:
            bef_pt = min(16.0, max(0.0, _twips_to_pt(sp_before[0], 0.0)))
            if bef_pt >= 1.5:
                p_styles.append(f"margin-top: {bef_pt:.1f}pt")
        elif _CENTER_DOC_KIND_RE.match(plain_text):
            p_styles.append("margin-top: 8pt")

        if sp_after:
            aft_pt = min(14.0, max(0.0, _twips_to_pt(sp_after[0], 3.5)))
            p_styles.append(f"margin-bottom: {aft_pt:.1f}pt")
    else:
        p_styles.append("margin-top: 0pt")
        p_styles.append("margin-bottom: 1.5pt")

    shd_fill = p_el.xpath("./w:pPr/w:shd/@w:fill")
    bg_hex = _sanitize_hex_color(shd_fill[0] if shd_fill else None, is_background=True)
    if bg_hex:
        p_styles.append(f"background-color: {bg_hex}")
        p_styles.append("padding: 1.5pt 3pt")

    style_attr = f' style="{"; ".join(p_styles)}"' if p_styles else ""
    prefix_br = '<div class="page-break"></div>' if page_break_before else ""
    return f"{prefix_br}<p{style_attr}>{inner_html}</p>", (explicit_br_count > 0)


def _classify_docx_table_kind(
    tbl_el: Any,
    row_count: int,
    col_count: int,
    first_row_texts: list[str],
) -> str:
    """
    Classify an OOXML table into:
    - `'req'`: 1-row 2-col official order requisite header (`от 25.11.2021` | `№ 3065-1`)
    - `'sig'`: 1-row 2-col official signature block (`Ректор` | `Д.В. Орлов`)
    - `'layout'`: borderless layout/corner-stamp frame (`1x1`/`1x2` without borders or explicit `borders=none`)
    - `'data'`: bordered normative/financial/approval table (`0.6pt solid` grid)
    """
    tbl_border_vals = [
        str(b.get(f"{_W_NS}val") or "").lower()
        for b in tbl_el.xpath("./w:tblPr/w:tblBorders/*")
    ]
    tc_border_vals = [
        str(b.get(f"{_W_NS}val") or "").lower()
        for b in tbl_el.xpath("./w:tr/w:tc/w:tcPr/w:tcBorders/*")[:48]
    ]

    has_visible_tbl_border = any(v in _VISIBLE_BORDER_VALS for v in tbl_border_vals)
    has_explicit_none_tbl_border = bool(tbl_border_vals) and all(v in _NONE_BORDER_VALS for v in tbl_border_vals)
    has_visible_tc_border = any(v in _VISIBLE_BORDER_VALS for v in tc_border_vals)

    non_empty_r0 = [t.strip() for t in first_row_texts if t.strip()]
    joined_r0 = " | ".join(non_empty_r0)

    if row_count == 1 and not (has_visible_tbl_border or has_visible_tc_border):
        if len(non_empty_r0) == 2 and _REQUISITE_PAIR_RE.search(joined_r0):
            return "req"
        if len(non_empty_r0) == 2 and _SIGNATURE_ROLE_RE.match(non_empty_r0[0]):
            return "sig"
        if col_count <= 3:
            return "layout"

    if has_explicit_none_tbl_border and not has_visible_tc_border:
        return "layout"

    return "data"


def _render_docx_table_html(
    table: DocxTable,
    doc: docx.document.Document,
    extra_images: dict[str, bytes],
) -> str:
    """
    Render a `DocxTable` directly from OOXML (`<w:tblGrid>`, `<w:tr>`, `<w:tc>`, `gridSpan`, `vMerge`, `w:shd`)
    into a clean layout or bordered data HTML `<table>`.
    """
    tbl_el = table._tbl
    tr_elements = tbl_el.xpath("./w:tr")
    if not tr_elements:
        return ""

    # Determine logical column widths from `<w:tblGrid><w:gridCol w:w="..."/>`
    grid_col_widths = [
        max(120.0, float(int(w)))
        for w in tbl_el.xpath("./w:tblGrid/w:gridCol/@w:w")
        if str(w).lstrip("-").isdigit()
    ]

    # Parse raw OOXML cell matrix respecting `gridSpan` and `vMerge`
    parsed_rows: list[list[dict[str, Any]]] = []
    max_cols = len(grid_col_widths)

    for tr_el in tr_elements:
        row_cells: list[dict[str, Any]] = []
        is_header_tr = bool(tr_el.xpath("./w:trPr/w:tblHeader"))
        col_cursor = 0

        for tc_el in tr_el.xpath("./w:tc"):
            gs_vals = tc_el.xpath("./w:tcPr/w:gridSpan/@w:val")
            try:
                colspan = max(1, int(gs_vals[0])) if gs_vals else 1
            except ValueError:
                colspan = 1

            vmerge_els = tc_el.xpath("./w:tcPr/w:vMerge")
            vmerge_mode: str | None = None
            if vmerge_els:
                vm_val = (vmerge_els[0].get(f"{_W_NS}val") or "continue").lower()
                vmerge_mode = "restart" if vm_val == "restart" else "continue"

            shd_vals = tc_el.xpath("./w:tcPr/w:shd/@w:fill")
            bg_hex = _sanitize_hex_color(shd_vals[0] if shd_vals else None, is_background=True)

            valign_vals = tc_el.xpath("./w:tcPr/w:vAlign/@w:val")
            valign_raw = str(valign_vals[0]).lower() if valign_vals else "top"
            valign = "middle" if valign_raw == "center" else ("bottom" if valign_raw == "bottom" else "top")

            cell_para_htmls: list[str] = []
            cell_plain_parts: list[str] = []
            for p_el in tc_el.xpath("./w:p"):
                p_obj = DocxParagraph(p_el, doc)
                if p_obj.text and p_obj.text.strip():
                    cell_plain_parts.append(p_obj.text.strip())
                p_html, _ = _render_docx_paragraph_html(
                    para=p_obj,
                    doc=doc,
                    num_resolver=None,
                    extra_images=extra_images,
                    in_table_cell=True,
                )
                if p_html:
                    cell_para_htmls.append(p_html)

            # Nested sub-tables inside a table cell (rare, flatten paragraphs cleanly)
            for sub_tbl_el in tc_el.xpath("./w:tbl"):
                for sub_p_el in sub_tbl_el.xpath(".//w:p"):
                    sp_obj = DocxParagraph(sub_p_el, doc)
                    if sp_obj.text and sp_obj.text.strip():
                        cell_plain_parts.append(sp_obj.text.strip())
                        cell_para_htmls.append(f"<p>{_escape_html_preserving_breaks(sp_obj.text.strip())}</p>")

            row_cells.append(
                {
                    "col_start": col_cursor,
                    "colspan": colspan,
                    "rowspan": 1,
                    "vmerge": vmerge_mode,
                    "skip": False,
                    "bg": bg_hex,
                    "valign": valign,
                    "is_header_tr": is_header_tr,
                    "html": "".join(cell_para_htmls) if cell_para_htmls else "\u00a0",
                    "plain": " ".join(cell_plain_parts).strip(),
                }
            )
            col_cursor += colspan

        max_cols = max(max_cols, col_cursor)
        parsed_rows.append(row_cells)

    if not parsed_rows or max_cols == 0:
        return ""

    # Resolve `vMerge` vertical spans, capping at `_MAX_HTML_ROWSPAN` so 500-row tables paginate smoothly!
    active_vmerge_by_col: dict[int, dict[str, Any]] = {}
    for row in parsed_rows:
        seen_cols_in_row: set[int] = set()
        for cell in row:
            c_start = cell["col_start"]
            seen_cols_in_row.add(c_start)
            vm = cell["vmerge"]
            if vm == "restart":
                active_vmerge_by_col[c_start] = cell
            elif vm == "continue":
                origin = active_vmerge_by_col.get(c_start)
                if origin is not None and origin["colspan"] == cell["colspan"] and origin["rowspan"] < _MAX_HTML_ROWSPAN:
                    origin["rowspan"] += 1
                    cell["skip"] = True
                else:
                    # Start a fresh chunk after `_MAX_HTML_ROWSPAN` rows, carrying parent label so context stays readable
                    if origin is not None and not cell["plain"] and origin["plain"]:
                        cell["html"] = origin["html"]
                        cell["plain"] = origin["plain"]
                    cell["vmerge"] = "restart"
                    active_vmerge_by_col[c_start] = cell
            else:
                active_vmerge_by_col.pop(c_start, None)

        for tracked_col in list(active_vmerge_by_col.keys()):
            if tracked_col not in seen_cols_in_row:
                active_vmerge_by_col.pop(tracked_col, None)

    # Drop completely blank rows (unless single-row table)
    non_empty_rows = [
        r for r in parsed_rows if any(c["plain"] or c["skip"] for c in r)
    ]
    if not non_empty_rows:
        return ""

    first_row_texts = [c["plain"] for c in non_empty_rows[0]]
    tbl_kind = _classify_docx_table_kind(tbl_el, len(non_empty_rows), max_cols, first_row_texts)

    # Column `<colgroup>` percentages (boost narrow columns that hold long unbroken words in 8+ col tables)
    colgroup_html = ""
    if grid_col_widths and len(grid_col_widths) == max_cols:
        adj_widths = list(grid_col_widths)
    elif max_cols >= 8 and tbl_kind == "data":
        adj_widths = [1000.0] * max_cols
    else:
        adj_widths = []

    pcts: list[float] = []
    if adj_widths and len(adj_widths) == max_cols:
        if max_cols >= 8 and tbl_kind == "data":
            total_base = sum(adj_widths) or 1.0
            for c_i in range(max_cols):
                longest_tok = 0
                for r in non_empty_rows[:30]:
                    for cell in r:
                        if not cell["skip"] and cell["col_start"] == c_i and cell["colspan"] == 1 and cell["plain"]:
                            for tok in re.split(r"[\s/\-.,;:()]+", cell["plain"]):
                                if len(tok) > longest_tok:
                                    longest_tok = len(tok)
                if longest_tok >= 14:
                    min_needed = total_base * min(0.18, longest_tok * 0.0085)
                    if adj_widths[c_i] < min_needed:
                        adj_widths[c_i] = min_needed
        total_w = sum(adj_widths) or 1.0
        raw_pcts = [max(3.4, (w / total_w) * 100.0) for w in adj_widths]
        norm_sum = sum(raw_pcts) or 100.0
        pcts = [(p / norm_sum) * 100.0 for p in raw_pcts]
        cols_tags = "".join(f'<col style="width: {p:.2f}%;"/>' for p in pcts)
        colgroup_html = f"<colgroup>{cols_tags}</colgroup>"

    css_class = {
        "req": "req-table",
        "sig": "sig-table",
        "layout": "layout-table",
        "data": "data-table",
    }[tbl_kind]
    extra_tbl_style = ' style="font-size: 6.7pt;"' if (tbl_kind == "data" and max_cols >= 12) else ""

    tr_html_list: list[str] = []
    for r_idx, row in enumerate(non_empty_rows):
        td_list: list[str] = []
        is_top_data_header = (
            tbl_kind == "data"
            and len(non_empty_rows) >= 2
            and (row[0]["is_header_tr"] or r_idx == 0)
        )

        for c_idx, cell in enumerate(row):
            if cell["skip"]:
                continue
            tag = "th" if is_top_data_header else "td"
            attrs: list[str] = []
            if cell["colspan"] > 1:
                attrs.append(f'colspan="{cell["colspan"]}"')
            if cell["rowspan"] > 1:
                attrs.append(f'rowspan="{cell["rowspan"]}"')

            c_styles: list[str] = [f'vertical-align: {cell["valign"]}']
            if r_idx == 0 and pcts:
                c0 = cell["col_start"]
                c1 = min(len(pcts), c0 + cell["colspan"])
                if c0 < c1:
                    span_pct = sum(pcts[c0:c1])
                    c_styles.append(f"width: {span_pct:.2f}%")
            if cell["bg"]:
                c_styles.append(f'background-color: {cell["bg"]}')

            # Right-align the second cell of a `1x2` requisite (`от ... | № ...`) or signature (`Ректор | Д.В. Орлов`)
            if tbl_kind in {"req", "sig"} and len(row) == 2 and c_idx == 1:
                c_styles.append("text-align: right")

            attrs.append(f'style="{"; ".join(c_styles)}"')
            attr_str = " " + " ".join(attrs) if attrs else ""
            td_list.append(f"<{tag}{attr_str}>{cell['html']}</{tag}>")

        if td_list:
            tr_html_list.append(f"<tr>{''.join(td_list)}</tr>")

    if not tr_html_list:
        return ""
    return f'<table class="{css_class}"{extra_tbl_style}>{colgroup_html}{"".join(tr_html_list)}</table>'


def _extract_docx_header_footer_html(
    doc: docx.document.Document,
    body_plain_alnum: str,
    extra_images: dict[str, bytes],
) -> tuple[str, str]:
    """Extract non-duplicated running header and EDO/registration footer blocks from DOCX sections."""
    hdr_parts: list[str] = []
    ftr_parts: list[str] = []
    seen_el_ids: set[int] = set()

    for sec in doc.sections:
        for is_ftr, hf in (
            (False, getattr(sec, "first_page_header", None)),
            (False, getattr(sec, "header", None)),
            (True, getattr(sec, "first_page_footer", None)),
            (True, getattr(sec, "footer", None)),
        ):
            if hf is None or getattr(hf, "_element", None) is None:
                continue
            try:
                if getattr(hf, "is_linked_to_previous", False) and seen_el_ids:
                    continue
            except Exception:
                pass
            eid = id(hf._element)
            if eid in seen_el_ids:
                continue
            seen_el_ids.add(eid)

            plain_chunks: list[str] = []
            html_chunks: list[str] = []
            for p in getattr(hf, "paragraphs", []):
                pt = (p.text or "").strip()
                if pt and not re.fullmatch(r"(?:Стр\.?\s*)?\d+(?:\s*(?:из|/)\s*\d+)?", pt, flags=re.IGNORECASE):
                    plain_chunks.append(pt)
                    p_html, _ = _render_docx_paragraph_html(p, doc, None, extra_images, in_table_cell=False)
                    if p_html:
                        html_chunks.append(p_html)
            for tbl in getattr(hf, "tables", []):
                t_plain = " ".join(c.text.strip() for r in tbl.rows for c in r.cells if c.text.strip())
                if t_plain:
                    plain_chunks.append(t_plain)
                    t_html = _render_docx_table_html(tbl, doc, extra_images)
                    if t_html:
                        html_chunks.append(t_html)

            if not plain_chunks or not html_chunks:
                continue
            combined_alnum = re.sub(r"[^0-9a-zа-я]+", "", " ".join(plain_chunks).lower().replace("ё", "е"))
            if len(combined_alnum) < 5 or combined_alnum in body_plain_alnum:
                continue
            body_plain_alnum += combined_alnum

            joined_plain = " ".join(plain_chunks)
            if is_ftr:
                if _EDO_STAMP_RE.search(joined_plain):
                    ftr_parts.append(f'<div class="edo-stamp">{"".join(html_chunks)}</div>')
                else:
                    ftr_parts.append("".join(html_chunks))
            else:
                hdr_parts.append(f'<div class="doc-running-header">{"".join(html_chunks)}</div>')

    return "".join(hdr_parts), "".join(ftr_parts)


def _docx_to_html_bundle(
    file_path: Path,
) -> tuple[str, dict[str, bytes], tuple[float, float], tuple[float, float, float, float], bool]:
    """
    Convert a `.docx` file into `(full_html, extra_images, (page_w, page_h), (m_left, m_top, m_right, m_bottom), is_landscape)`.
    """
    doc = docx.Document(str(file_path))
    sec0 = doc.sections[0] if doc.sections else None

    raw_w = float(sec0.page_width.pt) if (sec0 and sec0.page_width) else A4_PORTRAIT_W
    raw_h = float(sec0.page_height.pt) if (sec0 and sec0.page_height) else A4_PORTRAIT_H
    is_landscape = raw_w > raw_h * 1.05

    if is_landscape:
        page_w, page_h = A4_LANDSCAPE_W, A4_LANDSCAPE_H
        m_left = max(24.0, min(56.0, float(sec0.left_margin.pt) if (sec0 and sec0.left_margin) else 34.0))
        m_right = max(24.0, min(56.0, float(sec0.right_margin.pt) if (sec0 and sec0.right_margin) else 34.0))
        m_top = max(24.0, min(48.0, float(sec0.top_margin.pt) if (sec0 and sec0.top_margin) else 28.0))
        m_bottom = max(28.0, min(52.0, float(sec0.bottom_margin.pt) if (sec0 and sec0.bottom_margin) else 34.0))
    else:
        page_w, page_h = A4_PORTRAIT_W, A4_PORTRAIT_H
        m_left = max(42.0, min(85.0, float(sec0.left_margin.pt) if (sec0 and sec0.left_margin) else 70.0))
        m_right = max(32.0, min(56.0, float(sec0.right_margin.pt) if (sec0 and sec0.right_margin) else 42.5))
        m_top = max(36.0, min(64.0, float(sec0.top_margin.pt) if (sec0 and sec0.top_margin) else 50.0))
        m_bottom = max(40.0, min(64.0, float(sec0.bottom_margin.pt) if (sec0 and sec0.bottom_margin) else 52.0))

    num_resolver: WordNumberingResolver | None = None
    try:
        num_resolver = WordNumberingResolver(file_path)
    except Exception:
        num_resolver = None

    extra_images: dict[str, bytes] = {}
    body_fragments: list[str] = []
    plain_accumulator: list[str] = []

    for child in doc.element.body.iterchildren():
        if isinstance(child, CT_P):
            p_obj = DocxParagraph(child, doc)
            if p_obj.text:
                plain_accumulator.append(p_obj.text)
            frag, br_after = _render_docx_paragraph_html(
                para=p_obj,
                doc=doc,
                num_resolver=num_resolver,
                extra_images=extra_images,
                in_table_cell=False,
            )
            if frag:
                body_fragments.append(frag)
            if br_after:
                body_fragments.append('<div class="page-break"></div>')

            # Section break inside paragraph properties (`w:pPr/w:sectPr`) with explicit new-page type
            sect_type_vals = child.xpath("./w:pPr/w:sectPr/w:type/@w:val")
            if sect_type_vals and str(sect_type_vals[0]).lower() in {"nextpage", "evenpage", "oddpage"}:
                body_fragments.append('<div class="page-break"></div>')

        elif isinstance(child, CT_Tbl):
            tbl_obj = DocxTable(child, doc)
            t_html = _render_docx_table_html(tbl_obj, doc, extra_images)
            if t_html:
                body_fragments.append(t_html)

    body_alnum = re.sub(r"[^0-9a-zа-я]+", "", " ".join(plain_accumulator).lower().replace("ё", "е"))
    hdr_html, ftr_html = _extract_docx_header_footer_html(doc, body_alnum, extra_images)

    full_body = "".join([hdr_html, *body_fragments, ftr_html])
    if not full_body.strip():
        full_body = "<p>\u00a0</p>"

    return (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'/></head><body>{full_body}</body></html>",
        extra_images,
        (page_w, page_h),
        (m_left, m_top, m_right, m_bottom),
        is_landscape,
    )


def _parsed_blocks_to_a4_html(blocks: list[ParsedBlock], title_fallback: str = "") -> str:
    """
    Reconstruct an authentic official A4 document HTML layout from `ParsedBlock` records
    produced for legacy `.doc` files (Word 97–2003 OLE2, RTF, HTML, WordML XML).
    """
    fragments: list[str] = []
    cur_explicit_page = 1

    for idx, blk in enumerate(blocks):
        b_page = max(1, int(blk.page_num or 1))
        if idx > 0 and b_page > cur_explicit_page:
            fragments.append('<div class="page-break"></div>')
            cur_explicit_page = b_page
        elif b_page > cur_explicit_page:
            cur_explicit_page = b_page

        btype = str(blk.block_type or "paragraph")
        txt = (blk.text or "").strip()

        if btype == BlockType.STAMP.value or (txt and _EDO_STAMP_RE.search(txt) and len(txt) <= 280):
            fragments.append(f'<div class="edo-stamp">{_escape_html_preserving_breaks(txt)}</div>')
            continue

        if btype == BlockType.TABLE.value and (blk.table_headers or blk.table_rows):
            hdrs = list(blk.table_headers or [])
            rows = list(blk.table_rows or [])
            # 1x2 requisite table (`от DD.MM.YYYY | № ...`)
            if not rows and len([c for c in hdrs if c.strip()]) == 2 and _REQUISITE_PAIR_RE.search(" | ".join(hdrs)):
                non_e = [c.strip() for c in hdrs if c.strip()]
                fragments.append(
                    f'<table class="req-table"><tr>'
                    f'<td>{_escape_html_preserving_breaks(non_e[0])}</td>'
                    f'<td style="text-align: right;">{_escape_html_preserving_breaks(non_e[1])}</td>'
                    f"</tr></table>"
                )
                continue

            col_cnt = max(len(hdrs), max((len(r) for r in rows), default=0))
            if col_cnt > 0:
                tr_list: list[str] = []
                if any(h.strip() for h in hdrs):
                    ths = "".join(
                        f"<th>{_escape_html_preserving_breaks(hdrs[c] if c < len(hdrs) else '')}</th>"
                        for c in range(col_cnt)
                    )
                    tr_list.append(f"<tr>{ths}</tr>")
                for r in rows:
                    tds = "".join(
                        f"<td>{_escape_html_preserving_breaks(str(r[c] if c < len(r) and r[c] is not None else ''))}</td>"
                        for c in range(col_cnt)
                    )
                    tr_list.append(f"<tr>{tds}</tr>")
                fragments.append(f'<table class="data-table">{"".join(tr_list)}</table>')
                continue

        if not txt:
            continue

        # Split multi-line paragraph blocks into individual visual paragraphs so headers/clauses align authentically
        lines = [ln.strip() for ln in txt.split("\n") if ln.strip()]
        for ln in lines:
            esc = _escape_html_preserving_breaks(ln)
            # Detect inline `ПРИКАЗ от DD.MM.YYYY № N` on one line -> render centered ПРИКАЗ + 1x2 requisite bar!
            m_ord = re.match(
                r"^\s*(ПРИКАЗ|РАСПОРЯЖЕНИЕ)\s+(от\s+\d{1,2}[.\s/]\d{1,2}[.\s/]\d{2,4}(?:\s*г\.?)?)\s+((?:№|N)\s*\S+)\s*$",
                ln,
                flags=re.IGNORECASE,
            )
            if m_ord:
                kind_lbl = html.escape(m_ord.group(1).upper())
                dt_lbl = html.escape(m_ord.group(2))
                no_lbl = html.escape(m_ord.group(3))
                fragments.append(f'<p style="text-align: center; font-size: 13.5pt; margin-top: 6pt;"><b>{kind_lbl}</b></p>')
                fragments.append(
                    f'<table class="req-table"><tr><td>{dt_lbl}</td><td style="text-align: right;">{no_lbl}</td></tr></table>'
                )
                continue

            if btype == BlockType.HEADING.value or blk.heading_level is not None:
                lvl = min(3, max(1, int(blk.heading_level or 2)))
                align_st = "center" if (lvl == 1 or idx <= 2) else "left"
                fragments.append(f'<h{lvl} style="text-align: {align_st};">{esc}</h{lvl}>')
            elif _CENTER_DOC_KIND_RE.match(ln):
                fragments.append(f'<p style="text-align: center; font-size: 13pt; margin-top: 7pt;"><b>{esc}</b></p>')
            elif _MINISTRY_HEADER_RE.match(ln):
                is_uni = "институт" in ln.lower() or "университет" in ln.lower()
                inner = f"<b>{esc}</b>" if is_uni else esc
                fragments.append(f'<p style="text-align: center; margin-bottom: 2pt;">{inner}</p>')
            elif _CORNER_STAMP_START_RE.match(ln) and len(ln) < 160:
                fragments.append(f'<p style="margin-left: 210pt; text-align: left;">{esc}</p>')
            else:
                # Bold leading clause number (`1.`, `1.1.`, `2.3.`) for clear normative readability
                m_cl = re.match(r"^(\d{1,2}(?:\.\d{1,2})*\.)\s+(.+)$", ln)
                if m_cl:
                    cl_html = f"<b>{html.escape(m_cl.group(1))}</b> {_escape_html_preserving_breaks(m_cl.group(2))}"
                    fragments.append(f'<p style="text-align: justify; text-indent: 22pt;">{cl_html}</p>')
                elif len(ln) >= 75:
                    fragments.append(f'<p style="text-align: justify; text-indent: 22pt;">{esc}</p>')
                else:
                    fragments.append(f"<p>{esc}</p>")

    if not fragments:
        fallback = html.escape(title_fallback or "Документ пуст")
        fragments.append(f"<p>{fallback}</p>")

    return f"<!DOCTYPE html><html><head><meta charset='utf-8'/></head><body>{''.join(fragments)}</body></html>"


def _doc_to_html_bundle(
    file_path: Path,
) -> tuple[str, dict[str, bytes], tuple[float, float], tuple[float, float, float, float], bool]:
    """Convert any `.doc` file (disguised OOXML ZIP, OLE2 Word 97-2003, RTF, HTML/XML) into an A4 HTML bundle."""
    raw_head = file_path.read_bytes()[:16]
    if raw_head[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(file_path) as zf:
                if any(n.startswith("word/") for n in zf.namelist()):
                    return _docx_to_html_bundle(file_path)
        except Exception:
            pass

    blocks, _meta = parse_document(file_path)
    html_doc = _parsed_blocks_to_a4_html(blocks, title_fallback=file_path.stem)
    return (
        html_doc,
        {},
        (A4_PORTRAIT_W, A4_PORTRAIT_H),
        (68.0, 50.0, 42.5, 52.0),
        False,
    )


def _try_libreoffice_headless(src_path: Path, dest_pdf_path: Path) -> bool:
    """Use headless LibreOffice (`soffice`) if installed on the host; returns True on valid PDF creation."""
    soffice_bin = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice_bin:
        return False

    try:
        with tempfile.TemporaryDirectory(prefix="mipt_lo_") as tmp_dir:
            profile_uri = Path(tmp_dir, "profile").as_uri()
            cmd = [
                soffice_bin,
                f"-env:UserInstallation={profile_uri}",
                "--headless",
                "--nologo",
                "--nofirststartwizard",
                "--convert-to",
                "pdf",
                "--outdir",
                tmp_dir,
                str(src_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, timeout=45, check=False)
            out_cand = Path(tmp_dir) / f"{src_path.stem}.pdf"
            if proc.returncode == 0 and out_cand.is_file() and out_cand.stat().st_size > 200:
                dest_pdf_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_target = dest_pdf_path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp")
                shutil.copyfile(out_cand, tmp_target)
                os.replace(tmp_target, dest_pdf_path)
                return True
    except Exception as exc:
        logger.warning("LibreOffice headless conversion failed for %s, falling back to Story renderer: %s", src_path.name, exc)
    return False


def _render_html_bundle_to_pdf_bytes(
    html_str: str,
    extra_images: dict[str, bytes],
    page_size: tuple[float, float],
    margins: tuple[float, float, float, float],
    is_landscape: bool,
) -> bytes:
    """Paginate `html_str` onto A4 sheets via `fitz.Story` + `fitz.Archive` and stamp page numbers."""
    font_dirs, font_map, reg_ttf_path = _discover_fonts()
    arch = fitz.Archive()
    for d in font_dirs:
        arch.add(d)
    for img_name, img_bytes in extra_images.items():
        arch.add(img_bytes, img_name)

    css_str = _build_print_css(font_map, is_landscape=is_landscape)
    page_w, page_h = page_size
    m_left, m_top, m_right, m_bottom = margins

    mediabox = fitz.Rect(0.0, 0.0, page_w, page_h)
    where = fitz.Rect(m_left, m_top, page_w - m_right, page_h - m_bottom)

    story = fitz.Story(html=html_str, user_css=css_str, archive=arch)
    bio = io.BytesIO()
    writer = fitz.DocumentWriter(bio)

    max_pages_guard = 250
    page_counter = 0
    more = 1
    while more and page_counter < max_pages_guard:
        page_counter += 1
        dev = writer.begin_page(mediabox)
        more, _filled = story.place(where)
        story.draw(dev)
        writer.end_page()
    writer.close()

    raw_pdf = bio.getvalue()

    # Stamp clean running page numbers on multi-page documents
    pdf_doc = fitz.open("pdf", raw_pdf)
    try:
        total_p = len(pdf_doc)
        if total_p >= 2:
            font_kwargs: dict[str, Any] = {}
            if reg_ttf_path is not None and reg_ttf_path.is_file():
                font_kwargs = {"fontname": "FtrSerif", "fontfile": str(reg_ttf_path)}
            for idx, page in enumerate(pdf_doc):
                footer_rect = fitz.Rect(m_left, page_h - 32.0, page_w - m_right, page_h - 12.0)
                page.insert_textbox(
                    footer_rect,
                    f"{idx + 1}",
                    fontsize=9.0,
                    color=(0.35, 0.38, 0.43),
                    align=fitz.TEXT_ALIGN_CENTER,
                    **font_kwargs,
                )
        final_bytes = pdf_doc.tobytes(garbage=3, deflate=True)
    finally:
        pdf_doc.close()

    return final_bytes


def ensure_word_preview_pdf(
    file_path: str | Path,
    sha256: str | None = None,
    cache_dir: Path | None = None,
    force_rebuild: bool = False,
) -> Path:
    """
    Ensure a cached high-fidelity A4 preview PDF exists at `data/preview_cache/{sha256}.pdf`
    for the given `.docx` or `.doc` file, creating it atomically if not yet cached.
    """
    src_path = Path(file_path).resolve()
    if not src_path.is_file():
        raise FileNotFoundError(f"Word document not found on disk: {src_path}")

    effective_sha = (sha256 or "").strip() or compute_path_sha256(src_path)
    cache_pdf_path = get_preview_cache_path(effective_sha, cache_dir=cache_dir)

    if not force_rebuild and cache_pdf_path.is_file() and cache_pdf_path.stat().st_size > 100:
        return cache_pdf_path

    with _RENDER_LOCK:
        if not force_rebuild and cache_pdf_path.is_file() and cache_pdf_path.stat().st_size > 100:
            return cache_pdf_path

        if _try_libreoffice_headless(src_path, cache_pdf_path):
            return cache_pdf_path

        ext = src_path.suffix.lower()
        if ext == ".docx":
            try:
                bundle = _docx_to_html_bundle(src_path)
            except Exception as docx_exc:
                logger.warning("OOXML direct bundle fallback to parser on %s: %s", src_path.name, docx_exc)
                bundle = _doc_to_html_bundle(src_path)
        else:
            bundle = _doc_to_html_bundle(src_path)

        html_str, extra_images, page_size, margins, is_landscape = bundle
        pdf_bytes = _render_html_bundle_to_pdf_bytes(
            html_str=html_str,
            extra_images=extra_images,
            page_size=page_size,
            margins=margins,
            is_landscape=is_landscape,
        )

        cache_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_pdf_path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp")
        try:
            tmp_path.write_bytes(pdf_bytes)
            os.replace(tmp_path, cache_pdf_path)
        except Exception:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise

    return cache_pdf_path


def render_preview_pdf_page_png(
    pdf_path: str | Path,
    page_number: int,
    dpi: int = 150,
) -> bytes:
    """Render 1-based `page_number` of `pdf_path` as a crisp RGB PNG image at `dpi` (clamped 96..240)."""
    if page_number < 1:
        raise IndexError(f"Некорректный номер страницы: {page_number}")
    clamped_dpi = max(96, min(240, int(dpi)))

    pdf_doc = fitz.open(str(pdf_path))
    try:
        total = len(pdf_doc)
        if page_number > total:
            raise IndexError(f"Страница {page_number} выходит за пределы документа ({total} стр.)")
        page = pdf_doc[page_number - 1]
        pix = page.get_pixmap(dpi=clamped_dpi, alpha=False)
        return pix.tobytes("png")
    finally:
        pdf_doc.close()

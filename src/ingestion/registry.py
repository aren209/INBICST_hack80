from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.config import DATA_DIR, REGISTRY_OVERRIDES_PATH


@dataclass
class DocumentMetadata:
    file_name: str
    file_path: str
    file_sha256: str
    doc_id: str
    doc_title: str
    doc_type: str = "документ"
    doc_number: str = "б/н"
    doc_date: str = ""
    effective_from: str = ""
    status: str = "active"  # active | superseded | draft
    supersedes: list[str] = field(default_factory=list)
    superseded_by: list[str] = field(default_factory=list)
    related_docs: list[str] = field(default_factory=list)
    version_note: str = ""
    quality_warning: str = ""
    companion_file: str = ""
    master_doc_id: str = ""
    is_duplicate_appendix_of: str = ""
    skip_embedded_cfo_table: bool = False
    rotated_pages: list[int] = field(default_factory=list)
    topic_tags: list[str] = field(default_factory=list)


class DocumentRegistry:
    """Automatic document metadata extractor + YAML override merger."""

    DATE_PATTERN = re.compile(
        r"(?:от\s*)?(\d{2})\.(\d{2})\.(20\d{2})", re.IGNORECASE
    )
    ORDER_NO_PATTERN = re.compile(
        r"(?:Приказ|Распоряжение).*?(?:№|N)\s*([0-9]+(?:-[0-9а-яА-Яa-zA-Z]+)?)",
        re.IGNORECASE | re.DOTALL,
    )
    FILENAME_NO_PATTERN = re.compile(
        r"(?:№|N|^|\s)(\d{3,4}-\d|\d{3})(?:\s|_|$)", re.IGNORECASE
    )
    AMENDS_PATTERN = re.compile(
        r"(?:во\s+изменение\s+приказа|внесении\s+изменений\s+в\s+приказ).*?(?:№|N)\s*([0-9]+(?:-[0-9]+)?)",
        re.IGNORECASE | re.DOTALL,
    )
    REVOKES_PATTERN = re.compile(
        r"(?:признать\s+утративши[мй]\s+силу).*?(?:№|N)\s*([0-9]+(?:-[0-9]+)?)",
        re.IGNORECASE | re.DOTALL,
    )

    def __init__(
        self,
        overrides_path: Path = REGISTRY_OVERRIDES_PATH,
        corpus_dir: Path = DATA_DIR,
    ) -> None:
        self.overrides: dict[str, dict[str, Any]] = {}
        self._sha_to_override: dict[str, dict[str, Any]] = {}
        if overrides_path.exists():
            with open(overrides_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                self.overrides = data.get("documents", {})
        if corpus_dir.exists():
            for name, ov in self.overrides.items():
                candidate = corpus_dir / name
                if candidate.is_file():
                    try:
                        self._sha_to_override[self.compute_sha256(candidate)] = ov
                    except Exception:
                        pass

    def _find_override(self, fname: str, file_sha: str) -> dict[str, Any]:
        if fname in self.overrides:
            return self.overrides[fname]
        if file_sha in self._sha_to_override:
            return self._sha_to_override[file_sha]
        # Fallback: если в отладочную форму загружена копия с префиксом (copy_..., 1_...)
        lower_fname = fname.lower()
        for known_name, ov in self.overrides.items():
            if lower_fname.endswith(known_name.lower()):
                return ov
        return {}

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _slugify(text: str) -> str:
        slug = re.sub(r"[^0-9a-zA-Zа-яА-Я]+", "_", text.strip().lower())
        return slug.strip("_")[:48] or "doc"

    def resolve(self, file_path: Path, sample_text: str = "") -> DocumentMetadata:
        fname = file_path.name
        file_sha = self.compute_sha256(file_path)
        override = self._find_override(fname, file_sha)

        # 1. Auto-extract from filename + first 1500 chars of text
        header_probe = f"{fname}\n{sample_text[:1500]}"
        doc_date = ""
        m_date = self.DATE_PATTERN.search(header_probe)
        if m_date:
            dd, mm, yyyy = m_date.group(1), m_date.group(2), m_date.group(3)
            doc_date = f"{yyyy}-{mm}-{dd}"

        doc_number = "б/н"
        m_no = self.ORDER_NO_PATTERN.search(header_probe) or self.FILENAME_NO_PATTERN.search(fname)
        if m_no:
            doc_number = m_no.group(1)

        lower_name = fname.lower()
        if "распоряжение" in lower_name:
            doc_type = "распоряжение"
        elif "приказ" in lower_name:
            doc_type = "приказ и регламент"
        elif "приложение" in lower_name or "бланк" in lower_name:
            doc_type = "приложение / форма"
        else:
            doc_type = "регламент"

        supersedes: list[str] = []
        for m in self.AMENDS_PATTERN.finditer(sample_text[:3000]):
            supersedes.append(f"order_{m.group(1).replace('-', '_')}")
        for m in self.REVOKES_PATTERN.finditer(sample_text[:3000]):
            supersedes.append(f"order_{m.group(1).replace('-', '_')}")

        default_doc_id = (
            f"doc_{doc_number.replace('-', '_')}_{doc_date[:4]}"
            if doc_number != "б/н" and doc_date
            else f"doc_{file_sha[:10]}"
        )

        # 2. Merge with verified YAML override (if present)
        return DocumentMetadata(
            file_name=fname,
            file_path=str(file_path.resolve()),
            file_sha256=file_sha,
            doc_id=override.get("doc_id", default_doc_id),
            doc_title=override.get("doc_title", file_path.stem),
            doc_type=override.get("doc_type", doc_type),
            doc_number=override.get("doc_number", doc_number),
            doc_date=override.get("doc_date", doc_date),
            effective_from=override.get("effective_from", doc_date),
            status=override.get("status", "active"),
            supersedes=override.get("supersedes", supersedes),
            superseded_by=override.get("superseded_by", []),
            related_docs=override.get("related_docs", []),
            version_note=override.get("version_note", ""),
            quality_warning=override.get("quality_warning", ""),
            companion_file=override.get("companion_file", ""),
            master_doc_id=override.get("master_doc_id", ""),
            is_duplicate_appendix_of=override.get("is_duplicate_appendix_of", ""),
            skip_embedded_cfo_table=bool(override.get("skip_embedded_cfo_table", False)),
            rotated_pages=list(override.get("rotated_pages", [])),
            topic_tags=list(override.get("topic_tags", [])),
        )

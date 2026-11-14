"""Dense (Local GPU/CPU BAAI/bge-m3) and Local Russian Sparse BM25 (razdel + pymorphy3 + CRC32) Embedders."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
import zlib
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import pymorphy3
import razdel
from openai import OpenAI
from qdrant_client.http import models
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import (
    EMBED_CACHE_DIR,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIM,
    settings,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Russian Stopwords (strictly function words; NO numbers or domain terms!)
# ---------------------------------------------------------------------------
RUSSIAN_STOPWORDS: frozenset[str] = frozenset(
    {
        "и",
        "в",
        "во",
        "не",
        "что",
        "он",
        "на",
        "я",
        "с",
        "со",
        "как",
        "а",
        "то",
        "все",
        "она",
        "так",
        "его",
        "но",
        "да",
        "ты",
        "к",
        "у",
        "же",
        "вы",
        "за",
        "бы",
        "по",
        "только",
        "ее",
        "мне",
        "было",
        "вот",
        "от",
        "меня",
        "еще",
        "нет",
        "о",
        "из",
        "ему",
        "теперь",
        "когда",
        "даже",
        "ну",
        "вдруг",
        "ли",
        "если",
        "уже",
        "или",
        "ни",
        "быть",
        "был",
        "него",
        "до",
        "вас",
        "нибудь",
        "опять",
        "уж",
        "вам",
        "ведь",
        "там",
        "потом",
        "себя",
        "ничего",
        "ей",
        "может",
        "они",
        "тут",
        "где",
        "есть",
        "надо",
        "ней",
        "для",
        "мы",
        "тебя",
        "их",
        "чем",
        "была",
        "сам",
        "чтоб",
        "без",
        "будто",
        "чего",
        "раз",
        "тоже",
        "себе",
        "под",
        "будет",
        "ж",
        "тогда",
        "кто",
        "этот",
        "того",
        "потому",
        "этого",
        "какой",
        "совсем",
        "ним",
        "здесь",
        "этом",
        "почти",
        "мой",
        "тем",
        "чтобы",
        "нее",
        "сейчас",
        "были",
        "куда",
        "зачем",
        "всех",
        "никогда",
        "можно",
        "при",
        "наконец",
        "об",
        "другой",
        "хоть",
        "после",
        "над",
        "больше",
        "тот",
        "через",
        "эти",
        "нас",
        "про",
        "всего",
        "них",
        "какая",
        "много",
        "разве",
        "эту",
        "моя",
        "впрочем",
        "хорошо",
        "свою",
        "этой",
        "перед",
        "иногда",
        "лучше",
        "чуть",
        "том",
        "нельзя",
        "такой",
        "им",
        "более",
        "всегда",
        "конечно",
        "всю",
        "между",
        "это",
        "также",
        "либо",
        "который",
        "которая",
        "которое",
        "которые",
        "которого",
        "которой",
        "которых",
        "которому",
        "которым",
        "которыми",
        "свой",
        "своя",
        "свое",
        "свои",
        "своего",
        "своей",
        "своих",
        "наш",
        "ваш",
        "таковой",
        "данный",
        "иной",
    }
)

# Domain abbreviations that must NOT be altered by pymorphy3 lemmatization
PROTECTED_DOMAIN_TERMS: frozenset[str] = frozenset(
    {
        "флс",
        "цфо",
        "кфо",
        "квр",
        "косгу",
        "1с:до",
        "1с",
        "пфо",
        "фэу",
        "рид",
        "рцни",
        "кмо",
        "кз",
        "цод",
        "эдо",
        "ппс",
        "пумс",
        "иот",
        "эб",
        "мфти",
        "уит",
        "убуифк",
        "отитб",
        "днвр",
        "аху",
        "ндс",
        "гпх",
        "сз",
        "тз",
        "физтех",
        "фш",
        "sla",
        "kmo@mipt.ru",
        "1а",
        "1б",
    }
)

# Number words to digits so queries with "два месяца" / "2 месяца" or "шесть дней" / "6 дней" cross-match
NUM_WORD_TO_DIGIT: dict[str, str] = {
    "один": "1",
    "одна": "1",
    "одно": "1",
    "одного": "1",
    "одной": "1",
    "два": "2",
    "две": "2",
    "двух": "2",
    "двумя": "2",
    "три": "3",
    "трех": "3",
    "тремя": "3",
    "четыре": "4",
    "четырех": "4",
    "пять": "5",
    "пяти": "5",
    "шесть": "6",
    "шести": "6",
    "семь": "7",
    "семи": "7",
    "восемь": "8",
    "восьми": "8",
    "девять": "9",
    "девяти": "9",
    "десять": "10",
    "десяти": "10",
    "двенадцать": "12",
    "двенадцати": "12",
}

# ---------------------------------------------------------------------------
# Special Token Regexes (preserved before/alongside razdel splitting)
# ---------------------------------------------------------------------------
EMAIL_RE = re.compile(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", re.IGNORECASE)
BRAND_1C_RE = re.compile(r"\b1[cс]:[a-zа-яё0-9_]+\b", re.IGNORECASE)
ORDER_HYPHEN_RE = re.compile(r"\b(\d{1,5})-([0-9a-zа-я]{1,4})\b", re.IGNORECASE)
DOC_CODE_HYPHEN_RE = re.compile(r"\b([a-zа-я]{2,6})-(\d{4})\b", re.IGNORECASE)
PERCENT_RE = re.compile(r"(?<![a-zа-я0-9])(\d+(?:[.,]\d+)?)\s*%", re.IGNORECASE)
DATE_FULL_RE = re.compile(r"\b(0?[1-9]|[12]\d|3[01])\.(0?[1-9]|1[0-2])\.(19\d{2}|20\d{2})\b")
CLAUSE_DOT_RE = re.compile(r"(?<![\d.])(\d{1,2}(?:\.\d{1,3})+)(?![\d.])")
APPENDIX_ALNUM_RE = re.compile(r"\b(\d{1,2})([абab])\b", re.IGNORECASE)
COMPOUND_WORD_RE = re.compile(r"\b([а-я]{2,})-([а-я]{2,})\b", re.IGNORECASE)
CYRILLIC_WORD_RE = re.compile(r"^[а-я]+$")
LATIN_WORD_RE = re.compile(r"^[a-z0-9_]+$")
PURE_INT_RE = re.compile(r"^\d{1,6}$")


@lru_cache(maxsize=1)
def _get_morph_analyzer() -> pymorphy3.MorphAnalyzer:
    """Singleton pymorphy3 Russian morphological analyzer."""
    return pymorphy3.MorphAnalyzer(lang="ru")


@lru_cache(maxsize=65536)
def _lemmatize_ru_word(word: str) -> str:
    """Lemmatize a lowercased Russian word (with 'ё' already replaced by 'е')."""
    if word in PROTECTED_DOMAIN_TERMS:
        return word
    if len(word) <= 2:
        return word
    morph = _get_morph_analyzer()
    parsed = morph.parse(word)
    if not parsed:
        return word
    normal = parsed[0].normal_form.lower().replace("ё", "е")
    return normal


def term_to_crc32(term: str) -> int:
    """Deterministically hash a token/lemma string to an unsigned 32-bit integer."""
    return zlib.crc32(term.encode("utf-8")) & 0xFFFFFFFF


def tokenize_bm25(text: str) -> list[str]:
    """
    Tokenize and lemmatize Russian normative text for Sparse BM25 indexing & search.

    Preserves intact:
    - Hyphenated order numbers: '3065-1', '3699-1', '794-1', '5990-1', '1292-1', '3897-1' (+ bare number)
    - Clause numbers: '2.6', '6.1', '2.3.1'
    - Percentages: '16.4%', '8.2%', '20%' (normalizing comma '16,4%' -> '16.4%' + bare number '16.4')
    - Emails & brands: 'kmo@mipt.ru', '1с:до'
    - KVR/KOSGU codes and numbers: '244', '700', '2', '6', '13', '100', '1а', '1б'
    - Russian lemmas via razdel.tokenize + pymorphy3 ('ё' -> 'е', stopwords removed)
    """
    if not text or not text.strip():
        return []

    norm = text.lower().replace("ё", "е")
    tokens: list[str] = []

    # 1. Extract and preserve emails (e.g. kmo@mipt.ru)
    for m in EMAIL_RE.finditer(norm):
        email = m.group(0)
        tokens.append(email)
        local_part = email.split("@", 1)[0]
        if local_part:
            tokens.append(local_part)
    # Mask emails from norm so razdel doesn't split them into noise while keeping rest of text
    masked = EMAIL_RE.sub(" ", norm)

    # 2. Extract 1C brands with colon (e.g. 1С:ДО / 1c:до -> 1с:до)
    for m in BRAND_1C_RE.finditer(masked):
        raw_brand = m.group(0)
        # Normalize Latin 'c' after '1' to Cyrillic 'с'
        canon_brand = "1с:" + raw_brand.split(":", 1)[1]
        tokens.append(canon_brand)
        tokens.append("1с")
        if canon_brand == "1с:до":
            tokens.append("документооборот")
    masked = BRAND_1C_RE.sub(" ", masked)

    # 3. Extract percentages (e.g. 16.4%, 16,4%, 8.2%, 20%)
    for m in PERCENT_RE.finditer(masked):
        num_part = m.group(1).replace(",", ".")
        tokens.append(f"{num_part}%")
        tokens.append(num_part)
    masked = PERCENT_RE.sub(" ", masked)

    # 4. Extract hyphenated order numbers (e.g. 3065-1, 3699-1, 794-1, 5990-1, 1292-1, 3897-1)
    for m in ORDER_HYPHEN_RE.finditer(masked):
        full_order = f"{m.group(1)}-{m.group(2)}"
        tokens.append(full_order)
        tokens.append(m.group(1))
    masked = ORDER_HYPHEN_RE.sub(" ", masked)

    # 5. Extract hyphenated document codes (e.g. КЗ-2025, ЦОД-2025, ПОС-2025, HR-2025)
    for m in DOC_CODE_HYPHEN_RE.finditer(masked):
        full_code = f"{m.group(1)}-{m.group(2)}"
        tokens.append(full_code)
        tokens.append(m.group(1))
        tokens.append(m.group(2))

    # 6. Extract full dates DD.MM.YYYY first so they are not misread as clauses
    for m in DATE_FULL_RE.finditer(masked):
        full_date = m.group(0)
        year_str = m.group(3)
        tokens.append(full_date)
        tokens.append(year_str)
    masked = DATE_FULL_RE.sub(" ", masked)

    # 7. Extract dotted clause/decimal numbers (e.g. 2.6, 6.1, 2.3.1)
    for m in CLAUSE_DOT_RE.finditer(masked):
        clause_tok = m.group(1).strip(".")
        if clause_tok:
            tokens.append(clause_tok)
    masked = CLAUSE_DOT_RE.sub(" ", masked)

    # 8. Extract alphanumeric appendix numbers like 1а, 1б (normalizing Latin a/b -> Cyrillic а/б)
    for m in APPENDIX_ALNUM_RE.finditer(masked):
        num_p, let_p = m.group(1), m.group(2)
        let_Map = {"a": "а", "b": "б", "а": "а", "б": "б"}
        canon_app = f"{num_p}{let_Map.get(let_p, let_p)}"
        tokens.append(canon_app)

    # 9. Extract hyphenated Cyrillic compound words (e.g. физтех-школа, финансово-лицевой)
    for m in COMPOUND_WORD_RE.finditer(masked):
        p1, p2 = m.group(1), m.group(2)
        lem2 = _lemmatize_ru_word(p2)
        tokens.append(f"{p1}-{lem2}")

    # 10. Tokenize remaining text with razdel.tokenize + pymorphy3
    for tok in razdel.tokenize(masked):
        val = tok.text.strip().strip(".,;:!?()[]{}«»\"'—–-")
        if not val:
            continue

        # Normalize decimal comma inside numbers if any remained
        if re.match(r"^\d+,\d+$", val):
            val = val.replace(",", ".")
            tokens.append(val)
            continue

        if val in PROTECTED_DOMAIN_TERMS:
            tokens.append(val)
            continue

        if PURE_INT_RE.match(val):
            # Keep all integer tokens including single-digit ('2', '6', '13', '244', '700')
            tokens.append(val)
            continue

        if CYRILLIC_WORD_RE.match(val):
            if val in RUSSIAN_STOPWORDS:
                continue
            if val in NUM_WORD_TO_DIGIT:
                tokens.append(NUM_WORD_TO_DIGIT[val])
            lemma = _lemmatize_ru_word(val)
            if lemma in RUSSIAN_STOPWORDS:
                continue
            if lemma in NUM_WORD_TO_DIGIT and NUM_WORD_TO_DIGIT[lemma] not in tokens:
                tokens.append(NUM_WORD_TO_DIGIT[lemma])
            if len(lemma) >= 2 or lemma in PROTECTED_DOMAIN_TERMS:
                tokens.append(lemma)
            continue

        if LATIN_WORD_RE.match(val):
            if len(val) >= 2:
                tokens.append(val)
            continue

    return tokens


# Aliases for convenience / unit tests
tokenize_for_bm25 = tokenize_bm25
extract_bm25_tokens = tokenize_bm25


class SparseBM25Encoder:
    """Local Russian BM25 sparse vector encoder using razdel + pymorphy3 + CRC32 term hashing."""

    def tokenize(self, text: str) -> list[str]:
        """Return list of normalized tokens and lemmas for BM25."""
        return tokenize_bm25(text)

    def encode(self, text: str) -> models.SparseVector:
        """Encode text into a Qdrant SparseVector (CRC32 indices + TF counts)."""
        tokens = tokenize_bm25(text)
        if not tokens:
            return models.SparseVector(indices=[], values=[])

        tf_by_index: dict[int, float] = {}
        for token, count in Counter(tokens).items():
            idx = term_to_crc32(token)
            tf_by_index[idx] = tf_by_index.get(idx, 0.0) + float(count)

        sorted_items = sorted(tf_by_index.items(), key=lambda kv: kv[0])
        indices = [k for k, _ in sorted_items]
        values = [v for _, v in sorted_items]
        return models.SparseVector(indices=indices, values=values)

    def encode_with_tokens(self, text: str) -> tuple[models.SparseVector, list[str]]:
        """Encode text into a Qdrant SparseVector and also return the token list."""
        tokens = tokenize_bm25(text)
        if not tokens:
            return models.SparseVector(indices=[], values=[]), []

        tf_by_index: dict[int, float] = {}
        for token, count in Counter(tokens).items():
            idx = term_to_crc32(token)
            tf_by_index[idx] = tf_by_index.get(idx, 0.0) + float(count)

        sorted_items = sorted(tf_by_index.items(), key=lambda kv: kv[0])
        indices = [k for k, _ in sorted_items]
        values = [v for _, v in sorted_items]
        return models.SparseVector(indices=indices, values=values), tokens

    def encode_batch(self, texts: list[str]) -> list[models.SparseVector]:
        """Encode a batch of texts into Qdrant SparseVectors."""
        return [self.encode(t) for t in texts]


def encode_sparse_bm25(text: str) -> models.SparseVector:
    """Module-level helper to encode text into a Qdrant SparseVector."""
    return SparseBM25Encoder().encode(text)


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute exact cosine similarity between two dense vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _normalize_l2(vec: list[float]) -> list[float]:
    """L2-normalize a dense vector if its norm is non-zero."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 1e-12:
        return vec
    if abs(norm - 1.0) <= 1e-5:
        return vec
    return [float(x / norm) for x in vec]


_LOCAL_ST_MODELS: dict[tuple[str, str], Any] = {}
_LOCAL_ST_LOCK = threading.RLock()


def _resolve_torch_device(preferred: str = "auto") -> str:
    """Resolve optimal PyTorch execution device ('cuda', 'mps', or 'cpu')."""
    pref = (preferred or "auto").strip().lower()
    if pref and pref != "auto":
        return pref
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _get_local_sentence_transformer(model_name: str, device: str) -> Any:
    """Load and cache SentenceTransformer(BAAI/bge-m3) once per process and device."""
    key = (model_name, device)
    if key in _LOCAL_ST_MODELS:
        return _LOCAL_ST_MODELS[key]
    with _LOCAL_ST_LOCK:
        if key not in _LOCAL_ST_MODELS:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading local embedding model '%s' on device '%s'...", model_name, device)
            _LOCAL_ST_MODELS[key] = SentenceTransformer(model_name, device=device)
    return _LOCAL_ST_MODELS[key]


class DenseEmbedder:
    """
    Dense embedder powered by local `BAAI/bge-m3` (1024d Cosine, SentenceTransformer on CUDA/CPU)
    with optional OpenAI-compatible API fallback, batching (32), and SHA-256 disk caching in `data/embed_cache/`.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        dim: int = EMBEDDING_DIM,
        cache_dir: Path | None = None,
        backend: str | None = None,
        device: str | None = None,
    ) -> None:
        self.backend = (backend or settings.EMBEDDING_BACKEND or "local").strip().lower()
        self.device_preference = (device or settings.EMBEDDING_DEVICE or "auto").strip().lower()
        self.api_key = api_key if api_key is not None else settings.POLZA_API_KEY
        self.base_url = base_url if base_url is not None else (
            "local://sentence-transformers" if self.backend == "local" else settings.POLZA_BASE_URL
        )
        self.model = model if model is not None else (
            settings.EMBEDDING_MODEL if self.backend == "local" else settings.POLZA_EMBEDDING_MODEL
        )
        self.batch_size = max(1, batch_size)
        self.dim = dim
        self.cache_dir = cache_dir if cache_dir is not None else EMBED_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem_cache: dict[str, list[float]] = {}
        self._client: OpenAI | None = None
        self._resolved_device: str | None = None

    @property
    def resolved_device(self) -> str:
        if self._resolved_device is None:
            self._resolved_device = _resolve_torch_device(self.device_preference)
        return self._resolved_device

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key or "missing-api-key",
                base_url=self.base_url,
                timeout=60.0,
            )
        return self._client

    @staticmethod
    def _text_sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _cache_file(self, sha: str) -> Path:
        return self.cache_dir / f"{sha}.json"

    def _load_cached(self, sha: str) -> list[float] | None:
        if sha in self._mem_cache:
            return self._mem_cache[sha]
        path = self._cache_file(sha)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list) and len(data) == self.dim:
                vec = [float(x) for x in data]
                self._mem_cache[sha] = vec
                return vec
            if isinstance(data, dict) and isinstance(data.get("embedding"), list):
                emb = data["embedding"]
                if len(emb) == self.dim:
                    vec = [float(x) for x in emb]
                    self._mem_cache[sha] = vec
                    return vec
        except Exception as exc:
            logger.warning("Corrupted embedding cache file %s: %s", path, exc)
        return None

    def _save_cached(self, sha: str, vec: list[float]) -> None:
        self._mem_cache[sha] = vec
        path = self._cache_file(sha)
        tmp_path = path.with_suffix(f".{os.getpid()}.tmp")
        try:
            tmp_path.write_text(json.dumps(vec), encoding="utf-8")
            os.replace(tmp_path, path)
        except Exception as exc:
            logger.warning("Failed to write embedding cache %s: %s", path, exc)
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError as rm_exc:
                    logger.debug("Cleanup of temp embedding file failed: %s", rm_exc)

    def _encode_local_batch(self, batch: list[str]) -> list[list[float]]:
        """Encode a batch of texts locally using BAAI/bge-m3 (SentenceTransformer)."""
        with _LOCAL_ST_LOCK:
            st_model = _get_local_sentence_transformer(self.model, self.resolved_device)
            arr = st_model.encode(
                batch,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        vectors: list[list[float]] = []
        for row in arr:
            vec = _normalize_l2([float(x) for x in row.tolist()])
            if len(vec) != self.dim:
                raise ValueError(f"Expected embedding dim {self.dim}, got {len(vec)}")
            vectors.append(vec)
        return vectors

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1.0, min=1.0, max=15.0),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call_api_batch(self, batch: list[str]) -> list[list[float]]:
        """Encode batch via local BAAI/bge-m3 (default) or OpenAI-compatible embeddings endpoint."""
        if self.backend == "local" and self._client is None:
            return self._encode_local_batch(batch)
        response = self.client.embeddings.create(
            model=self.model,
            input=batch,
        )
        # Ensure items are ordered by index
        sorted_data = sorted(response.data, key=lambda item: item.index)
        vectors: list[list[float]] = []
        for item in sorted_data:
            vec = _normalize_l2([float(x) for x in item.embedding])
            if len(vec) != self.dim:
                raise ValueError(f"Expected embedding dim {self.dim}, got {len(vec)}")
            vectors.append(vec)
        return vectors

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts (no passage/query prefixes needed for baai/bge-m3).
        Uses persistent SHA-256 disk cache in data/embed_cache/ and batches uncached texts by 32.
        """
        if not texts:
            return []

        results: dict[int, list[float]] = {}
        uncached_by_sha: dict[str, str] = {}
        sha_to_indices: dict[str, list[int]] = {}

        for idx, raw_text in enumerate(texts):
            clean = (raw_text or "").strip()
            if not clean:
                results[idx] = [0.0] * self.dim
                continue
            sha = self._text_sha256(clean)
            cached = self._load_cached(sha)
            if cached is not None:
                results[idx] = cached
            else:
                uncached_by_sha[sha] = clean
                sha_to_indices.setdefault(sha, []).append(idx)

        if uncached_by_sha:
            unique_shas = list(uncached_by_sha.keys())
            for start in range(0, len(unique_shas), self.batch_size):
                batch_shas = unique_shas[start : start + self.batch_size]
                batch_texts = [uncached_by_sha[s] for s in batch_shas]
                batch_vecs = self._call_api_batch(batch_texts)
                for sha, vec in zip(batch_shas, batch_vecs):
                    self._save_cached(sha, vec)
                    for target_idx in sha_to_indices[sha]:
                        results[target_idx] = vec

        return [results[i] for i in range(len(texts))]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query (cached by SHA-256)."""
        vecs = self.embed_texts([query])
        return vecs[0] if vecs else [0.0] * self.dim

    # Aliases
    embed_batch = embed_texts
    encode = embed_texts
    embed_text = embed_query

    def check_health(self) -> dict[str, Any]:
        """Return local BAAI/bge-m3 + PaddleOCR-VL-1.6 configuration and cache health status."""
        cache_files_count = len(list(self.cache_dir.glob("*.json"))) if self.cache_dir.exists() else 0
        is_local = self.backend == "local"
        configured = True if is_local else bool(self.api_key and self.api_key != "your_polza_api_key_here")
        return {
            "configured": configured,
            "backend": self.backend,
            "device": self.resolved_device if is_local else "remote",
            "base_url": self.base_url,
            "embedding_model": self.model,
            "vlm_model": settings.OCR_VL_MODEL if settings.OCR_BACKEND != "polza_vlm" else settings.POLZA_VLM_MODEL,
            "dim": self.dim,
            "batch_size": self.batch_size,
            "cached_embeddings": cache_files_count,
        }


class HybridEmbedder:
    """Unified wrapper combining DenseEmbedder (baai/bge-m3) and SparseBM25Encoder (CRC32)."""

    def __init__(
        self,
        dense_embedder: DenseEmbedder | None = None,
        sparse_encoder: SparseBM25Encoder | None = None,
    ) -> None:
        self.dense = dense_embedder or DenseEmbedder()
        self.sparse = sparse_encoder or SparseBM25Encoder()

    def embed_chunks(self, texts: list[str]) -> tuple[list[list[float]], list[models.SparseVector]]:
        """Produce both dense vectors (1024d) and sparse BM25 vectors for a list of chunk embed_texts."""
        dense_vecs = self.dense.embed_texts(texts)
        sparse_vecs = self.sparse.encode_batch(texts)
        return dense_vecs, sparse_vecs

    def embed_query(
        self, query_text: str, sparse_query_text: str | None = None
    ) -> tuple[list[float], models.SparseVector, list[str]]:
        """
        Produce dense vector, sparse BM25 vector, and BM25 token list for a query.
        Allows passing an expanded query text for sparse/dense retrieval.
        """
        dense_vec = self.dense.embed_query(query_text)
        sparse_vec, tokens = self.sparse.encode_with_tokens(sparse_query_text or query_text)
        return dense_vec, sparse_vec, tokens

    def check_health(self) -> dict[str, Any]:
        return self.dense.check_health()


# Aliases for external callers
PolzaEmbedder = DenseEmbedder
BM25SparseEncoder = SparseBM25Encoder

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import pymorphy3
from fastembed import TextEmbedding
from qdrant_client import models
from razdel import tokenize

from src.chunking.glossary import GlossaryManager
from src.config import ARTIFACTS_DATA_DIR, DENSE_MODEL_NAME, MODELS_CACHE_DIR

BM25_STATS_PATH = ARTIFACTS_DATA_DIR / "bm25_stats.json"
DENSE_VEC_CACHE_PATH = ARTIFACTS_DATA_DIR / "dense_vectors_cache.npz"

RUSSIAN_STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще",
    "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли",
    "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь",
    "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
    "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя",
    "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже",
    "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому",
    "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти", "мой",
    "тем", "чтобы", "нее", "сейчас", "были", "куда", "зачем", "всех", "никогда",
    "можно", "при", "наконец", "два", "об", "другой", "хоть", "после", "над",
    "больше", "тот", "через", "эти", "нас", "про", "всего", "них", "какая",
    "много", "разве", "три", "эту", "моя", "впрочем", "хорошо", "свою", "этой",
    "перед", "иногда", "лучше", "чуть", "том", "нельзя", "такой", "им", "более",
    "всегда", "конечно", "всю", "между", "при", "также", "либо", "причем",
}


class LocalDenseEmbedder:
    """Local CPU ONNX Dense Embedder using FastEmbed (intfloat/multilingual-e5-large)."""

    def __init__(
        self,
        model_name: str = DENSE_MODEL_NAME,
        cache_dir: Path = MODELS_CACHE_DIR,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.is_e5 = "e5" in model_name.lower()
        specific_model_path = os.getenv("DENSE_MODEL_PATH")
        model_kwargs = {
            "model_name": self.model_name,
            "cache_dir": str(self.cache_dir),
        }
        if specific_model_path:
            model_kwargs["specific_model_path"] = specific_model_path
        threads = int(os.getenv("DENSE_THREADS", "0"))
        if threads > 0:
            model_kwargs["threads"] = threads
        self.model = TextEmbedding(**model_kwargs)
        self.dim = (
            1024
            if "large" in model_name.lower() or "bge-m3" in model_name.lower()
            else 768
            if "mpnet" in model_name.lower()
            else 384
        )

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        cache: dict[str, np.ndarray] = {}
        if DENSE_VEC_CACHE_PATH.exists():
            try:
                # Закрываем NpzFile: на Windows открытый read-handle
                # блокирует перезапись этого же файла.
                with np.load(str(DENSE_VEC_CACHE_PATH)) as loaded:
                    for k in loaded.files:
                        cache[k] = loaded[k]
            except Exception:
                pass

        hashes = [
            hashlib.sha256(f"{self.model_name}::{t}".encode("utf-8")).hexdigest()[:24]
            for t in texts
        ]
        missing_indices = [i for i, h in enumerate(hashes) if h not in cache]

        if missing_indices:
            missing_texts = [
                f"passage: {texts[i]}" if self.is_e5 else texts[i]
                for i in missing_indices
            ]
            new_vecs = list(self.model.embed(missing_texts, batch_size=batch_size))
            for idx, v in zip(missing_indices, new_vecs):
                arr = np.asarray(v, dtype=np.float32)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                cache[hashes[idx]] = arr
            DENSE_VEC_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            # Атомарная замена: не перезаписываем npz, пока его кто-то читает,
            # и не оставляем недописанный файл при падении.
            tmp_cache_path = DENSE_VEC_CACHE_PATH.with_suffix(".tmp.npz")
            np.savez_compressed(str(tmp_cache_path), **cache)
            os.replace(tmp_cache_path, DENSE_VEC_CACHE_PATH)

        out: list[list[float]] = [cache[h].tolist() for h in hashes]
        if out:
            self.dim = len(out[0])
        return out

    def embed_query(self, query: str) -> list[float]:
        q_text = f"query: {query}" if self.is_e5 else query
        v = next(iter(self.model.embed([q_text])))
        arr = np.asarray(v, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()


class LocalSparseBM25:
    """Lemmatized Russian BM25 Sparse Vector encoder with pymorphy3 and MIPT glossary."""

    TOKEN_VALID_RE = re.compile(r"^[а-яА-ЯёЁa-zA-Z0-9][а-яА-ЯёЁa-zA-Z0-9_-]*$")

    def __init__(
        self,
        glossary: GlossaryManager | None = None,
        stats_path: Path = BM25_STATS_PATH,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.morph = pymorphy3.MorphAnalyzer()
        self.glossary = glossary or GlossaryManager()
        self.stats_path = stats_path
        self.k1 = k1
        self.b = b
        self._lemma_cache: dict[str, str] = {}
        self.doc_count: int = 1
        self.avg_dl: float = 80.0
        self.doc_freqs: dict[str, int] = {}

        if self.stats_path.exists():
            self.load_stats()

    def _lemmatize_word(self, token: str) -> str:
        low = token.lower().replace("ё", "е")
        if low in self._lemma_cache:
            return self._lemma_cache[low]
        # Keep numbers and order numbers (e.g. 3065-1, 460-1, 700) intact
        if any(ch.isdigit() for ch in low):
            self._lemma_cache[low] = low
            return low
        parsed = self.morph.parse(low)
        lemma = parsed[0].normal_form.replace("ё", "е") if parsed else low
        self._lemma_cache[low] = lemma
        return lemma

    def tokenize_and_lemmatize(self, text: str, expand_glossary: bool = True) -> list[str]:
        work_text = self.glossary.expand_text(text) if expand_glossary else text
        tokens: list[str] = []
        # Also extract hyphenated order numbers like 3065-1, 3699-1, 460-1 explicitly
        for m in re.finditer(r"\b\d{3,4}-\d\b", work_text):
            tokens.append(m.group(0).lower())

        for tok in tokenize(work_text):
            raw = tok.text.strip()
            if not raw or not self.TOKEN_VALID_RE.match(raw):
                continue
            lemma = self._lemmatize_word(raw)
            if len(lemma) < 2 and not lemma.isdigit():
                continue
            if lemma in RUSSIAN_STOPWORDS:
                continue
            tokens.append(lemma)
        return tokens

    @staticmethod
    def _token_to_index(lemma: str) -> int:
        return zlib.crc32(lemma.encode("utf-8")) & 0x7FFFFFFF

    def fit(self, corpus_texts: list[str]) -> list[list[str]]:
        tokenized_docs: list[list[str]] = []
        df_counter: Counter[str] = Counter()
        total_len = 0

        for text in corpus_texts:
            lemmas = self.tokenize_and_lemmatize(text, expand_glossary=True)
            tokenized_docs.append(lemmas)
            total_len += len(lemmas)
            for uniq_lemma in set(lemmas):
                df_counter[uniq_lemma] += 1

        self.doc_count = max(len(tokenized_docs), 1)
        self.avg_dl = max(total_len / self.doc_count, 1.0)
        self.doc_freqs = dict(df_counter)
        self.save_stats()
        return tokenized_docs

    def save_stats(self) -> None:
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        self.stats_path.write_text(
            json.dumps(
                {
                    "doc_count": self.doc_count,
                    "avg_dl": self.avg_dl,
                    "doc_freqs": self.doc_freqs,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def load_stats(self) -> None:
        try:
            data = json.loads(self.stats_path.read_text(encoding="utf-8"))
            self.doc_count = int(data.get("doc_count", 1))
            self.avg_dl = float(data.get("avg_dl", 80.0))
            self.doc_freqs = {str(k): int(v) for k, v in data.get("doc_freqs", {}).items()}
        except Exception:
            pass

    def _idf(self, lemma: str) -> float:
        df = self.doc_freqs.get(lemma, 0)
        return math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))

    def encode_document_lemmas(self, lemmas: list[str]) -> models.SparseVector:
        if not lemmas:
            return models.SparseVector(indices=[], values=[])
        tf_counts = Counter(lemmas)
        dl = len(lemmas)
        idx_to_val: dict[int, float] = {}

        for lemma, tf in tf_counts.items():
            idf = self._idf(lemma)
            denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_dl))
            score = idf * ((tf * (self.k1 + 1.0)) / denom)
            idx = self._token_to_index(lemma)
            idx_to_val[idx] = idx_to_val.get(idx, 0.0) + float(score)

        sorted_items = sorted(idx_to_val.items(), key=lambda x: x[0])
        return models.SparseVector(
            indices=[k for k, _ in sorted_items],
            values=[v for _, v in sorted_items],
        )

    def encode_query(self, query: str) -> models.SparseVector:
        lemmas = self.tokenize_and_lemmatize(query, expand_glossary=True)
        if not lemmas:
            return models.SparseVector(indices=[], values=[])
        tf_counts = Counter(lemmas)
        idx_to_val: dict[int, float] = {}

        for lemma, qtf in tf_counts.items():
            idf = self._idf(lemma)
            idx = self._token_to_index(lemma)
            idx_to_val[idx] = idx_to_val.get(idx, 0.0) + float(idf * (1.0 + 0.5 * (qtf - 1)))

        sorted_items = sorted(idx_to_val.items(), key=lambda x: x[0])
        return models.SparseVector(
            indices=[k for k, _ in sorted_items],
            values=[v for _, v in sorted_items],
        )

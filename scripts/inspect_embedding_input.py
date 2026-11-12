from __future__ import annotations

"""Показать, что именно уходит в dense-модель для чанков.

Примеры:
    python scripts/inspect_embedding_input.py --contains "Утвердить бланки смет"
    python scripts/inspect_embedding_input.py --chunk-id order_794_1_2023::u01_main::s01::c005
    python scripts/inspect_embedding_input.py --stats
    python scripts/inspect_embedding_input.py --stats --tokenizer /path/to/tokenizer.json

Токены считаются токенизатором самой модели (e5-large), а не tiktoken/веб-формы:
у них разная BPE-разметка, поэтому числа могут отличаться.
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CHUNKS_DUMP_PATH


def load_chunks(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_tokenizer(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    model_dir = os.getenv("DENSE_MODEL_PATH")
    if model_dir:
        p = Path(model_dir) / "tokenizer.json"
        if p.exists():
            return p
    # Частый случай: fastembed скачал модель в models_cache
    for p in sorted(Path("models_cache").rglob("tokenizer.json")):
        return p
    return None


def count_tokens(tokenizer_path: Path | None, text: str) -> int | None:
    if tokenizer_path is None:
        return None
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_path))
    return len(tok.encode(text).ids)


def show_chunk(chunk: dict, tokenizer_path: Path | None) -> None:
    dense_input = f"passage: {chunk['embed_text']}"
    n_tokens = count_tokens(tokenizer_path, dense_input)
    print("=" * 100)
    print("chunk_id:", chunk["chunk_id"])
    print(
        f"doc: {chunk.get('doc_title')} | unit: {chunk.get('unit_title')} | "
        f"section {chunk.get('section_no')}: {chunk.get('section_title')} | "
        f"clause: {chunk.get('clause_no')} | zone: {chunk.get('zone')}"
    )
    print(f"raw_text:   {len(chunk['raw_text'])} chars")
    print(f"embed_text: {len(chunk['embed_text'])} chars")
    if n_tokens is None:
        print("tokens:     токенизатор не найден (укажите --tokenizer или DENSE_MODEL_PATH)")
    else:
        print(
            f"tokens:     {n_tokens} (dense input = 'passage: ' + embed_text); "
            f"модельный лимит 512 -> {'ОБРЕЗАНО' if n_tokens > 512 else 'ок'}"
        )
    print("-" * 100)
    print("RAW_TEXT (источник для цитаты/ответа):")
    print(chunk["raw_text"])
    print("-" * 100)
    print("EMBED_TEXT (то, что реально уходит в dense, с префиксом 'passage: '):")
    print(dense_input)


def show_stats(chunks: list[dict], tokenizer_path: Path | None) -> None:
    print(f"chunks: {len(chunks)}")
    if tokenizer_path is None:
        print("Токенизатор не найден: показываю только символы raw/embed.")
        for field in ("raw_text", "embed_text"):
            vals = [len(c[field]) for c in chunks]
            print(
                f"{field}: min={min(vals)} median={statistics.median(vals)} "
                f"p90={sorted(vals)[int(0.9 * len(vals))]} max={max(vals)}"
            )
        return

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_path))
    counts = [len(tok.encode(f"passage: {c['embed_text']}").ids) for c in chunks]
    print(
        "e5 tokens: min={} median={} p90={} max={}".format(
            min(counts),
            statistics.median(counts),
            sorted(counts)[int(0.9 * len(counts))],
            max(counts),
        )
    )
    for limit in (128, 256, 384, 512):
        print(f"  <= {limit} tokens: {sum(1 for n in counts if n <= limit)}")
    over = [c for c, n in zip(chunks, counts) if n > 512]
    print(f"  > 512 tokens (обрезаются моделью): {len(over)}")
    for c in over[:10]:
        print("   ", c["chunk_id"], "|", c["raw_text"][:90].replace("\n", " "))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=str(CHUNKS_DUMP_PATH), help="путь к chunks_dump.json")
    parser.add_argument("--chunk-id", default=None)
    parser.add_argument("--contains", default=None, help="подстрока в raw_text/embed_text")
    parser.add_argument("--limit", type=int, default=3, help="сколько чанков показать")
    parser.add_argument("--stats", action="store_true", help="распределение токенов по всем чанкам")
    parser.add_argument("--tokenizer", default=None, help="путь к tokenizer.json модели")
    args = parser.parse_args()

    chunks = load_chunks(Path(args.json))
    tokenizer_path = find_tokenizer(args.tokenizer)

    if args.stats:
        show_stats(chunks, tokenizer_path)
        return

    if args.chunk_id:
        selected = [c for c in chunks if c["chunk_id"] == args.chunk_id]
    elif args.contains:
        needle = args.contains.lower()
        selected = [
            c
            for c in chunks
            if needle in c["raw_text"].lower() or needle in c["embed_text"].lower()
        ]
    else:
        parser.error("укажите --chunk-id, --contains или --stats")

    if not selected:
        print("Ничего не найдено.")
        return
    for c in selected[: args.limit]:
        show_chunk(c, tokenizer_path)


if __name__ == "__main__":
    main()

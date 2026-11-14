"""Tests for clean repository structure, Docker packaging, and Debian setup per Technical Specification (Criterion 6)."""

from __future__ import annotations

from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_clean_repository_layout_and_categorized_documents() -> None:
    # Extraneous draft/prompt files must not pollute repository root
    forbidden_root_files = [
        "PROMPT_RAG_BUILDER.md",
        "ответ_соннет.txt",
        "анализ_идей_чанкинга.pdf",
        "анализ_идей_чанкинга.tex",
        "Чанкинг и Hybrid Search - анализ (2).docx",
    ]
    for name in forbidden_root_files:
        assert not (REPO_ROOT / name).exists(), f"Extraneous file must not exist in root: {name}"

    # Source corpus is neatly categorized under documents/
    docs_root = REPO_ROOT / "documents"
    assert (docs_root / "regulations").is_dir()
    assert (docs_root / "orders").is_dir()
    assert (docs_root / "appendices_and_forms").is_dir()


def test_dockerignore_excludes_databases_and_regulations() -> None:
    dockerignore_path = REPO_ROOT / ".dockerignore"
    assert dockerignore_path.exists(), ".dockerignore must exist"
    lines = {
        line.strip()
        for line in dockerignore_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    required_exclusions = {
        "documents",
        "documents/",
        "docs",
        "docs/",
        "доки",
        "доки/",
        "data",
        "data/",
        "qdrant_storage",
        "qdrant_storage/",
        ".env",
        "*.pdf",
        "*.docx",
        "*.doc",
        "*.xlsx",
        "*.xls",
    }
    missing = required_exclusions - lines
    assert not missing, f".dockerignore is missing required exclusions: {missing}"


def test_dockerfile_clean_no_baked_db_or_regulations() -> None:
    dockerfile_path = REPO_ROOT / "Dockerfile"
    assert dockerfile_path.exists(), "Dockerfile must exist"
    content = dockerfile_path.read_text(encoding="utf-8")

    copy_lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip().upper().startswith(("COPY ", "ADD "))
    ]
    assert copy_lines, "Dockerfile must contain explicit COPY instructions"

    forbidden_sources = ("data", "documents", "доки", "docs", "qdrant_storage", "registry_overrides.yaml", ".env")
    for line in copy_lines:
        tokens = line.split()
        srcs = tokens[1:-1]
        for src in srcs:
            normalized = src.strip("/").lower()
            assert normalized not in {".", "./"}, (
                f"Dockerfile must not copy entire build context blindly: {line}"
            )
            for forbidden in forbidden_sources:
                assert forbidden.lower() not in normalized, (
                    f"Dockerfile must not bake '{forbidden}' into image: {line}"
                )

    assert "EXPOSE 8000" in content
    assert "HEALTHCHECK" in content
    assert "ENTRYPOINT" in content
    assert "/app/scripts/docker-entrypoint.sh" in content


def test_scripts_exist_and_valid() -> None:
    for script_name in ("docker-entrypoint.sh", "setup_debian_server.sh", "start_local_stack.sh"):
        script_path = REPO_ROOT / "scripts" / script_name
        assert script_path.exists(), f"scripts/{script_name} must exist"
        text = script_path.read_text(encoding="utf-8")
        assert text.startswith("#!/usr/bin/env bash")


def test_docker_compose_reproducible_clean_volumes() -> None:
    compose_path = REPO_ROOT / "docker-compose.yml"
    assert compose_path.exists(), "docker-compose.yml must exist"
    compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))

    assert isinstance(compose, dict)
    services = compose.get("services", {})
    assert {"qdrant", "vllm-qwen38", "rag-app"}.issubset(services.keys())

    named_volumes = compose.get("volumes", {})
    assert {"qdrant_storage", "rag_data", "hf_cache"}.issubset(named_volumes.keys())

    rag_app = services["rag-app"]
    for vol in rag_app.get("volumes", []):
        vol_str = str(vol)
        assert not vol_str.startswith("./data"), f"Host ./data must not be bind-mounted by default: {vol_str}"
        assert "documents" not in vol_str and "доки" not in vol_str, (
            f"Host documents must not be bind-mounted by default: {vol_str}"
        )

    env_files = rag_app.get("env_file", [])
    assert env_files, "rag-app should support optional .env via env_file"
    for item in env_files:
        if isinstance(item, dict):
            assert item.get("required") is False, ".env must have required: false for zero-config clean startup"


def test_readme_compact_and_covers_tz_plus_debian_guide() -> None:
    readme_path = REPO_ROOT / "README.md"
    assert readme_path.exists(), "README.md must exist"
    text = readme_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert len(lines) <= 320, f"README.md should stay compact (got {len(lines)} lines)"

    required_markers = [
        "Debian",
        "nvidia-container-toolkit",
        "setup_debian_server.sh",
        "docker compose up",
        "docker compose down",
        "Dockerfile",
        ".env",
        "PaddlePaddle/PaddleOCR-VL-1.6",
        "BAAI/bge-m3",
        "Qwen/Qwen3.8-27B",
        "Use Case 1",
        "Use Case 2",
    ]
    for marker in required_markers:
        assert marker in text, f"README.md is missing required documentation marker: {marker}"

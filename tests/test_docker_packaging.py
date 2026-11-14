"""Tests for clean Docker & Docker Compose packaging according to Technical Specification (Criterion 6).

Verifies:
  1. Existing databases (data/), regulations (доки/), and office/PDF documents are strictly excluded
     from the Docker build context (.dockerignore) and NEVER copied in Dockerfile.
  2. Dockerfile installs required runtime dependencies, defines HEALTHCHECK, EXPOSE 8000, and entrypoint.
  3. docker-compose.yml defines a reproducible self-contained stack (qdrant, vllm-qwen38, rag-app)
     using clean named volumes without requiring manual code edits or pre-existing .env files.
  4. README.md documents environment requirements, start/stop commands, dependencies, .env variables,
     services list, architecture/data flow, and >= 2 typical user scenarios (Use Cases).
"""

from __future__ import annotations

from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_dockerignore_excludes_databases_and_regulations() -> None:
    dockerignore_path = REPO_ROOT / ".dockerignore"
    assert dockerignore_path.exists(), ".dockerignore must exist"
    lines = {
        line.strip()
        for line in dockerignore_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    required_exclusions = {
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

    forbidden_sources = ("data", "доки", "qdrant_storage", "registry_overrides.yaml", ".env")
    for line in copy_lines:
        # Ensure neither wildcard root COPY (. /app) nor explicit data/regulations COPY is used
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


def test_entrypoint_script_exists_and_executable() -> None:
    entrypoint = REPO_ROOT / "scripts" / "docker-entrypoint.sh"
    assert entrypoint.exists(), "scripts/docker-entrypoint.sh must exist"
    text = entrypoint.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash")
    assert 'exec "$@"' in text


def test_docker_compose_reproducible_clean_volumes() -> None:
    compose_path = REPO_ROOT / "docker-compose.yml"
    assert compose_path.exists(), "docker-compose.yml must exist"
    compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))

    assert isinstance(compose, dict)
    services = compose.get("services", {})
    assert {"qdrant", "vllm-qwen38", "rag-app"}.issubset(services.keys())

    # Named volumes must be used instead of host bind-mounts of ./data or ./доки
    named_volumes = compose.get("volumes", {})
    assert {"qdrant_storage", "rag_data", "hf_cache"}.issubset(named_volumes.keys())

    rag_app = services["rag-app"]
    app_volumes = rag_app.get("volumes", [])
    for vol in app_volumes:
        vol_str = str(vol)
        assert not vol_str.startswith("./data"), f"Host ./data must not be bind-mounted by default: {vol_str}"
        assert "доки" not in vol_str, f"Host ./доки must not be bind-mounted by default: {vol_str}"

    # Clean clone without .env must still boot without docker compose error
    env_files = rag_app.get("env_file", [])
    assert env_files, "rag-app should support optional .env via env_file"
    for item in env_files:
        if isinstance(item, dict):
            assert item.get("required") is False, ".env must have required: false for zero-config clean startup"


def test_readme_covers_criterion_6_sections() -> None:
    readme_path = REPO_ROOT / "README.md"
    assert readme_path.exists(), "README.md must exist"
    text = readme_path.read_text(encoding="utf-8")

    required_markers = [
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

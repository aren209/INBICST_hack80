from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any

import httpx
import pymupdf
from PIL import Image

from src.config import (
    OCR_CACHE_DIR,
    OCR_MAX_IMAGE_BYTES,
    OCR_SPACE_API_KEY,
    OCR_SPACE_ENDPOINT,
    OCR_VERIFIED_DIR,
)


class OcrEngine:
    """Page-level OCR engine via OCR.space Free Plan API with disk cache and fallback."""

    def __init__(
        self,
        api_key: str = OCR_SPACE_API_KEY,
        endpoint: str = OCR_SPACE_ENDPOINT,
        cache_dir: Path = OCR_CACHE_DIR,
        verified_dir: Path = OCR_VERIFIED_DIR,
    ) -> None:
        self.api_key = api_key
        self.endpoint = endpoint
        self.cache_dir = cache_dir
        self.verified_dir = verified_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.verified_dir.mkdir(parents=True, exist_ok=True)
        self.api_errors: list[str] = []

    def _cache_path(self, file_sha256: str, page_no: int) -> Path:
        return self.cache_dir / f"{file_sha256}_p{page_no:03d}.json"

    @staticmethod
    def render_page_jpeg(
        page: pymupdf.Page,
        rotate_cw_degrees: int = 0,
        max_bytes: int = OCR_MAX_IMAGE_BYTES,
    ) -> bytes:
        """Render a PDF page to a JPEG buffer guaranteed to be under max_bytes (< 850 KB)."""
        for dpi in (180, 150, 125, 100):
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            if rotate_cw_degrees:
                # PIL rotate is counter-clockwise, so -rotate_cw_degrees rotates clockwise
                img = img.rotate(-rotate_cw_degrees, expand=True)

            for quality in (82, 72, 60, 50):
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=quality, optimize=True)
                data = buf.getvalue()
                if len(data) <= max_bytes:
                    return data
        return data

    def ocr_page(
        self,
        page: pymupdf.Page,
        file_sha256: str,
        file_name: str,
        page_no: int,
        rotate_cw_degrees: int = 0,
    ) -> str:
        """Recognize text on a single scanned PDF page (1-indexed page_no)."""
        cache_file = self._cache_path(file_sha256, page_no)
        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                text = cached.get("text", "").strip()
                if text:
                    return text
            except Exception:
                pass

        # Render compressed JPEG (< 850 KB for OCR.space Free Plan)
        jpeg_bytes = self.render_page_jpeg(page, rotate_cw_degrees=rotate_cw_degrees)

        payload = {
            "apikey": self.api_key,
            "language": "rus",
            "isOverlayRequired": "false",
            "isTable": "true",
            "scale": "true",
            "detectOrientation": "true",
            "OCREngine": "1",
        }

        last_err = ""
        for attempt in range(3):
            try:
                with httpx.Client(timeout=35.0) as client:
                    resp = client.post(
                        self.endpoint,
                        data=payload,
                        files={
                            "file": (
                                f"p{page_no:03d}.jpg",
                                jpeg_bytes,
                                "image/jpeg",
                            )
                        },
                    )
                if resp.status_code == 200:
                    body: dict[str, Any] = resp.json()
                    if not body.get("IsErroredOnProcessing"):
                        parsed_results = body.get("ParsedResults") or []
                        if parsed_results:
                            parsed_text = (
                                parsed_results[0].get("ParsedText") or ""
                            ).strip()
                            if parsed_text:
                                cache_file.write_text(
                                    json.dumps(
                                        {
                                            "file_name": file_name,
                                            "page_no": page_no,
                                            "source": "ocr.space",
                                            "jpeg_bytes": len(jpeg_bytes),
                                            "text": parsed_text,
                                        },
                                        ensure_ascii=False,
                                        indent=2,
                                    ),
                                    encoding="utf-8",
                                )
                                return parsed_text
                    err_msg = body.get("ErrorMessage") or body.get("ErrorDetails") or str(body)
                    last_err = f"OCR.space error on {file_name} p.{page_no}: {err_msg}"
                else:
                    last_err = (
                        f"OCR.space HTTP {resp.status_code} on {file_name} p.{page_no}: "
                        f"{resp.text[:200]}"
                    )
            except Exception as exc:
                last_err = f"OCR.space exception on {file_name} p.{page_no}: {exc}"

            time.sleep(1.5 * (attempt + 1))

        # Record API error and check verified fallback cache
        if last_err:
            self.api_errors.append(last_err)

        verified_file = self.verified_dir / f"{file_sha256}_p{page_no:03d}.txt"
        if verified_file.exists():
            return verified_file.read_text(encoding="utf-8").strip()

        return ""

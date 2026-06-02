"""CV/Resume parser using multimodal LLM (vision + text).

Supports:
- PDF files (text-based and scanned)
- Image files (PNG, JPG, JPEG, WEBP)

Workflow:
1. Extract content from file (PDF → text/images, Image → raw bytes)
2. Send to LLM with structured prompt for student data extraction
3. Parse JSON response into structured ParsedCVResponse
"""
from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .llm_client import LLMClient


# ── PDF extraction helpers ───────────────────────────────────────────────────

def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF using PyMuPDF. Returns empty string on failure."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        texts = []
        for page in doc:
            texts.append(page.get_text())
        return "\n\n".join(texts)
    except Exception as e:
        print(f"[CVParser] PyMuPDF text extraction failed: {e}", file=sys.stderr, flush=True)
        return ""


def _extract_pdf_as_images(pdf_bytes: bytes) -> list[bytes]:
    """Convert each PDF page to PNG image bytes using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        images = []
        for page in doc:
            pix = page.get_pixmap(dpi=150)
            images.append(pix.tobytes("png"))
        return images
    except Exception as e:
        print(f"[CVParser] PyMuPDF image extraction failed: {e}", file=sys.stderr, flush=True)
        return []


# ── MIME type helpers ────────────────────────────────────────────────────────

def _get_mime_type(filename: str, content_bytes: bytes) -> str:
    """Detect MIME type from filename or magic bytes."""
    if not filename:
        # Magic byte detection
        if content_bytes[:3] == b"%PDF":
            return "application/pdf"
        elif content_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        elif content_bytes[:2] in (b"\xff\xd8",):
            return "image/jpeg"
        elif content_bytes[:4][:3] == b"WBP":
            return "image/webp"
        return "application/octet-stream"

    ext = filename.lower().split(".")[-1]
    ext_map = {
        "pdf": "application/pdf",
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }
    return ext_map.get(ext, "application/octet-stream")


# ── Main parser ─────────────────────────────────────────────────────────────

_STUDENT_CV_PROMPT = """You are an expert CV/resume analyst. Extract student profile data from the provided document (CV, resume, or academic record).

Extract ALL fields below. If a field is not found in the document, use null for strings/numbers and empty arrays/lists for collections. Return ONLY valid JSON — no markdown, no backticks, no explanations.

Required JSON structure:
{
  "personal": {
    "full_name": string or null,
    "gender": string or null (e.g., "Female", "Male"),
    "date_of_birth": string in YYYY-MM-DD format or null,
    "province": string or null (Indonesian province if applicable),
    "economic_background": string or null (e.g., "Low Income", "Middle Income", "Upper Middle Income", "High Income"),
    "from_underrepresented_region": boolean or null
  },
  "academic": {
    "school_level": string or null (e.g., "SMA", "SMK", "MA", "University", "Bachelor", "Master"),
    "major_program": string or null (e.g., "IPA", "IPS", "Computer Science"),
    "grade_class": string or null (e.g., "Grade 12", "Semester 5", "Year 3"),
    "school_name": string or null,
    "school_tier_accreditation": string or null (e.g., "Public School - Accredited A", "Accredited A"),
    "expected_graduation_year": integer or null,
    "average_grade": number or null (scale 0-100),
    "math_score": number or null (scale 0-100),
    "english_score": number or null (scale 0-100),
    "major_subject_average": number or null (scale 0-100),
    "extracurricular_achievements": string or null,
    "olympiad_level": string or null (e.g., "City / District", "Provincial", "National", "International"),
    "intended_career_track": string or null (e.g., "Industry / Tech", "Academic / Research", "Public Service"),
    "willing_to_return_home": boolean or null,
    "needs_full_funding": boolean or null
  },
  "skills": {
    "hard_skills": [string],
    "soft_skills": [string],
    "languages": [string],
    "language_certificates": [
      {"test_type": string, "score": number, "valid_until": string}
    ],
    "target_countries": [string]
  }
}

Rules:
- For academic scores, look for GPA, average grade, subject scores on a scale of 0-100
- olympiad_level should be one of: None, School Level, City / District, Provincial, National, International
- economic_background should reflect the family income level mentioned (or null if not stated)
- from_underrepresented_region is true only if explicitly mentioned (e.g., 3T region, underprivileged area)
- Extract ALL skills listed — do not limit to a few examples
- For language certificates, extract test type (IELTS, TOEFL, etc.), score, and expiry date
- If the document contains both personal CV info AND academic records, combine them into one profile
"""


def _extract_json_from_text(text: str) -> Optional[dict[str, Any]]:
    """Extract JSON from LLM response text."""
    cleaned = text.strip()

    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        json_lines = []
        inside_block = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                inside_block = not inside_block
                continue
            if inside_block or not any(c.isalpha() for c in stripped[:1]):
                json_lines.append(line)
        cleaned = "\n".join(json_lines).strip()

    # Try finding JSON object boundaries
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            pass

    # Try parsing the whole text as JSON
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    return None


def parse_cv(
    llm_client: "LLMClient",
    file_bytes: bytes,
    filename: str = "",
) -> Optional[dict[str, Any]]:
    """Parse a CV/resume file and extract student profile data.

    Args:
        llm_client: LLMClient instance with valid configuration.
        file_bytes: Raw file content (PDF or image).
        filename: Original filename for MIME detection.

    Returns:
        Parsed student profile dict, or None on failure.
    """
    if not llm_client.is_available:
        print("[CVParser] Skipping — LLM is unavailable", flush=True)
        return None

    mime_type = _get_mime_type(filename, file_bytes)

    # Handle PDF
    if mime_type == "application/pdf":
        pdf_text = _extract_pdf_text(file_bytes)
        if pdf_text and len(pdf_text.strip()) > 20:
            # Text-based PDF — send text to LLM
            try:
                response = llm_client._call_with_retry(
                    f"{_STUDENT_CV_PROMPT}\n\n--- CV TEXT ---\n{pdf_text}"
                )
                return _extract_json_from_text(response) if response else None
            except Exception as e:
                print(f"[CVParser] PDF text parsing failed: {e}", flush=True)
        else:
            # Scanned PDF — convert to images
            pdf_images = _extract_pdf_as_images(file_bytes)
            if pdf_images:
                try:
                    response = llm_client._call_with_pdf_images(
                        file_bytes, mime_type, _STUDENT_CV_PROMPT
                    )
                    return _extract_json_from_text(response) if response else None
                except Exception as e:
                    print(f"[CVParser] PDF image parsing failed: {e}", flush=True)

    # Handle images (PNG, JPG, WEBP)
    elif mime_type.startswith("image/"):
        try:
            response = llm_client._call_with_pdf_images(
                file_bytes, mime_type, _STUDENT_CV_PROMPT
            )
            return _extract_json_from_text(response) if response else None
        except Exception as e:
            print(f"[CVParser] Image parsing failed: {e}", flush=True)

    return None
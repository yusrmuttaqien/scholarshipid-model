"""LLM client for generating recommendation text and fit scores.

Uses llama.cpp via OpenAI-compatible API to produce:
- Personalized scholarship recommendations
- Fit scores (academic, leadership, language alignment)

Uses temperature=0 for deterministic output on identical inputs.

If LLM_API_KEY is not set, the client treats itself as unavailable
and all methods return empty/None gracefully.
"""
from __future__ import annotations

import http.client
import json
import os
import random
import time
from typing import TYPE_CHECKING, Any, Optional

# Runtime imports for classes used in method bodies
try:
    from openai import OpenAI as OAIClient  # type: ignore[import-untyped]
except ImportError:
    class _Stub:  # type: ignore
        def __init__(self, *a, **kw): pass
    OAIClient = _Stub


class LLMClient:
    """Thin wrapper around llama.cpp via OpenAI-compatible API.

    If LLM_API_KEY is not set in the environment, is_available returns False
    and all methods return empty/None so the /recommend endpoint still works
    with scores but without text recommendations.
    """

    def __init__(self, cfg: dict) -> None:
        self._base_url_env = cfg.get("llm", {}).get("base_url_env", "LLM_BASE_URL")
        self._model_env = cfg.get("llm", {}).get("model_env", "LLM_MODEL")
        self._api_key_env = cfg.get("llm", {}).get("api_key_env", "LLM_API_KEY")
        self._temperature = cfg.get("llm", {}).get("temperature", 0.0)
        self._max_tokens = cfg.get("llm", {}).get("max_tokens", 500)

        # Retry configuration
        self._max_retries = cfg.get("llm", {}).get("max_retries", 2)
        self._retry_base_delay = cfg.get("llm", {}).get("retry_base_delay", 0.5)
        self._retry_max_delay = cfg.get("llm", {}).get("retry_max_delay", 5.0)

        # Lazy-load the client (avoids import error if openai is not installed)
        self._client: Optional["OAIClient"] = None

    @property
    def is_available(self) -> bool:
        """Check if LLM API is configured with a base URL and API key."""
        api_key = os.environ.get(self._api_key_env, "")
        base_url = os.environ.get(self._base_url_env, "")
        return len(api_key.strip()) > 0 and len(base_url.strip()) > 0

    def is_reachable(self) -> bool:
        """Check if the LLM API is configured AND reachable.

        Returns True only when:
        1. base_url env var is set (configured)
        2. The base URL responds to a lightweight HTTP HEAD request

        Returns False if configuration is missing or the server is unreachable.
        Uses a short timeout to avoid blocking the health check.
        """
        api_key = os.environ.get(self._api_key_env, "")
        base_url = os.environ.get(self._base_url_env, "")

        # Not configured — cannot be reachable
        if not api_key.strip() or not base_url.strip():
            return False

        # Parse the base URL for a direct HTTP connection (no OpenAI client needed)
        try:
            from urllib.parse import urlparse
            parsed = urlparse(base_url)
            host = parsed.hostname or ""
            port = parsed.port
            scheme = parsed.scheme or "http"

            if not host:
                return False

            # Use the parsed port, defaulting to 80/443 based on scheme
            if port is None:
                port = 443 if scheme == "https" else 80

            # Connect with a short timeout (2 seconds)
            connect_timeout = 2
            if scheme == "https":
                import ssl
                context = ssl.create_default_context()
                conn = http.client.HTTPSConnection(host, port, timeout=connect_timeout, context=context)
            else:
                conn = http.client.HTTPConnection(host, port, timeout=connect_timeout)

            # Send a lightweight HEAD request to the base path
            path = parsed.path or "/"
            conn.request("HEAD", path)
            resp = conn.getresponse()
            conn.close()

            # Any 2xx/3xx response means the server is reachable
            return resp.status < 400

        except Exception:
            return False

    # ── Retry helpers ───────────────────────────────────────────────────────

    def _should_retry(self, e: Exception) -> bool:
        """Determine if an exception warrants a retry."""
        error_str = str(e).lower()
        retryable_keywords = [
            "rate limit", "429", "request entity too large", "timeout",
            "internal error", "500", "503", "service unavailable",
            "gateway", "overloaded", "backpressure",
            "connection refused", "connection reset", "no connection",
            "econnrefused", "econnreset", "tunnel",
        ]
        return any(kw in error_str for kw in retryable_keywords)

    def _get_client(self) -> "OAIClient":
        """Lazy-initialize the OpenAI-compatible client."""
        if self._client is None:
            base_url = os.environ[self._base_url_env]
            api_key = os.environ[self._api_key_env]
            model = os.environ.get(self._model_env, "")
            self._client = OAIClient(
                base_url=f"{base_url}/v1",
                api_key=api_key or "sk-no-key-required",
            )
        return self._client

    def _call_with_retry(self, prompt: str) -> str:
        """Call the LLM API with exponential backoff retry logic.

        Returns the model's text response or empty string on failure.
        """
        if not self.is_available:
            print("[LLM] Skipping — is_available=False (check env vars)", flush=True)
            return ""

        last_error: Optional[Exception] = None
        delay = self._retry_base_delay

        for attempt in range(self._max_retries):
            try:
                client = self._get_client()
                model = os.environ.get(self._model_env, "qwen3-4b")
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
                content = (response.choices[0].message.content or "").strip()
                if not content:
                    print(f"[LLM] Empty response on attempt {attempt + 1}", flush=True)
                return content

            except Exception as e:
                last_error = e
                print(f"[LLM] Error (attempt {attempt + 1}/{self._max_retries}): {e}", flush=True)
                if not self._should_retry(e):
                    break
                if attempt < self._max_retries - 1:
                    jitter = random.uniform(0.5, 1.5) * delay
                    time.sleep(min(jitter, self._retry_max_delay))
                    delay *= 2

        print(f"[LLM] All {self._max_retries} retries exhausted: {last_error}", flush=True)
        return ""

    # ── Public API ────────────────────────────────────────────────────────

    def generate_recommendation(
        self, student_data: dict, scholarship_metadata: dict
    ) -> str:
        """Generate a personalized HTML recommendation for the student-scholarship match.

        Uses fit scores to determine tone:
        - HIGH MATCH (avg > 0.7): Congratulations message with alignment highlights
        - MEDIUM MATCH (0.4-0.7): Acknowledge strengths, suggest improvements
        - LOW MATCH (< 0.4): Honest assessment with encouragement and action items

        Returns empty string if LLM is unavailable.
        """
        if not self.is_available:
            return ""

        # Build conditional fields to avoid nested f-string syntax errors
        personal_statement = student_data.get('personal_statement', '')
        future_goals = student_data.get('future_goals', '')
        extra_fields = ""
        if personal_statement:
            extra_fields += f"- Personal Statement: {personal_statement}\n"
        if future_goals:
            extra_fields += f"- Future Goals: {future_goals}\n"

        prompt = (
            f"You are an expert scholarship counselor.\n\n"
            f"## Student Profile\n"
            f"- Nationality: {student_data.get('nationality', 'N/A')}\n"
            f"- Age: {student_data.get('age', 'N/A')}\n"
            f"- Degree Level: {student_data.get('target_degree_level', 'N/A')}\n"
            f"- Overall GPA: {student_data.get('overall_report_card_average', 'N/A')}/100\n"
            f"- Math Score: {student_data.get('math_score', 'N/A')}/100\n"
            f"- English Score: {student_data.get('english_score', 'N/A')}/100\n"
            f"- High School Track: {student_data.get('high_school_track', 'N/A')}\n"
            f"- Intended Career: {student_data.get('intended_career_track', 'N/A')}\n"
            f"- Leadership Experience: {student_data.get('leadership_experience_count', 0)} roles\n"
            f"- Olympiad Achievements: {student_data.get('olympiad_subjects', [])}\n"
            f"- Volunteer Work: {student_data.get('volunteer_experience_count', 0)} activities\n"
            f"- Language Proficiency: {student_data.get('language_proficiency', 'N/A')}\n"
            f"- Key Achievements: {student_data.get('achievements_narrative', 'N/A')}\n"
            f"{extra_fields}"
            f"\n## Scholarship\n"
            f"- Name: {scholarship_metadata.get('name', 'N/A')}\n"
            f"- Mission: {scholarship_metadata.get('mission_statement', 'N/A')}\n"
            f"- Selection Criteria: {scholarship_metadata.get('selection_criteria', 'N/A')}\n"
            f"- Host Country: {scholarship_metadata.get('host_country', 'N/A')}\n"
            f"- Funding: {scholarship_metadata.get('funding_coverage_summary', 'N/A')}\n"
            f"- Language Requirements: {scholarship_metadata.get('language_requirements', 'N/A')}\n"
            f"- Target Recipients: {scholarship_metadata.get('target_recipient_profile', 'N/A')}\n\n"
            f"## Your Task\n"
            f"Output a concise AI Optimization card in HTML format.\n\n"
            f"<h3>Improve Your Match</h3>\n"
            f"<ul>\n"
            f"  <li>Action item 1 — specific and under 60 chars.</li>\n"
            f"  <li>Action item 2 — specific and under 60 chars.</li>\n"
            f"  <li>Action item 3 — specific and under 60 chars.</li>\n"
            f"</ul>\n\n"
            f"Rules:\n"
            f"- Be specific — mention exact fields, scores, or actions from the student's profile\n"
            f"- Keep each bullet under 60 characters\n"
            f"- Never write paragraphs\n"
            f"- Output ONLY valid HTML using <h3>, <p>, <ul>, <li> tags. No markdown, no backticks."
        )

        return self._call_with_retry(prompt)

    def _call_with_pdf_images(
        self, file_bytes: bytes, mime_type: str, prompt: str
    ) -> str:
        """Call the LLM with a PDF or image file (vision/multimodal).

        For images: converts to base64 and sends as data:image/*;base64,...
        For PDFs: delegates to _call_with_pdf_pages which renders pages as PNG.
        """
        import base64

        if not self.is_available:
            return ""

        # PDFs must be converted to images (LLM vision only accepts data:image/*)
        if mime_type == "application/pdf":
            return self._call_with_pdf_pages(file_bytes, prompt)

        b64 = base64.b64encode(file_bytes).decode("ascii")
        data_uri = f"data:image/{mime_type.split('/')[-1]};base64,{b64}"

        # Build a combined prompt with the image instruction
        content_parts = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]

        last_error: Optional[Exception] = None
        delay = self._retry_base_delay

        for attempt in range(self._max_retries):
            try:
                client = self._get_client()
                model = os.environ.get(self._model_env, "qwen3-4b")
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": content_parts}],
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
                text = (response.choices[0].message.content or "").strip()
                if not text:
                    print(f"[LLM] Empty response on attempt {attempt + 1}", flush=True)
                return text

            except Exception as e:
                last_error = e
                print(f"[LLM] Error (attempt {attempt + 1}/{self._max_retries}): {e}", flush=True)
                if not self._should_retry(e):
                    break
                if attempt < self._max_retries - 1:
                    jitter = random.uniform(0.5, 1.5) * delay
                    time.sleep(min(jitter, self._retry_max_delay))
                    delay *= 2

        print(f"[LLM] All {self._max_retries} retries exhausted: {last_error}", flush=True)
        return ""

    def _call_with_pdf_pages(
        self, pdf_bytes: bytes, prompt: str
    ) -> str:
        """Convert PDF pages to PNG images and send them as a multimodal message.

        Uses PyMuPDF (fitz) to render each page at 2x DPI for good text clarity.
        All pages are sent together in one API call so the LLM can read the full document.
        """
        import base64
        try:
            import fitz  # PyMuPDF
        except ImportError:
            print("[LLM] PyMuPDF (fitz) not installed — cannot parse PDF pages", flush=True)
            return ""

        if not self.is_available:
            return ""

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        image_parts: list[dict[str, Any]] = []
        for page_num, page in enumerate(doc):
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for clarity
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            b64 = base64.b64encode(img_bytes).decode("ascii")
            image_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        doc.close()

        # Build content: text prompt first, then all page images
        content_parts: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
        ]
        content_parts.extend(image_parts)

        last_error: Optional[Exception] = None
        delay = self._retry_base_delay

        for attempt in range(self._max_retries):
            try:
                client = self._get_client()
                model = os.environ.get(self._model_env, "qwen3-4b")
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": content_parts}],
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
                text = (response.choices[0].message.content or "").strip()
                if not text:
                    print(f"[LLM] Empty response on attempt {attempt + 1}", flush=True)
                return text

            except Exception as e:
                last_error = e
                print(f"[LLM] Error (attempt {attempt + 1}/{self._max_retries}): {e}", flush=True)
                if not self._should_retry(e):
                    break
                if attempt < self._max_retries - 1:
                    jitter = random.uniform(0.5, 1.5) * delay
                    time.sleep(min(jitter, self._retry_max_delay))
                    delay *= 2

        print(f"[LLM] All {self._max_retries} retries exhausted: {last_error}", flush=True)
        return ""

    def _extract_json(self, text: str) -> Optional[dict[str, Any]]:
        """Extract and parse a JSON object from model response text.

        Handles common formats: raw JSON, markdown-wrapped JSON (```json ... ```),
        and JSON embedded in surrounding prose.
        """
        if not text:
            return None

        # Strip markdown code fences if present
        cleaned = text.strip()
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

    def _extract_json_array(self, text: str) -> Optional[list[dict[str, Any]]]:
        """Extract and parse a JSON array from model response text.

        Handles common formats: raw JSON array, markdown-wrapped JSON array (```json ... ```),
        and JSON array embedded in surrounding prose.
        """
        if not text:
            return None

        # Strip markdown code fences if present
        cleaned = text.strip()
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

        # Try finding JSON array boundaries
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start >= 0 and end > start:
            try:
                result = json.loads(cleaned[start:end + 1])
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        # Try parsing the whole text as JSON array
        try:
            result = json.loads(cleaned)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        return None

    def compute_fit_scores(
        self, student_data: dict, scholarship_metadata: dict
    ) -> Optional[dict[str, Any]]:
        """Compute fit scores for academic, leadership, and language alignment.

        Returns a dict like:
        {
            "academic": 0.85,
            "leadership": 0.65,
            "language": 0.72,
        }

        Returns None if LLM is unavailable or parsing fails.
        """
        if not self.is_available:
            return None

        # Build conditional fields to avoid nested f-string syntax errors
        score = student_data.get('overall_report_card_average', 'N/A')
        math_s = student_data.get('math_score', 'N/A')
        eng_s = student_data.get('english_score', 'N/A')
        toefl = student_data.get('toefl_score', 0)
        ielts = student_data.get('ielts_score', 0)
        leadership_n = student_data.get('leadership_experience_count', 0)
        olympiad_s = student_data.get('olympiad_subjects', [])
        volunteer_n = student_data.get('volunteer_experience_count', 0)
        wins_n = student_data.get('competition_wins_count', 0)

        toefl_str = f"- TOEFL: {toefl}\n" if toefl > 0 else ""
        ielts_str = f"- IELTS: {ielts}\n" if ielts > 0 else ""
        full_fund = scholarship_metadata.get('funding_is_full_funding', False)

        prompt = (
            f"## Student Profile\n"
            f"- Overall GPA: {score}/100\n"
            f"- Math Score: {math_s}/100\n"
            f"- English Score: {eng_s}/100\n"
            f"{toefl_str}"
            f"{ielts_str}"
            f"- Leadership Experience: {leadership_n} roles\n"
            f"- Olympiad Subjects: {olympiad_s}\n"
            f"- Volunteer Hours (divided by 50): {volunteer_n}\n"
            f"- Competition Wins: {wins_n}\n\n"
            f"## Scholarship\n"
            f"- Name: {scholarship_metadata.get('name', 'N/A')}\n"
            f"- Selection Criteria: {scholarship_metadata.get('selection_criteria', 'N/A')}\n"
            f"- Funding: {full_fund}\n\n"
            f"Output ONLY a JSON object with these three keys (values 0.0-1.0):\n"
            f"{{\"academic\": <score>, \"leadership\": <score>, \"language\": <score>}}"
        )

        text = self._call_with_retry(prompt)
        return self._extract_json(text)

    def generate_recommendation_with_fit_scores(
        self, student_data: dict, scholarship_metadata: dict
    ) -> tuple[Optional[dict[str, Any]], str]:
        """Generate fit scores AND recommendation in a single LLM call.

        Returns a combined JSON response with both fit scores and the HTML
        recommendation text in one round-trip to llama.cpp. This is significantly
        faster than calling compute_fit_scores() + generate_recommendation() separately,
        which is critical when running on a single-slot llama.cpp server.

        Returns:
            (fit_scores_dict, recommendation_text) — either can be None/"" on failure.
        """
        if not self.is_available:
            return None, ""

        # Build conditional fields to avoid nested f-string syntax errors
        personal_statement = student_data.get('personal_statement', '')
        future_goals = student_data.get('future_goals', '')
        extra_fields = ""
        if personal_statement:
            extra_fields += f"- Personal Statement: {personal_statement}\n"
        if future_goals:
            extra_fields += f"- Future Goals: {future_goals}\n"

        prompt = (
            f"You are an expert scholarship counselor.\n\n"
            f"## Student Profile\n"
            f"- Nationality: {student_data.get('nationality', 'N/A')}\n"
            f"- Age: {student_data.get('age', 'N/A')}\n"
            f"- Degree Level: {student_data.get('target_degree_level', 'N/A')}\n"
            f"- Overall GPA: {student_data.get('overall_report_card_average', 'N/A')}/100\n"
            f"- Math Score: {student_data.get('math_score', 'N/A')}/100\n"
            f"- English Score: {student_data.get('english_score', 'N/A')}/100\n"
            f"- High School Track: {student_data.get('high_school_track', 'N/A')}\n"
            f"- Intended Career: {student_data.get('intended_career_track', 'N/A')}\n"
            f"- Leadership Experience: {student_data.get('leadership_experience_count', 0)} roles\n"
            f"- Olympiad Achievements: {student_data.get('olympiad_subjects', [])}\n"
            f"- Volunteer Work: {student_data.get('volunteer_experience_count', 0)} activities\n"
            f"- Language Proficiency: {student_data.get('language_proficiency', 'N/A')}\n"
            f"- Key Achievements: {student_data.get('achievements_narrative', 'N/A')}\n"
            f"{extra_fields}"
            f"\n## Scholarship\n"
            f"- Name: {scholarship_metadata.get('name', 'N/A')}\n"
            f"- Mission: {scholarship_metadata.get('mission_statement', 'N/A')}\n"
            f"- Selection Criteria: {scholarship_metadata.get('selection_criteria', 'N/A')}\n"
            f"- Host Country: {scholarship_metadata.get('host_country', 'N/A')}\n"
            f"- Funding: {scholarship_metadata.get('funding_coverage_summary', 'N/A')}\n"
            f"- Language Requirements: {scholarship_metadata.get('language_requirements', 'N/A')}\n"
            f"- Target Recipients: {scholarship_metadata.get('target_recipient_profile', 'N/A')}\n\n"
            f"## Your Task\n"
            f"Respond with a SINGLE JSON object containing two keys:\n\n"
            f"1. `fit_scores`: an object with three keys (academic, leadership, language) — each value between 0.0 and 1.0\n"
            f"2. `recommendation`: a short HTML string (use <h3>, <p>, <ul>, <li> tags)\n\n"
            f"For the recommendation:\n"
            f"<h3>Improve Your Match</h3>\n"
            f"<ul>\n"
            f"  <li>Action item 1 — specific and under 60 chars.</li>\n"
            f"  <li>Action item 2 — specific and under 60 chars.</li>\n"
            f"  <li>Action item 3 — specific and under 60 chars.</li>\n"
            f"</ul>\n\n"
        )

        prompt += (
            f"Rules:\n"
            f"- Be specific — mention exact fields, scores, or actions from the student's profile\n"
            f"- Keep each bullet under 60 characters\n"
            f"- Never write paragraphs in recommendation\n"
            f"- Output ONLY a valid JSON object. No markdown, no backticks, no prose.\n"
        )

        text = self._call_with_retry(prompt)
        if not text:
            return None, ""

        # Parse the combined response
        data = self._extract_json(text)
        if data is None:
            print(f"[LLM] Failed to parse combined response", flush=True)
            return None, ""

        fit_scores = data.get("fit_scores")
        recommendation = data.get("recommendation", "")

        return fit_scores, recommendation

    def generate_batch_recommendation(
        self, student_data: dict, scholarship_metadata_list: list[dict]
    ) -> tuple[list[Optional[dict[str, Any]]], list[str]]:
        """Generate fit scores AND recommendations for multiple scholarships in one LLM call.

        Sends a single prompt with all scholarships and receives back a JSON array where
        each element contains `fit_scores` and `recommendation`. This dramatically reduces
        the number of HTTP round-trips to llama.cpp, which is critical when running on
        a single-slot server.

        Args:
            student_data: Student profile dict (same format as individual methods).
            scholarship_metadata_list: List of scholarship metadata dicts.

        Returns:
            (fit_scores_list, recommendation_list) — each element can be None/"" on failure.
        """
        if not self.is_available or not scholarship_metadata_list:
            empty_scores = [None] * len(scholarship_metadata_list)
            return empty_scores, [""] * len(scholarship_metadata_list)

        # Build conditional fields to avoid nested f-string syntax errors
        personal_statement = student_data.get('personal_statement', '')
        future_goals = student_data.get('future_goals', '')
        extra_fields = ""
        if personal_statement:
            extra_fields += f"- Personal Statement: {personal_statement}\n"
        if future_goals:
            extra_fields += f"- Future Goals: {future_goals}\n"

        # Format all scholarships for the prompt
        schols_str = ""
        for i, meta in enumerate(scholarship_metadata_list, start=1):
            schols_str += (
                f"\n### Scholarship {i}: {meta.get('name', 'N/A')}\n"
                f"- ID: {meta.get('scholarship_id', 'N/A')}\n"
                f"- Mission: {meta.get('mission_statement', 'N/A')}\n"
                f"- Selection Criteria: {meta.get('selection_criteria', 'N/A')}\n"
                f"- Host Country: {meta.get('host_country', 'N/A')}\n"
                f"- Funding: {meta.get('funding_coverage_summary', 'N/A')}\n"
                f"- Language Requirements: {meta.get('language_requirements', 'N/A')}\n"
                f"- Target Recipients: {meta.get('target_recipient_profile', 'N/A')}\n"
            )

        prompt = (
            f"You are an expert scholarship counselor.\n\n"
            f"## Student Profile\n"
            f"- Nationality: {student_data.get('nationality', 'N/A')}\n"
            f"- Age: {student_data.get('age', 'N/A')}\n"
            f"- Degree Level: {student_data.get('target_degree_level', 'N/A')}\n"
            f"- Overall GPA: {student_data.get('overall_report_card_average', 'N/A')}/100\n"
            f"- Math Score: {student_data.get('math_score', 'N/A')}/100\n"
            f"- English Score: {student_data.get('english_score', 'N/A')}/100\n"
            f"- High School Track: {student_data.get('high_school_track', 'N/A')}\n"
            f"- Intended Career: {student_data.get('intended_career_track', 'N/A')}\n"
            f"- Leadership Experience: {student_data.get('leadership_experience_count', 0)} roles\n"
            f"- Olympiad Achievements: {student_data.get('olympiad_subjects', [])}\n"
            f"- Volunteer Work: {student_data.get('volunteer_experience_count', 0)} activities\n"
            f"- Language Proficiency: {student_data.get('language_proficiency', 'N/A')}\n"
            f"- Key Achievements: {student_data.get('achievements_narrative', 'N/A')}\n"
            f"{extra_fields}"
            f"\n## Scholarships to Evaluate\n"
            f"{schols_str}"
            f"\n## Your Task\n"
            f"Output a JSON array with one object per scholarship. Each object must have:\n"
            f"1. `scholarship_id` (string)\n"
            f"2. `fit_scores`: an object with three keys (academic, leadership, language) — each value between 0.0 and 1.0\n"
            f"3. `recommendation`: a short HTML string (use <h3>, <p>, <ul>, <li> tags)\n\n"
        )

        prompt += (
            f"For the recommendation:\n"
            f"<h3>Improve Your Match</h3>\n"
            f"<ul>\n"
            f"  <li>Action item 1 — specific and under 60 chars.</li>\n"
            f"  <li>Action item 2 — specific and under 60 chars.</li>\n"
            f"  <li>Action item 3 — specific and under 60 chars.</li>\n"
            f"</ul>\n\n"
        )

        prompt += (
            f"Rules:\n"
            f"- Be specific — mention exact fields, scores, or actions from the student's profile\n"
            f"- Keep each bullet under 60 characters\n"
            f"- Never write paragraphs in recommendation\n"
            f"- Output ONLY a valid JSON array. No markdown, no backticks, no prose.\n"
        )

        print(prompt)
        text = self._call_with_retry(prompt)
        print(text)
        if not text:
            empty_scores = [None] * len(scholarship_metadata_list)
            return empty_scores, [""] * len(scholarship_metadata_list)

        # Parse the batch response
        results = self._extract_json_array(text)
        if results is None:
            print(f"[LLM] Failed to parse batch response", flush=True)
            empty_scores = [None] * len(scholarship_metadata_list)
            return empty_scores, [""] * len(scholarship_metadata_list)

        # Build lookup by scholarship_id for matching results back to input order
        id_to_result: dict[str, dict] = {}
        for item in results:
            sid = item.get("scholarship_id")
            if sid:
                id_to_result[sid] = item

        fit_scores_list = []
        recommendation_list = []
        for meta in scholarship_metadata_list:
            sid = meta.get("scholarship_id", "")
            result = id_to_result.get(sid, {})
            fit_scores_list.append(result.get("fit_scores"))
            recommendation_list.append(result.get("recommendation", ""))

        return fit_scores_list, recommendation_list

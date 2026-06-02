"""LLM client for generating recommendation text and fit scores.

Uses llama.cpp via OpenAI-compatible API to produce:
- Personalized scholarship recommendations
- Fit scores (academic, leadership, language alignment)

Uses temperature=0 for deterministic output on identical inputs.

If LLM_API_KEY is not set, the client treats itself as unavailable
and all methods return empty/None gracefully.
"""
from __future__ import annotations

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

        # Include fit scores so the LLM can reason about match quality
        fit_scores = student_data.get('_fit_scores', None)
        if fit_scores:
            academic = fit_scores.get('academic', '')
            leadership = fit_scores.get('leadership', '')
            language = fit_scores.get('language', '')
            scores_section = (
                f"## Fit Scores\n"
                f"- Academic Alignment: {academic}\n"
                f"- Leadership Alignment: {leadership}\n"
                f"- Language Alignment: {language}\n"
            )
        else:
            scores_section = ""

        # Calculate match percentage for the header
        academic_score = fit_scores.get('academic', 0) if fit_scores else 0
        match_pct = int(academic_score * 100)
        target_pct = min(98, match_pct + 5)

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
            f"{extra_fields}"
            f"\n## Scholarship\n"
            f"- Name: {scholarship_metadata.get('name', 'N/A')}\n"
            f"- Mission: {scholarship_metadata.get('mission_statement', 'N/A')}\n"
            f"- Selection Criteria: {scholarship_metadata.get('selection_criteria', 'N/A')}\n"
            f"- Host Country: {scholarship_metadata.get('host_country', 'N/A')}\n"
            f"- Funding: {scholarship_metadata.get('funding_coverage_summary', 'N/A')}\n"
            f"{scores_section}"
            f"\n## Your Task\n"
            f"Output a concise AI Optimization card in HTML format.\n\n"
            f"For matches below 95%:\n"
            f"<h3>To increase your {match_pct}% Match to {target_pct}%</h3>\n"
            f"<ul>\n"
            f"  <li>Action item 1 — specific and under 60 chars.</li>\n"
            f"  <li>Action item 2 — specific and under 60 chars.</li>\n"
            f"  <li>Action item 3 — specific and under 60 chars.</li>\n"
            f"</ul>\n\n"
            f"For matches at or above 95%:\n"
            f"<h3>Strong Match! Consider Applying Now</h3>\n"
            f"<p>Your profile aligns exceptionally well with this scholarship.</p>\n\n"
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

        Converts the file to base64 and sends it as an image content part
        alongside the text prompt.
        """
        import base64

        if not self.is_available:
            return ""

        b64 = base64.b64encode(file_bytes).decode("ascii")
        data_uri = f"data:{mime_type};base64,{b64}"

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
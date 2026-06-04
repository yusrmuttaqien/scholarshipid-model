"""Shared helper functions for the serving layer."""

from io import StringIO
from typing import Optional

import pandas as pd

from src.utils.feature_engineering import normalize_json_columns as _normalize_json_columns
from .config import STUDENT_JSON_COLS, SCHOLARSHIP_JSON_COLS


def student_profile_to_csv_schema(student_data: dict) -> dict:
    """Convert flat API student profile to CSV-compatible schema for encode_student.

    The API accepts flat fields (e.g. toefl_score=65), but encode_student expects
    CSV-compatible structures (e.g. language_proficiency=[{"test_type":"toefl","score":65}]).
    """
    row = dict(student_data)

    # Build language_proficiency as list of dicts (expected by encode_student)
    lang_prof = []
    if student_data.get("toefl_score", 0) > 0:
        lang_prof.append({"test_type": "toefl", "score": student_data["toefl_score"]})
    if student_data.get("ielts_score", 0) > 0:
        lang_prof.append({"test_type": "ielts", "score": student_data["ielts_score"]})
    row["language_proficiency"] = lang_prof

    # olympiad_subjects: convert comma-separated string to list if needed
    olympiad_subjects_raw = student_data.get("olympiad_subjects") or []
    if isinstance(olympiad_subjects_raw, str) and olympiad_subjects_raw:
        row["olympiad_subjects"] = [s.strip() for s in olympiad_subjects_raw.split(",")]
    else:
        row["olympiad_subjects"] = olympiad_subjects_raw

    # target_countries: convert comma-separated string to list if needed
    target_countries_raw = student_data.get("target_countries")
    if isinstance(target_countries_raw, str) and target_countries_raw:
        row["target_countries"] = [c.strip() for c in target_countries_raw.split(",")]
    else:
        row["target_countries"] = target_countries_raw

    return row


def _parse_csv_with_json(csv_text: str, json_cols: list[str]) -> pd.DataFrame:
    """Parse CSV text and decode JSON columns back to proper Python types.

    Args:
        csv_text: Raw CSV string content.
        json_cols: List of column names that contain JSON-encoded values.

    Returns:
        DataFrame with JSON columns decoded into proper Python types (lists, dicts).
    """
    df = pd.read_csv(StringIO(csv_text))
    return _normalize_json_columns(df, json_cols)


def _build_scholarship_metadata(df) -> list[dict]:
    """Extract enriched metadata from scholarship DataFrame for API responses.

    Includes fields useful for LLM-based recommendation generation:
    - Core identity (id, name)
    - Context (mission_statement, selection_criteria)
    - Location & funding info
    """
    metadata = []
    for _, row in df.iterrows():
        # Build human-readable funding summary from individual boolean fields
        funding_parts = []
        if row.get("funding_covers_tuition"):
            funding_parts.append("tuition")
        if row.get("funding_covers_living"):
            funding_parts.append("living expenses")
        if row.get("funding_covers_airfare"):
            funding_parts.append("airfare")
        if row.get("funding_covers_insurance"):
            funding_parts.append("insurance")
        if funding_parts:
            funding_summary = f"Covers {', '.join(funding_parts)}"
        else:
            funding_summary = "No specified funding coverage"

        # Add monthly stipend info if available
        stipend = row.get("funding_monthly_stipend")
        if stipend and float(stipend) > 0:
            funding_summary += f", plus ${float(stipend):,.0f}/month stipend"

        metadata.append({
            "scholarship_id": row.get("scholarship_id"),
            "name": row.get("name"),
            "mission_statement": row.get("mission_statement"),
            "selection_criteria": row.get("selection_criteria"),
            "host_country": row.get("host_country"),
            "host_region": row.get("host_region"),
            "funding_is_full_funding": bool(row.get("funding_is_full_funding", False)),
            "funding_coverage_summary": funding_summary,
            "language_requirements": row.get("language_requirements"),
            "target_recipient_profile": row.get("target_recipient_profile"),
        })
    return metadata


def _scholarship_to_metadata(sch: dict) -> dict:
    """Convert scholarship dict to enriched metadata for API responses."""
    # Build human-readable funding summary
    funding_parts = []
    if sch.get("funding_covers_tuition"):
        funding_parts.append("tuition")
    if sch.get("funding_covers_living"):
        funding_parts.append("living expenses")
    if sch.get("funding_covers_airfare"):
        funding_parts.append("airfare")
    if sch.get("funding_covers_insurance"):
        funding_parts.append("insurance")
    if funding_parts:
        funding_summary = f"Covers {', '.join(funding_parts)}"
    else:
        funding_summary = "No specified funding coverage"

    stipend = sch.get("funding_monthly_stipend")
    if stipend and float(stipend) > 0:
        funding_summary += f", plus ${float(stipend):,.0f}/month stipend"

    return {
        "scholarship_id": sch.get("scholarship_id"),
        "name": sch.get("name"),
        "mission_statement": sch.get("mission_statement"),
        "selection_criteria": sch.get("selection_criteria"),
        "host_country": sch.get("host_country"),
        "host_region": sch.get("host_region"),
        "funding_is_full_funding": bool(sch.get("funding_is_full_funding", False)),
        "funding_coverage_summary": funding_summary,
        "language_requirements": sch.get("language_requirements"),
        "target_recipient_profile": sch.get("target_recipient_profile"),
    }

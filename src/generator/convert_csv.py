import json
import os
from dataclasses import asdict
from typing import List
import pandas as pd

from src.generator.schemas import Feedback, Scholarship, Student

def flatten_scholarship(sch: Scholarship) -> dict:
    record = asdict(sch)
    record["eligible_nationalities"] = json.dumps(sch.eligible_nationalities)
    record["eligible_degree_levels"] = json.dumps(sch.eligible_degree_levels)
    record["eligible_high_school_tracks"] = json.dumps(sch.eligible_high_school_tracks)
    record["eligible_fields"] = json.dumps(sch.eligible_fields)
    record["language_requirements"] = json.dumps([asdict(lr) for lr in sch.language_requirements])
    record["selection_criteria"] = json.dumps(asdict(sch.selection_criteria))
    fc = sch.funding_coverage
    del record["funding_coverage"]
    record["funding_covers_tuition"] = fc.covers_tuition
    record["funding_covers_living"] = fc.covers_living_expense
    record["funding_covers_airfare"] = fc.covers_airfare
    record["funding_covers_insurance"] = fc.covers_insurance
    record["funding_monthly_stipend"] = fc.monthly_stipend
    record["funding_is_full_funding"] = fc.is_full_funding
    record["funding_coverage_count"] = fc.coverage_count
    return record


def flatten_student(s: Student) -> dict:
    record = asdict(s)
    record["language_proficiency"] = json.dumps([asdict(lp) for lp in s.language_proficiency])
    record["olympiad_subjects"] = json.dumps(s.olympiad_subjects)
    record["target_countries"] = json.dumps(s.target_countries)
    return record


def flatten_feedback(f: Feedback) -> dict:
    return {
        "student_id": f.student_id,
        "scholarship_id": f.scholarship_id,
        "feedback_type": f.feedback_type,
        "timestamp": f.timestamp,
    }


def save_to_csv(
    students: List[Student],
    scholarships: List[Scholarship],
    feedbacks: List[Feedback],
    output_dir: str = "././data/raw/",
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame([flatten_scholarship(s) for s in scholarships]).to_csv(
        f"{output_dir}/scholarships.csv", index=False
    )
    pd.DataFrame([flatten_student(s) for s in students]).to_csv(
        f"{output_dir}/students.csv", index=False
    )
    pd.DataFrame([flatten_feedback(f) for f in feedbacks]).to_csv(
        f"{output_dir}/feedback.csv", index=False
    )
    return output_dir

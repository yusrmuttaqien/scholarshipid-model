"""Single-pool dataset generator for two-tower recommendation system.

Unlike generator.py which creates separate entity pools per split,
this generator creates ONE pool of students and scholarships, then
outputs flat CSV files. Splitting is done in code via time-based
partitioning on pair timestamps.

Total pool: 20,000 students + 800 scholarships (same as generator.py combined)

Output structure:
    datasets_single_pool/
    ├── students.csv          # 20,000 students (no split column)
    ├── scholarships.csv      # 800 scholarships (no split column)
    ├── pairs.csv             # ALL pairs with `timestamp` column
    └── feedback.csv          # ALL feedback with `timestamp` column

Time-based splitting (done in code, not here):
    Sort pairs by timestamp → 70% train, 15% val, 15% test
"""

import json
import os
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Import shared dataclasses and enums from generator.py
from generator import (
    CareerTrack,
    DegreeLevel,
    FundingCoverage,
    HostRegion,
    LanguageProficiency,
    LanguageRequirement,
    LanguageTest,
    MajorField,
    Pair,
    Scholarship,
    SelectionCriteria,
    Student,
    UniversityTier,
)

# ============================================================
# SINGLE POOL DATASET GENERATOR
# ============================================================


class SinglePoolDatasetGenerator:
    """Generate a single pool of students and scholarships, then generate
    all pairs with timestamps for time-based splitting.

    Total pool: 20,000 students + 800 scholarships
    Pair split: 70% train, 15% val, 15% test (done in code via timestamp)
    """

    def __init__(
        self,
        num_students: int = 20_000,
        num_scholarships: int = 800,
        seed: int = 42,
    ):
        self.num_students = num_students
        self.num_scholarships = num_scholarships
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.prng = random.Random(seed)

        # Country lists for generation
        self.countries = [
            "Indonesia",
            "Malaysia",
            "Thailand",
            "Philippines",
            "Vietnam",
            "Singapore",
            "Japan",
            "South Korea",
            "China",
            "India",
            "United States",
            "United Kingdom",
            "Germany",
            "France",
            "Australia",
            "Canada",
            "Netherlands",
            "Sweden",
            "Norway",
        ]

        self.asian_countries = [
            "Indonesia",
            "Malaysia",
            "Thailand",
            "Philippines",
            "Vietnam",
            "Singapore",
            "Japan",
            "South Korea",
            "China",
            "India",
        ]
        self.european_countries = [
            "United Kingdom",
            "Germany",
            "France",
            "Netherlands",
            "Sweden",
            "Norway",
        ]
        self.american_countries = ["United States", "Canada"]

        self.universities = [
            "University of Indonesia",
            "National University of Singapore",
            "Tokyo University",
            "Seoul National University",
            "Tsinghua University",
            "MIT",
            "Stanford University",
            "Oxford University",
            "ETH Zurich",
            "University of Melbourne",
            "Harvard University",
        ]

        self.personal_statement_templates = [
            "I am passionate about {field} and aspire to make a significant contribution to {interest}.",
            "My journey in {field} began during my undergraduate studies where I discovered my love for {interest}.",
            "As a dedicated student of {field}, I have been actively involved in {activity} research.",
            "Growing up in {country}, I witnessed firsthand the challenges in {interest} and want to address them.",
        ]

        self.mission_statement_templates = [
            "We seek to support outstanding students in {field} who demonstrate commitment to {goal}.",
            "This scholarship aims to nurture future leaders in {field} with a passion for {interest}.",
            "Our mission is to empower talented individuals from {region} pursuing excellence in {field}.",
        ]

        self.target_profile_templates = [
            "We are looking for candidates with strong academic records and proven {activity} experience.",
            "Ideal candidates demonstrate excellence in {field} and commitment to {goal}.",
            "We value students who show leadership potential and dedication to {interest}.",
        ]

        self.research_interests = [
            "artificial intelligence",
            "machine learning",
            "climate change mitigation",
            "public health policy",
            "renewable energy systems",
            "financial technology",
            "human-computer interaction",
            "genomic research",
            "education reform",
            "sustainable agriculture",
            "quantum computing",
            "cybersecurity",
        ]

        self.achievement_templates = [
            "Published {n} papers in {field} conferences and journals.",
            "Led a team of {team_size} in a {field} research project.",
            "Won {competition} competition at the national level.",
            "Volunteered {hours} hours for community service in {activity}.",
            "Served as president of the {field} student association.",
        ]

    def generate_student(self, student_id: str) -> Student:
        """Generate a synthetic student record.

        All students are SMA (high school) per project brief requirement #1.
        They target S1 (Bachelor's) programs, with some targeting S2/S3
        for direct-to-graduate scholarships.
        """
        nationality = self.prng.choice(self.countries)
        age = self.rng.randint(16, 18)  # SMA students: 16-18 years old

        # All students are SMA (high schoolers)
        current_degree = DegreeLevel.SMA.value
        target_degree = self.prng.choices(
            [DegreeLevel.S1.value, DegreeLevel.S2.value],
            weights=[80, 20],  # 80% target S1, 20% target S2
        )[0]

        major = self.prng.choice([e.value for e in MajorField])
        university = self.prng.choice(self.universities)
        tier = self.prng.choice([e.value for e in UniversityTier])

        # GPA generation (dual scale)
        gpa_4scale = round(self.rng.uniform(2.0, 4.0), 2)
        gpa_percentage = round(gpa_4scale * 25, 1)

        # Language proficiency — prefer common tests (TOEFL/IELTS)
        language_proficiency = []
        if self.prng.random() > 0.3:
            test_type = self.prng.choices(
                [e.value for e in LanguageTest],
                weights=[25, 25, 5, 5, 5, 10, 25],
            )[0]
            score = (
                self.rng.uniform(40, 120)
                if test_type == "toefl"
                else (
                    self.rng.uniform(4.0, 9.0)
                    if test_type == "ielts"
                    else self.rng.randint(1, 6) * 50
                )
            )
            language_proficiency.append(LanguageProficiency(test_type, round(score, 1)))

        # Experience counts
        research_exp = self.rng.poisson(2)
        leadership_exp = self.rng.poisson(1)
        volunteer_exp = self.rng.poisson(3)
        competition_wins = self.rng.poisson(1)

        career_track = self.prng.choice([e.value for e in CareerTrack])
        willing_return = self.prng.random() > 0.2

        # Target countries — bias toward popular host countries
        # All SMA students want to study abroad, so all have target_countries
        num_targets = self.rng.randint(1, 4)
        target_countries = []

        # 60% chance to include a popular host country
        if self.prng.random() < 0.6:
            popular_hosts = [
                "United States",
                "United Kingdom",
                "Germany",
                "France",
                "Japan",
                "Australia",
                "Canada",
                "Netherlands",
            ]
            target_countries = [self.prng.choice(popular_hosts)]
            if num_targets > 1:
                extras = self.prng.sample(
                    self.countries, min(num_targets - 1, len(self.countries))
                )
                target_countries.extend(extras)
        else:
            target_countries = self.prng.sample(
                self.countries, min(num_targets, len(self.countries))
            )

        # Text fields
        interest = self.prng.choice(self.research_interests)
        template = self.prng.choice(self.personal_statement_templates)
        personal_statement = template.format(
            field=major,
            interest=interest,
            country=nationality,
            activity=self.prng.choice(["research", "community service", "innovation"]),
        )

        # Achievements narrative
        achievements = []
        for _ in range(self.rng.randint(1, 4)):
            tpl = self.prng.choice(self.achievement_templates)
            achievements.append(
                tpl.format(
                    n=self.rng.randint(1, 10),
                    field=major,
                    team_size=self.rng.randint(3, 15),
                    competition=self.prng.choice(["math", "science", "debate"]),
                    hours=self.rng.randint(20, 200),
                    activity=interest,
                )
            )
        achievements_narrative = ". ".join(achievements) + "."

        # Funding preferences
        needs_full_funding = self.prng.random() < 0.6  # 60% need full funding
        can_self_fund_living = not needs_full_funding and self.prng.random() > 0.5

        return Student(
            student_id=student_id,
            nationality=nationality,
            age=age,
            current_degree_level=current_degree,
            target_degree_level=target_degree,
            current_major_field=major,
            current_university=university,
            current_university_tier=tier,
            current_gpa_4scale=gpa_4scale,
            current_gpa_percentage=gpa_percentage,
            language_proficiency=language_proficiency,
            research_experience_count=research_exp,
            leadership_experience_count=leadership_exp,
            volunteer_experience_count=volunteer_exp,
            competition_wins_count=competition_wins,
            intended_career_track=career_track,
            willing_to_return_home=willing_return,
            target_countries=target_countries,
            personal_statement=personal_statement,
            achievements_narrative=achievements_narrative,
            research_interest=interest,
            needs_full_funding=needs_full_funding,
            can_self_fund_living=can_self_fund_living,
        )

    def generate_scholarship(self, scholarship_id: str) -> Scholarship:
        """Generate a synthetic scholarship record."""
        region = self.prng.choice([e.value for e in HostRegion])
        host_country_map = {
            "asia": self.asian_countries,
            "europe": self.european_countries,
            "north_america": self.american_countries,
            "south_america": ["Brazil", "Argentina", "Chile"],
            "africa": ["South Africa", "Nigeria", "Kenya"],
            "oceania": ["Australia", "New Zealand"],
        }
        host_country = self.prng.choice(host_country_map.get(region, self.countries))

        # Eligible nationalities (typically 2-8 countries)
        num_nationalities = self.rng.randint(2, 8)
        eligible_nationalities = self.prng.sample(
            self.countries, min(num_nationalities, len(self.countries))
        )

        # Age range — tailored for SMA students (16-18)
        min_age = self.rng.choice([15, 16, 17])
        max_age = min_age + self.rng.randint(3, 6)  # narrower range for high schoolers

        # Eligible degree levels — SMA always included (brief requirement #1)
        # Some scholarships also accept S1 grads
        eligible_degree_levels = [DegreeLevel.SMA.value]
        if self.prng.random() > 0.5:
            extra = self.prng.choice(
                [e for e in DegreeLevel if e.value != DegreeLevel.SMA.value]
            )
            eligible_degree_levels.append(extra.value)

        # Eligible fields
        num_fields = self.rng.randint(1, 5)
        eligible_fields = self.prng.sample([e.value for e in MajorField], num_fields)

        # Specific majors (optional, 30% chance)
        eligible_majors_specific = None
        if self.prng.random() < 0.3 and len(eligible_fields) == 1:
            eligible_majors_specific = self.prng.sample(
                [e.value for e in MajorField], 2
            )

        # GPA requirements
        min_gpa_4scale = (
            round(self.rng.uniform(2.5, 3.8), 2) if self.prng.random() > 0.3 else None
        )
        min_gpa_percentage = round(min_gpa_4scale * 25, 1) if min_gpa_4scale else None

        # Language requirements — prefer common tests (TOEFL/IELTS/GRE)
        language_requirements = []
        if self.prng.random() > 0.2:
            test_type = self.prng.choices(
                [e.value for e in LanguageTest],
                weights=[25, 25, 5, 5, 5, 10, 25],
            )[0]
            min_score = (
                self.rng.uniform(80, 110)
                if test_type == "toefl"
                else (
                    self.rng.uniform(6.0, 8.0)
                    if test_type == "ielts"
                    else self.rng.randint(3, 5) * 50
                )
            )
            is_mandatory = self.prng.random() > 0.3  # 70% mandatory, 30% preferred
            language_requirements.append(
                LanguageRequirement(test_type, round(min_score, 1), is_mandatory)
            )

        # Selection criteria weights
        criteria = SelectionCriteria(
            academic=round(self.rng.uniform(0.2, 0.5), 2),
            leadership=round(self.rng.uniform(0.05, 0.3), 2),
            research=round(self.rng.uniform(0.1, 0.4), 2),
            extracurricular=round(self.rng.uniform(0.05, 0.3), 2),
            essay=round(self.rng.uniform(0.05, 0.3), 2),
        )
        # Normalize to sum to 1.0
        total = (
            criteria.academic
            + criteria.leadership
            + criteria.research
            + criteria.extracurricular
            + criteria.essay
        )
        criteria.academic /= total
        criteria.leadership /= total
        criteria.research /= total
        criteria.extracurricular /= total
        criteria.essay /= total

        # Funding coverage — structured (tuition, living, airfare, insurance)
        funding_coverage = FundingCoverage(
            covers_tuition=self.prng.random() > 0.2,
            covers_living_expense=self.prng.random() > 0.3,
            covers_airfare=self.prng.random() > 0.5,
            covers_insurance=self.prng.random() > 0.6,
            total_amount=round(self.rng.uniform(3_000, 100_000), 2),
        )

        career_track_preference = (
            self.prng.choice([e.value for e in CareerTrack])
            if self.prng.random() > 0.5
            else None
        )
        requires_return = self.prng.random() < 0.3

        # Text fields
        field = eligible_fields[0] if eligible_fields else "computer_science"
        interest = self.prng.choice(self.research_interests)

        mission = self.prng.choice(self.mission_statement_templates).format(
            field=field,
            interest=interest,
            region=region.capitalize(),
            goal=self.prng.choice(
                ["societal impact", "innovation", "knowledge advancement"]
            ),
        )

        target_profile = self.prng.choice(self.target_profile_templates).format(
            field=field,
            activity=self.prng.choice(["research", "leadership"]),
            interest=interest,
            goal=self.prng.choice(["community service", "technological advancement"]),
        )

        name = f"{host_country} {field.replace('_', ' ').title()} Scholarship"

        return Scholarship(
            scholarship_id=scholarship_id,
            name=name,
            eligible_nationalities=eligible_nationalities,
            min_age=min_age,
            max_age=max_age,
            eligible_degree_levels=eligible_degree_levels,
            eligible_fields=eligible_fields,
            eligible_majors_specific=eligible_majors_specific,
            host_country=host_country,
            host_region=region,
            min_gpa_4scale=min_gpa_4scale,
            min_gpa_percentage=min_gpa_percentage,
            language_requirements=language_requirements,
            selection_criteria=criteria,
            funding_coverage=funding_coverage,
            career_track_preference=career_track_preference,
            requires_return_home_country=requires_return,
            mission_statement=mission,
            target_recipient_profile=target_profile,
        )

    def generate_all_students(self) -> List[Student]:
        """Generate all students for the single pool."""
        students = []
        for i in range(self.num_students):
            student_id = f"STU_{i:06d}"
            students.append(self.generate_student(student_id))
        return students

    def generate_all_scholarships(self) -> List[Scholarship]:
        """Generate all scholarships for the single pool."""
        scholarships = []
        for i in range(self.num_scholarships):
            scholarship_id = f"SCH_{i:06d}"
            scholarships.append(self.generate_scholarship(scholarship_id))
        return scholarships

    def save_to_csv(
        self,
        students: List[Student],
        scholarships: List[Scholarship],
        pairs: List[Pair],
        feedbacks: List["FeedbackRecord"],
        output_dir: str = "./datasets_single_pool",
    ):
        """Save all generated data to flat CSV files."""
        os.makedirs(output_dir, exist_ok=True)

        # Flatten student data for CSV
        student_records = []
        for s in students:
            record = s.__dict__.copy()
            record["language_proficiency"] = json.dumps(
                [lp.__dict__ for lp in s.language_proficiency]
            )
            record["target_countries"] = json.dumps(s.target_countries)
            student_records.append(record)

        # Flatten scholarship data for CSV
        scholarship_records = []
        for sch in scholarships:
            record = sch.__dict__.copy()
            record["eligible_nationalities"] = json.dumps(sch.eligible_nationalities)
            record["eligible_degree_levels"] = json.dumps(sch.eligible_degree_levels)
            record["eligible_fields"] = json.dumps(sch.eligible_fields)
            record["eligible_majors_specific"] = (
                json.dumps(sch.eligible_majors_specific)
                if sch.eligible_majors_specific
                else None
            )
            record["language_requirements"] = json.dumps(
                [lr.__dict__ for lr in sch.language_requirements]
            )
            record["selection_criteria"] = json.dumps(sch.selection_criteria.__dict__)
            # Flatten FundingCoverage into separate columns
            fc = sch.funding_coverage
            del record["funding_coverage"]
            record["funding_covers_tuition"] = fc.covers_tuition
            record["funding_covers_living"] = fc.covers_living_expense
            record["funding_covers_airfare"] = fc.covers_airfare
            record["funding_covers_insurance"] = fc.covers_insurance
            record["funding_total_amount"] = fc.total_amount
            scholarship_records.append(record)

        # Flatten pairs for CSV
        pair_records = []
        for p in pairs:
            pair_records.append(
                {
                    "student_id": p.student_id,
                    "scholarship_id": p.scholarship_id,
                    "label": p.label,
                    "relevance_score": round(p.relevance_score, 4),
                    "timestamp": p.timestamp,
                }
            )

        # Flatten feedback for CSV
        feedback_records = []
        for f in feedbacks:
            feedback_records.append(
                {
                    "student_id": f.student_id,
                    "scholarship_id": f.scholarship_id,
                    "feedback_type": f.feedback_type,
                    "weight": f.weight,
                    "timestamp": f.timestamp,
                }
            )

        # Save to CSV
        pd.DataFrame(student_records).to_csv(f"{output_dir}/students.csv", index=False)
        pd.DataFrame(scholarship_records).to_csv(
            f"{output_dir}/scholarships.csv", index=False
        )
        pd.DataFrame(pair_records).to_csv(f"{output_dir}/pairs.csv", index=False)
        pd.DataFrame(feedback_records).to_csv(f"{output_dir}/feedback.csv", index=False)

        return output_dir


# ============================================================
# HELPER DATA CLASSES
# ============================================================


class FeedbackRecord:
    """Flat feedback record for CSV output."""

    def __init__(
        self,
        student_id: str,
        scholarship_id: str,
        feedback_type: str,
        timestamp: str,
    ):
        self.student_id = student_id
        self.scholarship_id = scholarship_id
        self.feedback_type = feedback_type
        self.timestamp = timestamp

    @property
    def weight(self) -> float:
        """Weight for training — stronger signal gets higher weight."""
        weights = {
            "apply": 3.0,
            "click": 2.0,
            "view": 1.0,
            "reject": -1.0,
        }
        return weights.get(self.feedback_type, 1.0)


# ============================================================
# HARD FILTER (copied from generator.py for standalone use)
# ============================================================


def check_hard_filter(student: Student, scholarship: Scholarship) -> Tuple[bool, str]:
    """Check if a student-scholarship pair passes all hard eligibility filters.

    Returns:
        Tuple of (is_eligible, reason_if_not)
    """
    # Nationality check
    if student.nationality not in scholarship.eligible_nationalities:
        return False, "nationality"

    # Degree level check
    if student.target_degree_level not in scholarship.eligible_degree_levels:
        return False, "degree_level"

    # Age check
    if not (scholarship.min_age <= student.age <= scholarship.max_age):
        return False, "age"

    # GPA check (dual scale support)
    gpa_eligible = False
    if scholarship.min_gpa_4scale and student.current_gpa_4scale:
        if student.current_gpa_4scale >= scholarship.min_gpa_4scale:
            gpa_eligible = True
    elif scholarship.min_gpa_percentage and student.current_gpa_percentage:
        if student.current_gpa_percentage >= scholarship.min_gpa_percentage:
            gpa_eligible = True

    if not gpa_eligible and (
        scholarship.min_gpa_4scale or scholarship.min_gpa_percentage
    ):
        return False, "gpa"

    # Language requirements check (only mandatory ones are hard filters)
    for req in scholarship.language_requirements:
        if not req.is_mandatory:
            continue  # Preferred but not required — skip hard filter
        student_scores = [
            lp for lp in student.language_proficiency if lp.test_type == req.test_type
        ]
        if not any(lp.score >= req.min_score for lp in student_scores):
            return False, "language"

    # Return home check
    if scholarship.requires_return_home_country and not student.willing_to_return_home:
        return False, "return_home"

    # Target countries check (if student has preferences)
    if student.target_countries:
        if scholarship.host_country not in student.target_countries:
            # Check region
            if scholarship.host_region not in [c for c in student.target_countries]:
                return False, "target_country"

    return True, ""


def count_failed_criteria(student: Student, scholarship: Scholarship) -> int:
    """Count how many hard filter criteria a pair fails."""
    failed = 0

    # Nationality check
    if student.nationality not in scholarship.eligible_nationalities:
        failed += 1

    # Degree level check
    if student.target_degree_level not in scholarship.eligible_degree_levels:
        failed += 1

    # Age check
    if not (scholarship.min_age <= student.age <= scholarship.max_age):
        failed += 1

    # GPA check
    gpa_eligible = False
    if scholarship.min_gpa_4scale and student.current_gpa_4scale:
        if student.current_gpa_4scale >= scholarship.min_gpa_4scale:
            gpa_eligible = True
    elif scholarship.min_gpa_percentage and student.current_gpa_percentage:
        if student.current_gpa_percentage >= scholarship.min_gpa_percentage:
            gpa_eligible = True

    if not gpa_eligible and (
        scholarship.min_gpa_4scale or scholarship.min_gpa_percentage
    ):
        failed += 1

    # Language requirements check (only mandatory ones count)
    lang_failed = False
    for req in scholarship.language_requirements:
        if not req.is_mandatory:
            continue  # Preferred but not required
        student_scores = [
            lp for lp in student.language_proficiency if lp.test_type == req.test_type
        ]
        if not any(lp.score >= req.min_score for lp in student_scores):
            lang_failed = True
            break
    if lang_failed:
        failed += 1

    # Return-home check
    if scholarship.requires_return_home_country and not student.willing_to_return_home:
        failed += 1

    return failed


# ============================================================
# PAIR GENERATORS
# ============================================================


def compute_relevance_score(student: Student, scholarship: Scholarship) -> float:
    """Compute a soft relevance score (0.0 - 1.0) for ranking loss."""
    score = 0.0
    components = 0

    # GPA margin (higher GPA relative to min = better)
    if scholarship.min_gpa_4scale and student.current_gpa_4scale:
        margin = min(
            (student.current_gpa_4scale - scholarship.min_gpa_4scale) / 1.5, 1.0
        )
        score += max(margin, 0.0)
        components += 1

    # Field match
    if student.current_major_field in scholarship.eligible_fields:
        score += 1.0
        components += 1

    # Career track match
    if scholarship.career_track_preference == student.intended_career_track:
        score += 1.0
        components += 1

    # Language proficiency margin
    for req in scholarship.language_requirements:
        student_scores = [
            lp for lp in student.language_proficiency if lp.test_type == req.test_type
        ]
        if student_scores:
            max_score = max(lp.score for lp in student_scores)
            margin = min((max_score - req.min_score) / req.min_score, 1.0)
            score += max(margin, 0.0)
            components += 1

    if components == 0:
        return 0.5
    return round(score / components, 4)


def generate_positive_pairs(
    students: List[Student], scholarships: List[Scholarship]
) -> List[Tuple[Student, Scholarship]]:
    """Generate all positive pairs (pairs that pass hard filters)."""
    positive_pairs = []
    for student in students:
        for scholarship in scholarships:
            is_eligible, _ = check_hard_filter(student, scholarship)
            if is_eligible:
                positive_pairs.append((student, scholarship))
    return positive_pairs


def generate_random_negatives(
    students: List[Student],
    scholarships: List[Scholarship],
    positive_pairs: List[Tuple[Student, Scholarship]],
    ratio: int = 3,
) -> List[Tuple[Student, Scholarship]]:
    """Generate random negative pairs."""
    positive_set = set((s.student_id, sch.scholarship_id) for s, sch in positive_pairs)

    prng = random.Random(42)
    neg_count = len(positive_pairs) * ratio
    negatives = []

    attempts = 0
    max_attempts = len(students) * len(scholarships)

    while len(negatives) < neg_count and attempts < max_attempts:
        attempts += 1
        stu = prng.choice(students)
        sch = prng.choice(scholarships)

        if (stu.student_id, sch.scholarship_id) not in positive_set:
            negatives.append((stu, sch))

    return negatives


def generate_hard_negatives(
    students: List[Student],
    scholarships: List[Scholarship],
    positive_pairs: List[Tuple[Student, Scholarship]],
    max_per_student: int = 2,
) -> List[Tuple[Student, Scholarship]]:
    """Generate hard negative pairs (fail by exactly ONE criterion)."""
    positive_set = set((s.student_id, sch.scholarship_id) for s, sch in positive_pairs)

    hard_negatives = []
    rng = np.random.RandomState(42)

    for student in students:
        hard_for_student = []
        for scholarship in scholarships:
            key = (student.student_id, scholarship.scholarship_id)
            if key in positive_set:
                continue

            failed_count = count_failed_criteria(student, scholarship)
            if failed_count == 1:
                hard_for_student.append(scholarship)

        # Sample up to max_per_student hard negatives per student
        if hard_for_student:
            sampled = rng.choice(
                hard_for_student,
                size=min(max_per_student, len(hard_for_student)),
                replace=False,
            )
            hard_negatives.extend((student, s) for s in sampled)

    return hard_negatives


# ============================================================
# MAIN — Single Pool Dataset Generation Pipeline
# ============================================================


def main():
    """Generate and save a single pool dataset for the two-tower recommendation system."""

    # Configuration
    NUM_STUDENTS = 20_000
    NUM_SCHOLARSHIPS = 800
    SEED = 42

    print("=" * 60)
    print("Single Pool Dataset Generator")
    print("=" * 60)
    print("Configuration:")
    print(f"  Students: {NUM_STUDENTS}")
    print(f"  Scholarships: {NUM_SCHOLARSHIPS}")
    print(f"  Random seed: {SEED}")
    print("=" * 60)

    # Create generator
    generator = SinglePoolDatasetGenerator(NUM_STUDENTS, NUM_SCHOLARSHIPS, SEED)

    # Generate all entities
    print("\nGenerating students...")
    students = generator.generate_all_students()
    print(f"  Generated {len(students)} students")

    print("\nGenerating scholarships...")
    scholarships = generator.generate_all_scholarships()
    print(f"  Generated {len(scholarships)} scholarships")

    # Generate pairs
    print("\n" + "=" * 60)
    print("Generating training pairs...")
    print("=" * 60)

    print("\nGenerating positive pairs...")
    positive_pairs = generate_positive_pairs(students, scholarships)
    print(f"  Found {len(positive_pairs)} positive pairs")

    if not positive_pairs:
        print("  WARNING: No positive pairs found!")
        return

    # Build all Pair records with timestamps
    base_date = datetime(2024, 1, 1)
    rng = np.random.RandomState(SEED)

    all_pairs: List[Pair] = []

    # Positive pairs (label=1)
    for idx, (student, scholarship) in enumerate(positive_pairs):
        # Assign timestamps spread across the year
        offset_days = rng.randint(0, 365)
        offset_hours = rng.randint(0, 23)
        timestamp = (
            base_date + timedelta(days=offset_days, hours=offset_hours)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        relevance = compute_relevance_score(student, scholarship)
        all_pairs.append(
            Pair(
                student_id=student.student_id,
                scholarship_id=scholarship.scholarship_id,
                label=1,
                relevance_score=relevance,
                timestamp=timestamp,
            )
        )

    # Negative pairs — random (label=0)
    print("Generating random negatives...")
    random_negs = generate_random_negatives(
        students, scholarships, positive_pairs, ratio=3
    )
    print(f"  Generated {len(random_negs)} random negatives")

    for student, scholarship in random_negs:
        offset_days = rng.randint(0, 365)
        offset_hours = rng.randint(0, 23)
        timestamp = (
            base_date + timedelta(days=offset_days, hours=offset_hours)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        all_pairs.append(
            Pair(
                student_id=student.student_id,
                scholarship_id=scholarship.scholarship_id,
                label=0,
                relevance_score=0.0,
                timestamp=timestamp,
            )
        )

    # Hard negatives (label=0)
    print("Generating hard negatives...")
    hard_negs = generate_hard_negatives(
        students, scholarships, positive_pairs, max_per_student=2
    )
    print(f"  Generated {len(hard_negs)} hard negatives")

    for student, scholarship in hard_negs:
        offset_days = rng.randint(0, 365)
        offset_hours = rng.randint(0, 23)
        timestamp = (
            base_date + timedelta(days=offset_days, hours=offset_hours)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        all_pairs.append(
            Pair(
                student_id=student.student_id,
                scholarship_id=scholarship.scholarship_id,
                label=0,
                relevance_score=0.0,
                timestamp=timestamp,
            )
        )

    # Sort pairs by timestamp (for time-based splitting)
    all_pairs.sort(key=lambda p: p.timestamp or "")

    total_pos = sum(1 for p in all_pairs if p.label == 1)
    total_neg = sum(1 for p in all_pairs if p.label == 0)
    print(
        f"\n  Total: {total_pos} positive + {total_neg} negative = {len(all_pairs)} pairs"
    )

    # Generate implicit feedback
    print("\nGenerating implicit feedback...")
    feedbacks = _generate_feedback(
        students, scholarships, positive_pairs, num_feedback_per_student=5
    )
    print(f"  Generated {len(feedbacks)} feedback entries")

    # Count feedback types
    type_counts: Dict[str, int] = {}
    for fb in feedbacks:
        type_counts[fb.feedback_type] = type_counts.get(fb.feedback_type, 0) + 1
    for fb_type, count in sorted(type_counts.items()):
        print(f"    {fb_type}: {count}")

    # Save to flat CSV files
    output_dir = "./datasets_single_pool"
    generator.save_to_csv(students, scholarships, all_pairs, feedbacks, output_dir)

    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)

    # Summary
    print(f"\nGenerated files in {output_dir}/:")
    print("    students.csv")
    print("    scholarships.csv")
    print("    pairs.csv (combined positive + negative, with timestamps)")
    print("    feedback.csv (implicit feedback for retraining)")

    print("\nTime-based splitting (to be done in code):")
    print("    Sort pairs by timestamp → 70% train, 15% val, 15% test")
    print("    Same for feedback.csv")


def _generate_feedback(
    students: List[Student],
    scholarships: List[Scholarship],
    positive_pairs: List[Tuple[Student, Scholarship]],
    num_feedback_per_student: int = 5,
) -> List[FeedbackRecord]:
    """Generate synthetic implicit feedback for online learning / retraining."""
    rng = np.random.RandomState(42)
    prng = random.Random(42)
    base_date = datetime(2024, 1, 1)

    # Build lookup: student_id -> list of eligible scholarships
    student_to_pos_schols: Dict[str, List[Scholarship]] = {}
    for student, scholarship in positive_pairs:
        if student.student_id not in student_to_pos_schols:
            student_to_pos_schols[student.student_id] = []
        student_to_pos_schols[student.student_id].append(scholarship)

    feedbacks = []

    for student in students:
        sid = student.student_id
        eligible_schols = student_to_pos_schols.get(sid, [])

        # Random number of interactions per student
        num_interactions = rng.randint(1, num_feedback_per_student * 3)

        for _ in range(num_interactions):
            # Decide which scholarship to interact with
            if eligible_schols and prng.random() < 0.7:
                # 70% interact with eligible scholarships
                schol = prng.choice(eligible_schols)
                # For eligible: mostly positive feedback
                feedback_type = prng.choices(
                    ["apply", "click", "view", "reject"],
                    weights=[20, 35, 30, 15],
                )[0]
            else:
                # 30% interact with random (ineligible) scholarships
                schol = prng.choice(scholarships)
                # For ineligible: mostly negative feedback
                feedback_type = prng.choices(
                    ["apply", "click", "view", "reject"],
                    weights=[5, 15, 30, 50],
                )[0]

            # Generate timestamp (within 365 days)
            offset_days = rng.randint(0, 365)
            offset_hours = rng.randint(0, 23)
            timestamp = (
                base_date + timedelta(days=offset_days, hours=offset_hours)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")

            feedbacks.append(
                FeedbackRecord(
                    student_id=sid,
                    scholarship_id=schol.scholarship_id,
                    feedback_type=feedback_type,
                    timestamp=timestamp,
                )
            )

    return feedbacks


if __name__ == "__main__":
    main()

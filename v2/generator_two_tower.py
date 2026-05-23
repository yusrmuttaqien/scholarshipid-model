"""Two-Tower Recommendation System - Dataset Generator (Fusion)

Generates synthetic training data for a two-tower neural network
that predicts continuous relevance scores (0.0-1.0) between
students and scholarships.

Uses:
  - Yusr's student/scholarship generation logic
  - Almer's 5-stage relevance scorer (hard eligibility + component scores)

Output structure:
    datasets_two_tower/
    ├── students.csv          # 20,000 students
    ├── scholarships.csv      # 800 synthetic scholarships
    ├── pairs.csv             # Balanced pairs with continuous relevance_score
    └── feedback.csv          # Implicit feedback for retraining
"""

import json
import os
import random
import sys
from dataclasses import asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure project root is on sys.path so `from src.*` works when
# running this file from any subdirectory (e.g. python v2/generator_two_tower.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.schemas import (
    CareerTrack,
    Feedback,
    FundingCoverage,
    HighSchoolTrack,
    HostRegion,
    IncomeCategory,
    LanguageProficiency,
    LanguageRequirement,
    LanguageTest,
    MajorField,
    OlympiadLevel,
    OlympiadSubject,
    Pair,
    Scholarship,
    SchoolTier,
    SelectionCriteria,
    Student,
)
from src.io import save_to_csv
from src.scorer import compute_relevance_score as _scorer

# Enums and dataclasses are imported from src.schemas (see src/schemas/enums.py)
# No need to redefine here.


# ============================================================
# TWO-TOWER DATASET GENERATOR
# ============================================================


class TwoTowerDatasetGenerator:
    """Generate synthetic datasets for two-tower recommendation system.

    Produces pairs with continuous relevance scores (0.0-1.0) for
    regression training. Distribution reflects real-world scholarship
    matching: most random pairs are knockouts, few are strong matches.
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

        # Country groups
        self.asian_countries = [
            "indonesia",
            "malaysia",
            "thailand",
            "philippines",
            "vietnam",
            "singapore",
            "japan",
            "south_korea",
            "china",
            "india",
        ]
        self.european_countries = [
            "france",
            "germany",
            "netherlands",
            "sweden",
            "uk",
            "switzerland",
        ]
        self.american_countries = ["canada", "usa"]
        self.african_countries = [
            "egypt",
            "kenya",
            "morocco",
            "nigeria",
            "south_africa",
        ]
        self.oceanian_countries = ["australia", "new_zealand"]
        self.south_american_countries = ["argentina", "brazil", "chile"]

        self.all_countries = (
            self.asian_countries
            + self.european_countries
            + self.american_countries
            + self.african_countries
            + self.oceanian_countries
            + self.south_american_countries
        )

        # Host countries by region
        self.host_countries_by_region = {
            "asia": self.asian_countries,
            "europe": self.european_countries,
            "north_america": self.american_countries,
            "south_america": self.south_american_countries,
            "africa": self.african_countries,
            "oceania": self.oceanian_countries,
        }

        # School names per country
        self.school_names = {
            "indonesia": [
                "SMA Negeri 1 Jakarta",
                "SMAN 3 Surabaya",
                "SMA Muhammadiyah 1 Bandung",
                "SMAN 5 Medan",
                "SMA Katolik Santa Maria Yogyakarta",
            ],
            "malaysia": [
                "Sekolah Menengah Kebangsaan Kuala Lumpur",
                "SMK Damansara",
                "Sekolah Menengah Teknik Johor",
            ],
            "thailand": [
                "Chulalongkorn Academic Division School",
                "Suankularb Wittayalai School",
            ],
            "philippines": [
                "University of the Philippines High School",
                "Ateneo de Manila Senior High",
            ],
            "vietnam": [
                "Lý Tự Trọng High School",
                "Tuan Minh High School",
            ],
            "india": [
                "Delhi Public School",
                "Kendriya Vidyalaya",
            ],
            "japan": [
                "Tokyo Metropolitan Nagatsuta High School",
                "Bunka High School",
            ],
            "south_korea": [
                "Seoul High School",
                "Hwawon High School",
            ],
            "china": [
                "Beijing No.4 High School",
                "Shanghai High School",
            ],
            "singapore": [
                "Raffles Institution",
                "Hwa Chong Institution",
            ],
        }

        # Text templates
        self.personal_statement_templates = [
            "I am passionate about {field} and aspire to make a significant contribution to {interest}.",
            "My journey in {field} began during high school where I discovered my love for {interest}.",
            "As a dedicated student of {track}, I have been actively involved in {activity}.",
            "Growing up in {country}, I witnessed firsthand the challenges in {interest} and want to address them.",
        ]

        self.future_goals_templates = [
            "After completing my studies, I plan to return home and contribute to {interest} in my community.",
            "My long-term goal is to become a leader in {field} and drive innovation in {region}.",
            "I aspire to establish a program focused on {interest} that benefits underrepresented communities.",
            "I want to bridge the gap between {field} and {interest} through research and practice.",
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
            "Achieved {level} level in {subject} olympiad.",
            "Won {competition} competition at the {level} level.",
            "Led a team of {team_size} in a {subject} project.",
            "Volunteered {hours} hours for community service.",
            "Served as {position} in student council.",
        ]

    def _get_school_name(self, nationality: str) -> str:
        """Get a random school name for the given nationality."""
        schools = self.school_names.get(nationality, ["General High School"])
        return self.prng.choice(schools)

    def generate_student(self, student_id: str) -> Student:
        """Generate a synthetic high school student."""
        nationality = self.prng.choice(self.all_countries)
        age = self.rng.randint(16, 18)

        # High school track
        track = self.prng.choice([e.value for e in HighSchoolTrack])

        # Academic scores (0-100 scale)
        overall_avg = round(self.rng.uniform(60, 98), 1)
        math_score = round(self.rng.uniform(50, 100), 1)
        english_score = round(self.rng.uniform(40, 100), 1)
        major_avg = round(self.rng.uniform(55, 100), 1)

        # Language proficiency (many HS students have not taken tests yet)
        language_proficiency = []
        if self.prng.random() > 0.4:
            test_type = self.prng.choices(
                [e.value for e in LanguageTest],
                weights=[30, 30, 5, 5, 5, 5],
            )[0]
            score_map = {
                "toefl": self.rng.uniform(40, 120),
                "ielts": self.rng.uniform(4.0, 8.5),
                "topik": self.rng.randint(1, 6) * 50,
                "jlpt": self.rng.randint(1, 5) * 20,
                "delf": self.rng.randint(1, 5) * 20,
                "hsk": self.rng.randint(1, 6) * 20,
            }
            score = round(score_map.get(test_type, 50), 1)
            language_proficiency.append(
                LanguageProficiency(
                    test_type=test_type,
                    score=score,
                    valid_until=f"2026-{self.rng.randint(1, 12):02d}-01",
                )
            )

        # Olympiad info
        olympiad_level = self.prng.choices(
            [e.value for e in OlympiadLevel],
            weights=[40, 20, 15, 15, 7, 3],
        )[0]
        olympiad_subjects = []
        if olympiad_level != "none":
            num_subjects = self.rng.randint(1, 3)
            olympiad_subjects = self.prng.sample(
                [e.value for e in OlympiadSubject],
                min(num_subjects, len(OlympiadSubject)),
            )

        # Experience counts
        leadership_count = self.rng.poisson(2)
        volunteer_count = self.rng.poisson(3)
        competition_wins = self.rng.poisson(1)

        # Background
        school_tier = self.prng.choices(
            [e.value for e in SchoolTier],
            weights=[5, 20, 15, 25, 20, 10, 5],
        )[0]
        income = self.prng.choices(
            [e.value for e in IncomeCategory],
            weights=[10, 25, 35, 20, 10],
        )[0]
        underrepresented = self.prng.random() < 0.25

        # Career intent
        career_track = self.prng.choice([e.value for e in CareerTrack])
        willing_return = self.prng.random() > 0.2

        # Target countries (where student wants to study)
        num_targets = self.rng.randint(1, 4)
        target_countries = self.prng.sample(
            self.all_countries, min(num_targets, len(self.all_countries))
        )

        # Funding preferences
        needs_full_funding = self.prng.random() < 0.6
        can_self_fund_living = not needs_full_funding and self.prng.random() > 0.5

        # Text fields
        major_field = self.prng.choice([e.value for e in MajorField])
        interest = self.prng.choice(self.research_interests)

        personal_statement = self.prng.choice(
            self.personal_statement_templates
        ).format(
            field=major_field,
            interest=interest,
            track=track,
            country=nationality,
            activity=self.prng.choice(
                ["research", "community service", "innovation"]
            ),
        )

        future_goals = self.prng.choice(self.future_goals_templates).format(
            field=major_field,
            interest=interest,
            region=self.prng.choice(
                ["asia", "europe", "north_america"]
            ).capitalize(),
        )

        achievements = []
        for _ in range(self.rng.randint(1, 4)):
            tpl = self.prng.choice(self.achievement_templates)
            achievements.append(
                tpl.format(
                    level=self.prng.choice(
                        ["school", "city", "provincial", "national"]
                    ),
                    subject=self.prng.choice(
                        [e.value for e in OlympiadSubject]
                    ),
                    competition=self.prng.choice(
                        ["math", "science", "debate"]
                    ),
                    team_size=self.rng.randint(3, 15),
                    hours=self.rng.randint(20, 200),
                    position=self.prng.choice(
                        ["president", "vice president", "secretary"]
                    ),
                )
            )
        achievements_narrative = ". ".join(achievements) + "."

        return Student(
            student_id=student_id,
            nationality=nationality,
            age=age,
            current_degree_level="high_school",
            target_degree_level="bachelors",
            high_school_track=track,
            school_name=self._get_school_name(nationality),
            overall_report_card_average=overall_avg,
            math_score=math_score,
            english_score=english_score,
            major_subject_average=major_avg,
            language_proficiency=language_proficiency,
            olympiad_level=olympiad_level,
            olympiad_subjects=olympiad_subjects,
            leadership_experience_count=leadership_count,
            volunteer_experience_count=volunteer_count,
            competition_wins_count=competition_wins,
            school_tier=school_tier,
            family_income_category=income,
            from_underrepresented_region=underrepresented,
            intended_career_track=career_track,
            willing_to_return_home=willing_return,
            target_countries=target_countries,
            personal_statement=personal_statement,
            achievements_narrative=achievements_narrative,
            future_goals=future_goals,
            needs_full_funding=needs_full_funding,
            can_self_fund_living=can_self_fund_living,
        )

    def generate_scholarship(self, scholarship_id: str) -> Scholarship:
        """Generate a synthetic scholarship record."""
        region = self.prng.choice([e.value for e in HostRegion])
        host_country = self.prng.choice(
            self.host_countries_by_region.get(region, self.all_countries)
        )

        # Eligible nationalities (typically 2-8 countries)
        num_nationalities = self.rng.randint(2, 8)
        eligible_nationalities = self.prng.sample(
            self.all_countries, min(num_nationalities, len(self.all_countries))
        )

        # Age range for high school students
        min_age = self.rng.choice([15, 16, 17])
        max_age = min_age + self.rng.randint(3, 6)

        # Eligible degree levels (always includes high_school)
        eligible_degree_levels = ["high_school"]
        if self.prng.random() > 0.5:
            eligible_degree_levels.append("bachelors")

        # Eligible high school tracks
        num_tracks = self.rng.randint(1, 3)
        eligible_tracks = self.prng.sample(
            [e.value for e in HighSchoolTrack], num_tracks
        )

        # Eligible fields
        num_fields = self.rng.randint(1, 5)
        eligible_fields = self.prng.sample(
            [e.value for e in MajorField], num_fields
        )

        # Preferred school tier
        preferred_tier = self.prng.choice(
            ["excellence", "public_a", "private_a", "accredited_b"]
        )

        # Score requirements (0-100 scale)
        min_report_avg = round(self.rng.uniform(55, 85), 1)
        min_major_avg = round(self.rng.uniform(50, 80), 1)

        # Language requirements
        language_requirements = []
        if self.prng.random() > 0.2:
            test_type = self.prng.choices(
                [e.value for e in LanguageTest],
                weights=[30, 30, 5, 5, 5, 5],
            )[0]
            score_map = {
                "toefl": self.rng.uniform(80, 110),
                "ielts": self.rng.uniform(6.0, 8.0),
                "topik": self.rng.randint(3, 5) * 50,
                "jlpt": self.rng.randint(2, 4) * 20,
                "delf": self.rng.randint(2, 4) * 20,
                "hsk": self.rng.randint(3, 5) * 20,
            }
            min_score = round(score_map.get(test_type, 60), 1)
            is_mandatory = self.prng.random() > 0.3
            language_requirements.append(
                LanguageRequirement(
                    test_type=test_type, min_score=min_score, is_mandatory=is_mandatory
                )
            )

        # Financial need requirement
        requires_financial_need = self.prng.random() < 0.25
        max_income = self.prng.choice(
            ["very_low", "low", "middle", "upper_middle"]
        )

        # Selection criteria (normalized weights)
        raw_criteria = {
            "academic": round(self.rng.uniform(0.2, 0.5), 2),
            "leadership": round(self.rng.uniform(0.05, 0.3), 2),
            "olympiad": round(self.rng.uniform(0.1, 0.4), 2),
            "extracurricular": round(self.rng.uniform(0.05, 0.3), 2),
            "essay": round(self.rng.uniform(0.05, 0.3), 2),
        }
        total = sum(raw_criteria.values())
        selection_criteria = SelectionCriteria(
            academic=round(raw_criteria["academic"] / total, 4),
            leadership=round(raw_criteria["leadership"] / total, 4),
            olympiad=round(raw_criteria["olympiad"] / total, 4),
            extracurricular=round(raw_criteria["extracurricular"] / total, 4),
            essay=round(raw_criteria["essay"] / total, 4),
        )

        # Funding coverage
        funding_coverage = FundingCoverage(
            covers_tuition=self.prng.random() > 0.2,
            covers_living_expense=self.prng.random() > 0.3,
            covers_airfare=self.prng.random() > 0.5,
            covers_insurance=self.prng.random() > 0.6,
            monthly_stipend=round(self.rng.uniform(0, 2000), 2),
        )

        # Career preference
        career_track_preference = (
            self.prng.choice([e.value for e in CareerTrack])
            if self.prng.random() > 0.5
            else None
        )

        # Return home requirement
        requires_return = self.prng.random() < 0.3

        # Text fields
        fld = eligible_fields[0] if eligible_fields else "computer_science"
        interest = self.prng.choice(self.research_interests)

        mission = self.prng.choice(self.mission_statement_templates).format(
            field=fld,
            interest=interest,
            region=region.capitalize(),
            goal=self.prng.choice(
                ["societal impact", "innovation", "knowledge advancement"]
            ),
        )

        target_profile = self.prng.choice(self.target_profile_templates).format(
            field=fld,
            activity=self.prng.choice(["research", "leadership"]),
            interest=interest,
            goal=self.prng.choice(
                ["community service", "technological advancement"]
            ),
        )

        name = f"{host_country.replace('_', ' ').title()} {fld.replace('_', ' ').title()} Scholarship"

        return Scholarship(
            scholarship_id=scholarship_id,
            name=name,
            eligible_nationalities=eligible_nationalities,
            min_age=min_age,
            max_age=max_age,
            eligible_degree_levels=eligible_degree_levels,
            eligible_high_school_tracks=eligible_tracks,
            eligible_fields=eligible_fields,
            preferred_school_tier=preferred_tier,
            min_report_card_average=min_report_avg,
            min_major_subject_average=min_major_avg,
            language_requirements=language_requirements,
            requires_financial_need=requires_financial_need,
            max_family_income_category=max_income,
            host_country=host_country,
            host_region=region,
            selection_criteria=selection_criteria,
            funding_coverage=funding_coverage,
            career_track_preference=career_track_preference,
            requires_return_home_country=requires_return,
            mission_statement=mission,
            target_recipient_profile=target_profile,
        )

    def _debug_distribution(
        self,
        students: List[Student],
        scholarships: List[Scholarship],
        sample_students: int = 50,
    ) -> None:
        """Pre-generate sanity check: print histogram of relevance scores."""
        print("\n  [Distribution check] Sampling "
              f"{sample_students} students × {len(scholarships)} scholarships...")
        n = min(sample_students, len(students))
        sampled = self.prng.sample(students, n)
        scores = [
            _scorer(s, sch, self.rng, self.host_countries_by_region)
            for s in sampled
            for sch in scholarships
        ]
        total = len(scores)
        bins = [0] * 10
        for v in scores:
            idx = min(int(v * 10), 9)
            bins[idx] += 1

        print(f"  [Distribution] total={total:,}, "
              f"mean={np.mean(scores):.3f}, std={np.std(scores):.3f}")
        max_count = max(bins) or 1
        for i, c in enumerate(bins):
            bar = "#" * int(40 * c / max_count)
            lo, hi = i / 10, (i + 1) / 10
            pct = 100.0 * c / total if total else 0.0
            print(f"    {lo:.1f}-{hi:.1f} | {c:6,} ({pct:5.2f}%) {bar}")

        match = sum(1 for v in scores if v >= 0.7)
        inb = sum(1 for v in scores if 0.3 <= v < 0.7)
        notm = sum(1 for v in scores if v < 0.3)
        print(f"  [Bands] match={100*match/total:.2f}%  "
              f"in-between={100*inb/total:.2f}%  not-match={100*notm/total:.2f}%")
        if inb / total < 0.05:
            print("  [WARNING] In-Between band <5% — consider tuning weights.")

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

    def generate_balanced_pairs(
        self,
        students: List[Student],
        scholarships: List[Scholarship],
        target_match_count: int = 250_000,
        ratio_inbetween: float = 1.0,
        ratio_notmatch: float = 1.0,
    ) -> List[Pair]:
        """Generate balanced pairs across three categories."""
        base_date = datetime(2024, 1, 1)

        # Pre-generation sanity check
        self._debug_distribution(students, scholarships, sample_students=50)

        print("\n  Computing relevance scores for all candidate pairs...")

        match_pairs = []
        inbetween_pairs = []
        notmatch_pairs = []
        total_combinations = len(students) * len(scholarships)
        print(f"  Total combinations: {total_combinations:,}")

        report_every = max(1, len(students) // 20)
        for idx, student in enumerate(students):
            if idx % report_every == 0:
                print(f"    Processing student {idx}/{len(students)}...")
            for scholarship in scholarships:
                relevance = _scorer(student, scholarship, self.rng, self.host_countries_by_region)
                pair_record = (student.student_id, scholarship.scholarship_id, relevance)
                if relevance >= 0.7:
                    match_pairs.append(pair_record)
                elif relevance >= 0.3:
                    inbetween_pairs.append(pair_record)
                else:
                    notmatch_pairs.append(pair_record)

        print(f"  Found {len(match_pairs):,} match, {len(inbetween_pairs):,} in-between, {len(notmatch_pairs):,} not-match")
        target_inbetween = int(target_match_count * ratio_inbetween)
        target_notmatch = int(target_match_count * ratio_notmatch)
        final_match = self._sample_pairs(match_pairs, target_match_count)
        final_inbetween = self._sample_pairs(inbetween_pairs, target_inbetween)
        final_notmatch = self._sample_pairs(notmatch_pairs, target_notmatch)
        print(f"  Selected: {len(final_match):,} match, {len(final_inbetween):,} in-between, {len(final_notmatch):,} not-match")

        all_pairs = []
        for sid, schid, relevance in final_match + final_inbetween + final_notmatch:
            offset_days = self.rng.randint(0, 365)
            offset_hours = self.rng.randint(0, 23)
            timestamp = (base_date + timedelta(days=offset_days, hours=offset_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
            all_pairs.append(
                Pair(
                    student_id=sid,
                    scholarship_id=schid,
                    relevance_score=relevance,
                    timestamp=timestamp,
                )
            )
        all_pairs.sort(key=lambda p: p.timestamp)
        return all_pairs

    def _sample_pairs(
        self, pairs: List[Tuple[str, str, float]], target_count: int
    ) -> List[Tuple[str, str, float]]:
        """Sample pairs from a list to target count."""
        if len(pairs) <= target_count:
            return pairs

        indices = self.rng.choice(len(pairs), target_count, replace=False)
        return [pairs[i] for i in indices]

    def generate_feedback(
        self,
        students: List[Student],
        scholarships: List[Scholarship],
        pairs: List[Pair],
        num_feedback_per_student: int = 5,
    ) -> List[Feedback]:
        """Generate synthetic implicit feedback for retraining."""
        base_date = datetime(2024, 1, 1)

        # Build lookup: student_id -> list of high-relevance scholarships
        student_to_high_schols: Dict[str, List[str]] = {}
        for p in pairs:
            if p.relevance_score >= 0.7:
                if p.student_id not in student_to_high_schols:
                    student_to_high_schols[p.student_id] = []
                student_to_high_schols[p.student_id].append(p.scholarship_id)

        feedbacks = []

        for student in students:
            sid = student.student_id
            high_schols = student_to_high_schols.get(sid, [])

            num_interactions = self.rng.randint(1, num_feedback_per_student * 3)

            for _ in range(num_interactions):
                if high_schols and self.prng.random() < 0.7:
                    schol_id = self.prng.choice(high_schols)
                    feedback_type = self.prng.choices(
                        ["apply", "click", "view", "reject"],
                        weights=[20, 35, 30, 15],
                    )[0]
                else:
                    schol_id = self.prng.choice(
                        [s.scholarship_id for s in scholarships]
                    )
                    feedback_type = self.prng.choices(
                        ["apply", "click", "view", "reject"],
                        weights=[5, 15, 30, 50],
                    )[0]

                offset_days = self.rng.randint(0, 365)
                offset_hours = self.rng.randint(0, 23)
                timestamp = (
                    base_date + timedelta(days=offset_days, hours=offset_hours)
                ).strftime("%Y-%m-%dT%H:%M:%SZ")

                feedbacks.append(
                    Feedback(
                        student_id=sid,
                        scholarship_id=schol_id,
                        feedback_type=feedback_type,
                        timestamp=timestamp,
                    )
                )

        return feedbacks

    def save_to_csv(
        self,
        students: List[Student],
        scholarships: List[Scholarship],
        pairs: List[Pair],
        feedbacks: List[Feedback],
        output_dir: str = "./datasets_two_tower",
    ):
        """Save all generated data to flat CSV files."""
        os.makedirs(output_dir, exist_ok=True)

        # Flatten student data
        student_records = []
        for s in students:
            record = asdict(s)
            record["language_proficiency"] = json.dumps(
                [asdict(lp) for lp in s.language_proficiency]
            )
            record["olympiad_subjects"] = json.dumps(s.olympiad_subjects)
            record["target_countries"] = json.dumps(s.target_countries)
            student_records.append(record)

        # Flatten scholarship data
        scholarship_records = []
        for sch in scholarships:
            record = asdict(sch)
            record["eligible_nationalities"] = json.dumps(sch.eligible_nationalities)
            record["eligible_degree_levels"] = json.dumps(sch.eligible_degree_levels)
            record["eligible_high_school_tracks"] = json.dumps(
                sch.eligible_high_school_tracks
            )
            record["eligible_fields"] = json.dumps(sch.eligible_fields)
            record["language_requirements"] = json.dumps(
                [asdict(lr) for lr in sch.language_requirements]
            )
            record["selection_criteria"] = json.dumps(
                asdict(sch.selection_criteria)
            )
            fc = sch.funding_coverage
            del record["funding_coverage"]
            record["funding_covers_tuition"] = fc.covers_tuition
            record["funding_covers_living"] = fc.covers_living_expense
            record["funding_covers_airfare"] = fc.covers_airfare
            record["funding_covers_insurance"] = fc.covers_insurance
            record["funding_monthly_stipend"] = fc.monthly_stipend
            record["funding_is_full_funding"] = fc.is_full_funding
            record["funding_coverage_count"] = fc.coverage_count
            scholarship_records.append(record)

        # Flatten pairs
        pair_records = []
        for p in pairs:
            pair_records.append(
                {
                    "student_id": p.student_id,
                    "scholarship_id": p.scholarship_id,
                    "relevance_score": round(p.relevance_score, 4),
                    "timestamp": p.timestamp,
                }
            )

        # Flatten feedback
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
        pd.DataFrame(student_records).to_csv(
            f"{output_dir}/students.csv", index=False
        )
        pd.DataFrame(scholarship_records).to_csv(
            f"{output_dir}/scholarships.csv", index=False
        )
        pd.DataFrame(pair_records).to_csv(f"{output_dir}/pairs.csv", index=False)
        pd.DataFrame(feedback_records).to_csv(
            f"{output_dir}/feedback.csv", index=False
        )

        return output_dir


# ============================================================
# MAIN: Dataset Generation Pipeline
# ============================================================


def main():
    """Generate and save datasets for the two-tower recommendation system."""

    # Configuration
    NUM_STUDENTS = 20_000
    NUM_SCHOLARSHIPS = 800
    SEED = 42
    TARGET_MATCH_COUNT = 250_000

    print("=" * 60)
    print("Two-Tower Recommendation System - Dataset Generator")
    print("=" * 60)
    print("Configuration:")
    print(f"  Students: {NUM_STUDENTS:,}")
    print(f"  Scholarships: {NUM_SCHOLARSHIPS:,}")
    print(f"  Target match pairs: {TARGET_MATCH_COUNT:,}")
    print(f"  Random seed: {SEED}")
    print("=" * 60)

    # Create generator
    generator = TwoTowerDatasetGenerator(
        num_students=NUM_STUDENTS,
        num_scholarships=NUM_SCHOLARSHIPS,
        seed=SEED,
    )

    # Generate all entities
    print("\nGenerating students...")
    students = generator.generate_all_students()
    print(f"  Generated {len(students):,} students")

    print("\nGenerating scholarships...")
    scholarships = generator.generate_all_scholarships()
    print(f"  Generated {len(scholarships):,} scholarships")

    # Generate balanced pairs
    print("\n" + "=" * 60)
    print("Generating balanced training pairs...")
    print("=" * 60)

    pairs = generator.generate_balanced_pairs(
        students,
        scholarships,
        target_match_count=TARGET_MATCH_COUNT,
        ratio_inbetween=1.0,
        ratio_notmatch=1.0,
    )

    # Statistics on relevance score distribution
    scores = [p.relevance_score for p in pairs]
    high = sum(1 for s in scores if s >= 0.7)
    mid = sum(1 for s in scores if 0.3 <= s < 0.7)
    low = sum(1 for s in scores if s < 0.3)
    print(f"\n  Relevance distribution:")
    print(f"    Match (>=0.7):    {high:,}")
    print(f"    In-Between (0.3-0.7): {mid:,}")
    print(f"    Not Match (<0.3): {low:,}")
    print(f"    Total pairs:      {len(pairs):,}")

    # Generate feedback
    print("\nGenerating implicit feedback...")
    feedbacks = generator.generate_feedback(
        students, scholarships, pairs, num_feedback_per_student=5
    )
    print(f"  Generated {len(feedbacks):,} feedback entries")

    # Count feedback types
    type_counts: Dict[str, int] = {}
    for fb in feedbacks:
        type_counts[fb.feedback_type] = type_counts.get(fb.feedback_type, 0) + 1
    for fb_type, count in sorted(type_counts.items()):
        print(f"    {fb_type}: {count:,}")

    # Save to CSV
    output_dir = "./datasets_two_tower"
    generator.save_to_csv(students, scholarships, pairs, feedbacks, output_dir)

    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)

    print(f"\nGenerated files in {output_dir}/:")
    print("    students.csv")
    print("    scholarships.csv")
    print("    pairs.csv (balanced: match, in-between, not-match)")
    print("    feedback.csv (implicit feedback for retraining)")

    print("\nTime-based splitting (done in code):")
    print("    Sort pairs by timestamp → 70% train, 15% val, 15% test")


if __name__ == "__main__":
    main()
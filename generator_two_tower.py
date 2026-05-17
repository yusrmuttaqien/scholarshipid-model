"""Two-Tower Recommendation System - Dataset Generator

Loads real scholarship data directly from src/data/scholarships.py,
then generates synthetic students, realistic pairs, and implicit feedback.

Output structure:
    datasets_two_tower/
    ├── scholarships.csv      # produced by generate_scholarships()
    ├── students.csv          # 20,000 synthetic students
    ├── pairs.csv             # Pairs with continuous relevance_score
    └── feedback.csv          # Implicit feedback for retraining

Relevance bands:
    - Match (>=0.7):       strong attribute alignment (~2% of random pairs)
    - In-Between (0.3-0.7): partial alignment (~18% of random pairs)
    - Not Match (<0.3):    eligibility knockout or no alignment (~80%)

Scoring uses a 5-stage pipeline that leverages every student & scholarship
feature in the schemas: hard eligibility gating → component scores
(academic, olympiad, leadership, extracurricular, essay) → per-scholarship
selection_criteria weighting → lateral fit bonuses (track, field, location,
career, school tier, funding) → diversity boost + small noise.
"""

import json
import os
import random
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.schemas import (
    CareerTrack,
    Country,
    DegreeLevel,
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
        seed: int = 42,
    ):
        self.num_students = num_students
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

    def generate_scholarships(self) -> List[Scholarship]:
        """Load all hardcoded scholarships and assign sequential IDs."""
        from src.data.scholarships import ALL_SCRAPERS
        scholarships = []
        for name, fn in ALL_SCRAPERS:
            try:
                results = fn()
                scholarships.extend(results)
            except Exception as exc:
                print(f"  [WARNING] {name} failed: {exc}")
        for i, sch in enumerate(scholarships):
            sch.scholarship_id = f"SCH_{i:06d}"
        return scholarships

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

    # ------------------------------------------------------------
    # Relevance scoring (5-stage pipeline)
    # ------------------------------------------------------------

    _INCOME_ORDER = ["very_low", "low", "middle", "upper_middle", "high"]
    _TIER_ORDER = [
        "excellence",
        "public_a",
        "private_a",
        "accredited_b",
        "accredited_c",
        "unaccredited",
        "unknown",
    ]
    _OLYMPIAD_LEVEL_SCORE = {
        # "none" is 0.30, not 0: in real selection, lack of olympiad doesn't
        # zero-out the dimension — it's just a low-but-present signal.
        "none": 0.30,
        "school": 0.45,
        "city": 0.6,
        "provincial": 0.75,
        "national": 0.9,
        "international": 1.0,
    }
    # Olympiad subject → MajorField buckets (for cross-relevance with eligible_fields)
    _OLYMPIAD_TO_FIELDS = {
        "mathematics": {"mathematics", "computer_science", "engineering", "physics", "economics"},
        "physics": {"physics", "engineering", "mathematics"},
        "chemistry": {"chemistry", "medicine", "biology", "engineering"},
        "biology": {"biology", "medicine", "agriculture"},
        "economics": {"economics", "business", "social_sciences"},
        "geography": {"social_sciences", "agriculture"},
        "computer_science": {"computer_science", "engineering", "mathematics"},
        "informatics": {"computer_science", "engineering", "mathematics"},
        "linguistics": {"arts_humanities", "education"},
        "astronomy": {"physics", "mathematics"},
        "history": {"arts_humanities", "social_sciences", "education"},
        "english_language": {"arts_humanities", "education"},
        "business_studies": {"business", "economics"},
    }

    def _compute_eligibility_multiplier(
        self, student: Student, scholarship: Scholarship
    ) -> float:
        """Stage 1: hard knockouts. 0.0 = totally ineligible."""
        mult = 1.0

        # Nationality: absolute knockout
        if student.nationality not in scholarship.eligible_nationalities:
            return 0.0

        # Degree level: hard
        if (
            student.current_degree_level not in scholarship.eligible_degree_levels
            and student.target_degree_level not in scholarship.eligible_degree_levels
        ):
            return 0.0

        # Age: hard with off-by-one grace
        if (
            student.age < scholarship.min_age - 1
            or student.age > scholarship.max_age + 1
        ):
            return 0.0
        if not (scholarship.min_age <= student.age <= scholarship.max_age):
            mult *= 0.5

        # Mandatory language: knockout if test missing or score below min
        for req in scholarship.language_requirements:
            if not req.is_mandatory:
                continue
            student_scores = [
                lp.score
                for lp in student.language_proficiency
                if lp.test_type == req.test_type
            ]
            if not student_scores or max(student_scores) < req.min_score:
                return 0.0

        # Financial need ceiling: knockout if income exceeds allowed cap
        if scholarship.requires_financial_need:
            try:
                max_idx = self._INCOME_ORDER.index(scholarship.max_family_income_category)
                student_idx = self._INCOME_ORDER.index(student.family_income_category)
                if student_idx > max_idx:
                    return 0.0
            except ValueError:
                pass

        # Return-home requirement: soft penalty (not absolute, but very strong)
        if scholarship.requires_return_home_country and not student.willing_to_return_home:
            mult *= 0.5

        return mult

    @staticmethod
    def _score_academic(student: Student, scholarship: Scholarship) -> float:
        """Composite academic score using overall, major, math, english vs minimums."""
        def margin(score: float, minimum: float) -> float:
            if score >= minimum:
                # Reward headroom but saturate at +20 above min
                return min(0.7 + (score - minimum) / 20.0 * 0.3, 1.0)
            deficit = (minimum - score) / 20.0
            return max(0.7 - deficit * 0.7, 0.0)

        overall = margin(student.overall_report_card_average, scholarship.min_report_card_average)
        major = margin(student.major_subject_average, scholarship.min_major_subject_average)
        # math/english as auxiliary signals (use min as proxy threshold)
        math = margin(student.math_score, scholarship.min_major_subject_average)
        english = margin(student.english_score, scholarship.min_report_card_average)
        return 0.35 * overall + 0.35 * major + 0.15 * math + 0.15 * english

    def _score_olympiad(self, student: Student, scholarship: Scholarship) -> float:
        """Olympiad level × relevance of subjects to scholarship's eligible fields."""
        level_score = self._OLYMPIAD_LEVEL_SCORE.get(student.olympiad_level, 0.0)
        if level_score == 0.0 or not student.olympiad_subjects:
            return level_score  # 0 for none; level w/o subjects gets baseline

        eligible_fields = set(scholarship.eligible_fields)
        if not eligible_fields:
            return level_score * 0.6  # generic

        # Subject relevance: fraction of subjects mapping into eligible fields
        relevant = 0
        for subj in student.olympiad_subjects:
            mapped = self._OLYMPIAD_TO_FIELDS.get(subj, set())
            if mapped & eligible_fields:
                relevant += 1
        subj_relevance = relevant / len(student.olympiad_subjects)
        # 50% baseline + 50% subject-aware so even off-topic olympiad gets some credit
        return level_score * (0.5 + 0.5 * subj_relevance)

    @staticmethod
    def _score_leadership(student: Student) -> float:
        """Saturating function over leadership_experience_count.

        0→0.25 (baseline — most students have some leadership signal),
        1→0.55, 2→0.73, 3→0.84, 4→0.90, 5+→saturates ~1.0
        """
        n = max(student.leadership_experience_count, 0)
        return min(1.0, 0.25 + 0.75 * (1.0 - (0.6 ** n)))

    @staticmethod
    def _score_extracurricular(student: Student) -> float:
        """Composite of volunteer experience + competition wins (with baselines)."""
        v = max(student.volunteer_experience_count, 0)
        c = max(student.competition_wins_count, 0)
        vol = min(1.0, 0.25 + 0.75 * (1.0 - (0.65 ** v)))
        comp = min(1.0, 0.20 + 0.80 * (1.0 - (0.55 ** c)))
        return 0.6 * vol + 0.4 * comp

    def _score_essay_placeholder(self) -> float:
        """Stub: text-similarity needs embeddings. Mild-optimistic 0.6 + noise.

        For eligible pairs (the only ones essay actually weighs in for),
        average admissible essays score around 0.5–0.7.
        """
        return float(np.clip(0.6 + self.rng.normal(0, 0.08), 0.0, 1.0))

    def _score_language_bonus(self, student: Student, scholarship: Scholarship) -> float:
        """Language requirement: non-knockout portion (mandatory already gated in Stage 1)."""
        if not scholarship.language_requirements:
            return 1.0
        scores = []
        for req in scholarship.language_requirements:
            student_scores = [
                lp.score
                for lp in student.language_proficiency
                if lp.test_type == req.test_type
            ]
            if student_scores:
                max_s = max(student_scores)
                if req.min_score > 0:
                    scores.append(min(1.0, max_s / req.min_score))
                else:
                    scores.append(1.0)
            elif not req.is_mandatory:
                # Optional req not taken: mild penalty
                scores.append(0.5)
        return sum(scores) / len(scores) if scores else 1.0

    def _compute_fit_bonuses(
        self, student: Student, scholarship: Scholarship
    ) -> float:
        """Stage 4: lateral preferences. Returns weighted average 0–1."""

        # Track fit
        if student.high_school_track in scholarship.eligible_high_school_tracks:
            track_fit = 1.0
        else:
            track_fit = 0.3

        # Field fit (proxy via olympiad subjects)
        eligible_fields = set(scholarship.eligible_fields)
        if not eligible_fields or not student.olympiad_subjects:
            field_fit = 0.5  # neutral when no signal
        else:
            mapped_fields: set = set()
            for subj in student.olympiad_subjects:
                mapped_fields |= self._OLYMPIAD_TO_FIELDS.get(subj, set())
            if mapped_fields & eligible_fields:
                field_fit = 1.0
            elif mapped_fields:
                field_fit = 0.3
            else:
                field_fit = 0.5

        # Location fit: host_country in target_countries (strong), else region match
        if scholarship.host_country and scholarship.host_country in student.target_countries:
            loc_fit = 1.0
        elif scholarship.host_region:
            student_target_regions = {
                self.host_countries_by_region_lookup(c) for c in student.target_countries
            }
            student_target_regions.discard(None)
            if scholarship.host_region in student_target_regions:
                loc_fit = 0.6
            else:
                loc_fit = 0.2
        else:
            loc_fit = 0.5

        # Career fit
        if not scholarship.career_track_preference:
            career_fit = 0.6  # no preference → neutral-positive
        elif student.intended_career_track == scholarship.career_track_preference:
            career_fit = 1.0
        else:
            career_fit = 0.4

        # School tier fit (ordinal distance)
        try:
            pref_idx = self._TIER_ORDER.index(scholarship.preferred_school_tier)
            stu_idx = self._TIER_ORDER.index(student.school_tier)
            # Same/better tier = full credit; each step worse = -0.15
            if stu_idx <= pref_idx:
                tier_fit = 1.0
            else:
                tier_fit = max(0.2, 1.0 - 0.15 * (stu_idx - pref_idx))
        except ValueError:
            tier_fit = 0.6

        # Funding fit
        is_full = scholarship.funding_coverage.is_full_funding
        if student.needs_full_funding:
            fund_fit = 1.0 if is_full else 0.3
        elif student.can_self_fund_living:
            # Self-funded student: any scholarship works; full-funding still bonus
            fund_fit = 0.9 if is_full else 0.8
        else:
            fund_fit = 0.7 if is_full else 0.6

        return (
            0.20 * track_fit
            + 0.18 * field_fit
            + 0.18 * loc_fit
            + 0.14 * career_fit
            + 0.10 * tier_fit
            + 0.20 * fund_fit
        )

    def host_countries_by_region_lookup(self, country: str):
        """Reverse-lookup region from country (uses host_countries_by_region map)."""
        for region, countries in self.host_countries_by_region.items():
            if country in countries:
                return region
        return None

    def compute_relevance_score(
        self, student: Student, scholarship: Scholarship
    ) -> float:
        """Compute continuous relevance score (0.0-1.0) for two-tower regression.

        5-stage pipeline:
          1. Hard eligibility gating (nationality, degree, age, mandatory lang, income ceiling)
          2. Component scores (academic, olympiad, leadership, extracurricular, essay)
          3. Core score = sum(selection_criteria.X * component_X) — per-scholarship weights
          4. Fit bonuses (track, field, location, career, school tier, funding)
          5. Combine + diversity boost + small noise, then gate by eligibility multiplier
        """
        # Stage 1: hard eligibility
        elig = self._compute_eligibility_multiplier(student, scholarship)
        if elig == 0.0:
            # Even fully ineligible pairs get tiny noise so the model sees some
            # variance at the low end, but stay well below the In-Between threshold.
            return float(np.clip(self.rng.uniform(0.0, 0.05), 0.0, 1.0))

        # Stage 2: component scores
        acad = self._score_academic(student, scholarship)
        oly = self._score_olympiad(student, scholarship)
        lead = self._score_leadership(student)
        extra = self._score_extracurricular(student)
        essay = self._score_essay_placeholder()

        # Stage 3: per-scholarship selection_criteria weighting (already sums to ~1.0)
        sc = scholarship.selection_criteria
        core = (
            sc.academic * acad
            + sc.olympiad * oly
            + sc.leadership * lead
            + sc.extracurricular * extra
            + sc.essay * essay
        )

        # Stage 4: fit bonuses + language non-knockout signal
        fit = self._compute_fit_bonuses(student, scholarship)
        lang_bonus = self._score_language_bonus(student, scholarship)

        # Stage 5: combine
        base = 0.55 * core + 0.35 * fit + 0.10 * lang_bonus

        # Diversity boost: underrepresented students for scholarships that favor them
        favors_diversity = (
            scholarship.requires_financial_need
            or scholarship.host_country in {"japan", "south_korea", "uk", "usa", "australia"}
        )
        if student.from_underrepresented_region and favors_diversity:
            base = min(1.0, base + 0.05)

        # Apply eligibility multiplier (1.0 normally, 0.3 for grace-zone violations)
        relevance = base * elig

        # Small realistic noise
        relevance = float(np.clip(relevance + self.rng.normal(0, 0.02), 0.0, 1.0))

        return round(relevance, 4)

    def _debug_distribution(
        self,
        students: List[Student],
        scholarships: List[Scholarship],
        sample_students: int = 50,
    ) -> None:
        """Pre-generate sanity check: print histogram of relevance scores.

        Samples N students × all scholarships, prints 10-bin histogram so we
        can verify In-Between band (0.3–0.7) is not collapsed before committing
        to full 800k-pair generation.
        """
        print("\n  [Distribution check] Sampling "
              f"{sample_students} students × {len(scholarships)} scholarships...")
        n = min(sample_students, len(students))
        sampled = self.prng.sample(students, n)
        scores = [
            self.compute_relevance_score(s, sch)
            for s in sampled
            for sch in scholarships
        ]
        total = len(scores)
        bins = [0] * 10  # 0.0-0.1, 0.1-0.2, ..., 0.9-1.0
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
            print("  [WARNING] In-Between band <5% — consider tuning weights "
                  "before full generation.")

    def generate_all_students(self) -> List[Student]:
        """Generate all students for the single pool."""
        students = []
        for i in range(self.num_students):
            student_id = f"STU_{i:06d}"
            students.append(self.generate_student(student_id))
        return students

    def generate_balanced_pairs(
        self,
        students: List[Student],
        scholarships: List[Scholarship],
        target_match_count: int = 250_000,
        ratio_inbetween: float = 1.0,
        ratio_notmatch: float = 1.0,
    ) -> List[Pair]:
        """Generate balanced pairs across three categories.

        Categories:
            - Match (high relevance ~0.8-1.0): Strong attribute alignment
            - In-Between (medium relevance ~0.3-0.6): Partial alignment
            - Not Match (low relevance ~0.0-0.2): No alignment

        Args:
            students: All students in the pool.
            scholarships: All scholarships in the pool.
            target_match_count: Target number of high-relevance pairs.
            ratio_inbetween: Ratio of in-between pairs to match pairs.
            ratio_notmatch: Ratio of not-match pairs to match pairs.
        """
        base_date = datetime(2024, 1, 1)

        # Pre-generation sanity check: histogram on a small sample
        self._debug_distribution(students, scholarships, sample_students=50)

        # Step 1: Compute relevance for ALL student-scholarship combinations
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
                relevance = self.compute_relevance_score(student, scholarship)

                pair_record = (student.student_id, scholarship.scholarship_id, relevance)

                if relevance >= 0.7:
                    match_pairs.append(pair_record)
                elif relevance >= 0.3:
                    inbetween_pairs.append(pair_record)
                else:
                    notmatch_pairs.append(pair_record)

        # Step 2: Sample balanced pairs from each category
        print(f"  Found {len(match_pairs):,} match, {len(inbetween_pairs):,} in-between, {len(notmatch_pairs):,} not-match")

        target_inbetween = int(target_match_count * ratio_inbetween)
        target_notmatch = int(target_match_count * ratio_notmatch)

        # Sample from each category
        final_match = self._sample_pairs(match_pairs, target_match_count)
        final_inbetween = self._sample_pairs(inbetween_pairs, target_inbetween)
        final_notmatch = self._sample_pairs(notmatch_pairs, target_notmatch)

        print(f"  Selected: {len(final_match):,} match, {len(final_inbetween):,} in-between, {len(final_notmatch):,} not-match")

        # Step 3: Build Pair records with timestamps
        all_pairs = []

        for sid, schid, relevance in final_match:
            offset_days = self.rng.randint(0, 365)
            offset_hours = self.rng.randint(0, 23)
            timestamp = (
                base_date + timedelta(days=offset_days, hours=offset_hours)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            all_pairs.append(
                Pair(
                    student_id=sid,
                    scholarship_id=schid,
                    relevance_score=relevance,
                    timestamp=timestamp,
                )
            )

        for sid, schid, relevance in final_inbetween:
            offset_days = self.rng.randint(0, 365)
            offset_hours = self.rng.randint(0, 23)
            timestamp = (
                base_date + timedelta(days=offset_days, hours=offset_hours)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            all_pairs.append(
                Pair(
                    student_id=sid,
                    scholarship_id=schid,
                    relevance_score=relevance,
                    timestamp=timestamp,
                )
            )

        for sid, schid, relevance in final_notmatch:
            offset_days = self.rng.randint(0, 365)
            offset_hours = self.rng.randint(0, 23)
            timestamp = (
                base_date + timedelta(days=offset_days, hours=offset_hours)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            all_pairs.append(
                Pair(
                    student_id=sid,
                    scholarship_id=schid,
                    relevance_score=relevance,
                    timestamp=timestamp,
                )
            )

        # Sort by timestamp for time-based splitting
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

    @staticmethod
    def _flatten_scholarship(sch: Scholarship) -> dict:
        """Flatten a Scholarship dataclass to a CSV-compatible dict."""
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

    def save_to_csv(
        self,
        students: List[Student],
        scholarships: List[Scholarship],
        pairs: List[Pair],
        feedbacks: List[Feedback],
        output_dir: str = "./datasets_two_tower",
    ):
        """Save all datasets to CSV files."""
        os.makedirs(output_dir, exist_ok=True)

        # Save scholarships
        scholarship_records = [self._flatten_scholarship(sch) for sch in scholarships]
        pd.DataFrame(scholarship_records).to_csv(f"{output_dir}/scholarships.csv", index=False)

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

        pd.DataFrame(student_records).to_csv(f"{output_dir}/students.csv", index=False)
        pd.DataFrame(pair_records).to_csv(f"{output_dir}/pairs.csv", index=False)
        pd.DataFrame(feedback_records).to_csv(f"{output_dir}/feedback.csv", index=False)

        return output_dir


# ============================================================
# MAIN: Dataset Generation Pipeline
# ============================================================


def main():
    NUM_STUDENTS = 20_000
    SEED = 42
    # With the 5-stage realistic scorer, ~1.8% of random pairs land in the
    # match band (relevance >= 0.7). For 20k students × ~40 scholarships =
    # 800k candidate pairs, that yields ~14–16k matches. We cap at 20k and
    # take all available, then sample 3x as many in-between / not-match pairs
    # to keep the dataset roughly proportional to real-world distribution
    # while still giving the model enough match examples to learn from.
    TARGET_MATCH_COUNT = 20_000
    RATIO_INBETWEEN = 3.0
    RATIO_NOTMATCH = 3.0
    OUTPUT_DIR = "./datasets_two_tower"

    print("=" * 60)
    print("Two-Tower Dataset Generator v1")
    print("=" * 60)

    generator = TwoTowerDatasetGenerator(num_students=NUM_STUDENTS, seed=SEED)

    # Load scholarships
    print("\nLoading scholarships...")
    scholarships = generator.generate_scholarships()
    print(f"  Loaded {len(scholarships):,} scholarships")

    # Generate students
    print(f"\nGenerating {NUM_STUDENTS:,} students...")
    students = generator.generate_all_students()
    print(f"  Generated {len(students):,} students")

    # Generate balanced pairs
    print("\n" + "=" * 60)
    print("Generating balanced training pairs...")
    print("=" * 60)

    pairs = generator.generate_balanced_pairs(
        students, scholarships,
        target_match_count=TARGET_MATCH_COUNT,
        ratio_inbetween=RATIO_INBETWEEN,
        ratio_notmatch=RATIO_NOTMATCH,
    )

    scores = [p.relevance_score for p in pairs]
    print(f"\n  Relevance distribution:")
    print(f"    Match (>=0.7):        {sum(1 for s in scores if s >= 0.7):,}")
    print(f"    In-Between (0.3-0.7): {sum(1 for s in scores if 0.3 <= s < 0.7):,}")
    print(f"    Not Match (<0.3):     {sum(1 for s in scores if s < 0.3):,}")
    print(f"    Total pairs:          {len(pairs):,}")

    # Generate feedback
    print("\nGenerating implicit feedback...")
    feedbacks = generator.generate_feedback(students, scholarships, pairs, num_feedback_per_student=5)
    print(f"  Generated {len(feedbacks):,} feedback entries")

    type_counts: Dict[str, int] = {}
    for fb in feedbacks:
        type_counts[fb.feedback_type] = type_counts.get(fb.feedback_type, 0) + 1
    for fb_type, count in sorted(type_counts.items()):
        print(f"    {fb_type}: {count:,}")

    # Save all datasets
    generator.save_to_csv(students, scholarships, pairs, feedbacks, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)
    print(f"\nFiles saved to {OUTPUT_DIR}/:")
    for fname in ["scholarships.csv", "students.csv", "pairs.csv", "feedback.csv"]:
        print(f"  {fname}")


if __name__ == "__main__":
    main()
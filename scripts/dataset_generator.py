import random
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict
from dataclasses import asdict
import numpy as np
import pandas as pd

from src.generator.convert_csv import save_to_csv

from src.generator.data_seeds import (
    all_scrapers,
    HighSchoolTrack,
    LanguageTest,
    OlympiadLevel,
    OlympiadSubject,
    SchoolTier,
    IncomeCategory,
    CareerTrack,
    MajorField,
    COUNTRIES_BY_REGION,
    ALL_COUNTRIES,
    ACHIEVEMENT_TEMPLATES,
    FUTURE_GOALS_TEMPLATES,
    PERSONAL_STATEMENT_TEMPLATES,
    RESEARCH_INTERESTS,
    SCHOOL_NAMES,
)

from src.generator.schemas import (
    Feedback,
    LanguageProficiency,
    Scholarship,
    Student
)

_HIGH_AFF  = (0.25, 0.65, 0.10)
_MID_AFF   = (0.02, 0.50, 0.48)
_LOW_AFF   = (0.001, 0.25, 0.749)
_FB_TYPES  = ["accepted", "apply", "click"]


class DatasetGenerator:
    def __init__(self, num_students: int = 20_000, seed: int = 42):
        self.num_students = num_students
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.prng = random.Random(seed)

        self.all_countries = ALL_COUNTRIES
        self.school_names = SCHOOL_NAMES
        self.personal_statement_templates = PERSONAL_STATEMENT_TEMPLATES
        self.future_goals_templates = FUTURE_GOALS_TEMPLATES
        self.research_interests = RESEARCH_INTERESTS
        self.achievement_templates = ACHIEVEMENT_TEMPLATES

    def generate_scholarships(self) -> List[Scholarship]:
        scholarships = []
        for name, fn in all_scrapers:
            try:
                results = fn()
                scholarships.extend(results)
            except Exception as exc:
                print(f"  [WARNING] {name} failed: {exc}")
        for i, sch in enumerate(scholarships):
            sch.scholarship_id = f"SCH_{i:06d}"
        return scholarships

    def generate_student(self, student_id: str) -> Student:
        _NATIONALITY_WEIGHTS: Dict[str, float] = {
            "indonesia": 30.0, "philippines": 8.0, "vietnam": 8.0, "malaysia": 7.0,
            "thailand": 7.0, "india": 5.0, "china": 5.0, "south_korea": 3.0,
            "japan": 2.0, "singapore": 1.0, "nigeria": 3.0, "kenya": 3.0,
            "morocco": 2.0, "south_africa": 2.0, "egypt": 2.0, "brazil": 1.5,
            "argentina": 1.0, "chile": 1.0, "france": 0.5, "germany": 0.5,
            "netherlands": 0.5, "sweden": 0.5, "uk": 0.5, "switzerland": 0.5,
            "canada": 0.5, "usa": 0.5, "australia": 0.5, "new_zealand": 0.5,
        }

        countries = list(_NATIONALITY_WEIGHTS.keys())
        weights = list(_NATIONALITY_WEIGHTS.values())
        nationality = self.prng.choices(countries, weights=weights)[0]
        age = self.rng.randint(16, 18)

        track = self.prng.choice([e.value for e in HighSchoolTrack])

        overall_avg = round(self.rng.uniform(60, 98), 1)
        math_score = round(self.rng.uniform(50, 100), 1)
        english_score = round(self.rng.uniform(40, 100), 1)
        major_avg = round(self.rng.uniform(55, 100), 1)

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

        leadership_count = self.rng.poisson(2)
        volunteer_count = self.rng.poisson(3)
        competition_wins = self.rng.poisson(1)

        school_tier = self.prng.choices(
            [e.value for e in SchoolTier],
            weights=[5, 20, 15, 25, 20, 10, 5],
        )[0]
        income = self.prng.choices(
            [e.value for e in IncomeCategory],
            weights=[10, 25, 35, 20, 10],
        )[0]
        underrepresented = self.prng.random() < 0.25

        career_track = self.prng.choice([e.value for e in CareerTrack])
        willing_return = self.prng.random() > 0.2

        num_targets = self.rng.randint(1, 4)
        target_countries = self.prng.sample(
            self.all_countries, min(num_targets, len(self.all_countries))
        )

        needs_full_funding = self.prng.random() < 0.6
        can_self_fund_living = not needs_full_funding and self.prng.random() > 0.5

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

        schools = self.school_names.get(nationality, ["General High School"])
        school = self.prng.choice(schools)

        return Student(
            student_id=student_id,
            nationality=nationality,
            age=age,
            current_degree_level="high_school",
            target_degree_level="bachelors",
            high_school_track=track,
            school_name=school,
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

    def generate_students(self) -> List[Student]:
        students = []
        for i in range(self.num_students):
            student_id = f"STU_{i:06d}"
            students.append(self.generate_student(student_id))
        return students

    def _simulate_affinity(
        self, student: Student, scholarship: Scholarship
    ) -> float:
        _COUNTRY_REGION: Dict[str, str] = {
            country: region
            for region, countries in COUNTRIES_BY_REGION.items()
            for country in countries
        }

        # --- elig component ---
        eligible = True
        if (
            scholarship.eligible_nationalities
            and student.nationality not in scholarship.eligible_nationalities
        ):
            eligible = False
        if student.age < scholarship.min_age or student.age > scholarship.max_age:
            eligible = False
        if (
            scholarship.eligible_degree_levels
            and student.current_degree_level not in scholarship.eligible_degree_levels
        ):
            eligible = False
        elig = 1.0 if eligible else 0.05

        # --- field component ---
        if not scholarship.eligible_high_school_tracks:
            field = 0.88  # no restriction → mild bonus
        elif student.high_school_track in scholarship.eligible_high_school_tracks:
            field = 1.0
        else:
            field = 0.60

        # --- country component ---
        sch_region = _COUNTRY_REGION.get(scholarship.host_country, "")
        stu_region = _COUNTRY_REGION.get(student.nationality, "")
        if student.target_countries and scholarship.host_country in student.target_countries:
            country = 1.0
        elif sch_region and stu_region and sch_region == stu_region:
            country = 0.85  # same region as student's home country
        else:
            country = 0.72

        # --- funding component ---
        if student.needs_full_funding:
            funding = 1.0 if scholarship.funding_coverage.is_full_funding else 0.75
        else:
            funding = 1.0 if not scholarship.requires_financial_need else 0.88

        # --- academic component ---
        # 0.68 at/below min, rises to 1.0 at +15 pts above min
        academic_gap = (
            student.overall_report_card_average
            - scholarship.min_report_card_average
        )
        academic = 0.68 + 0.32 * float(np.clip((academic_gap + 10.0) / 25.0, 0.0, 1.0))

        affinity = elig * field * country * funding * academic
        return float(np.clip(affinity, 0.0, 1.0))

    def generate_feedback(
        self,
        students: List[Student],
        scholarships: List[Scholarship],
        n_per_student: int = 5,
    ) -> List[Feedback]:
        base_date = datetime(2024, 1, 1)
        sch_ids = [s.scholarship_id for s in scholarships]
        n_sch = len(scholarships)
        feedbacks: List[Feedback] = []

        for student in students:
            affinities = np.array(
                [self._simulate_affinity(student, sch) for sch in scholarships],
                dtype=np.float64,
            )

            affinities = np.clip(affinities, 1e-9, None)
            probs = affinities / affinities.sum()

            n_pick = min(n_per_student, n_sch)
            chosen_indices = self.rng.choice(n_sch, size=n_pick, replace=False, p=probs)

            for idx in chosen_indices:
                aff = affinities[idx]
                if aff >= 0.6:
                    weights = _HIGH_AFF
                elif aff >= 0.3:
                    weights = _MID_AFF
                else:
                    weights = _LOW_AFF

                fb_type = self.prng.choices(_FB_TYPES, weights=weights)[0]

                offset_days = self.rng.randint(0, 365)
                offset_hours = self.rng.randint(0, 23)
                timestamp = (
                    base_date + timedelta(days=int(offset_days), hours=int(offset_hours))
                ).strftime("%Y-%m-%dT%H:%M:%SZ")

                feedbacks.append(
                    Feedback(
                        student_id=student.student_id,
                        scholarship_id=sch_ids[idx],
                        feedback_type=fb_type,
                        timestamp=timestamp,
                    )
                )

        return feedbacks

def main():
    numStudents = 20000
    generatorEngine = DatasetGenerator(num_students=numStudents, seed=42)
    
    scholarshipsList = generatorEngine.generate_scholarships()
    studentsList = generatorEngine.generate_students()
    feedbacksList = generatorEngine.generate_feedback(studentsList, scholarshipsList)
    
    save_to_csv(studentsList, scholarshipsList, feedbacksList)
    print("Dataset generation complete!")


if __name__ == "__main__":
    main()

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

import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np

from src.data.students import (
    ALL_COUNTRIES,
    ACHIEVEMENT_TEMPLATES,
    COUNTRIES_BY_REGION,
    FUTURE_GOALS_TEMPLATES,
    PERSONAL_STATEMENT_TEMPLATES,
    RESEARCH_INTERESTS,
    SCHOOL_NAMES,
)
from src.schemas import (
    CareerTrack,
    Feedback,
    HighSchoolTrack,
    IncomeCategory,
    LanguageProficiency,
    LanguageTest,
    MajorField,
    OlympiadLevel,
    OlympiadSubject,
    Pair,
    Scholarship,
    SchoolTier,
    Student,
)
from src.io import save_to_csv
from src.scorer import compute_relevance_score


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
        self.all_countries = ALL_COUNTRIES
        self.host_countries_by_region = COUNTRIES_BY_REGION
        self.school_names = SCHOOL_NAMES
        self.personal_statement_templates = PERSONAL_STATEMENT_TEMPLATES
        self.future_goals_templates = FUTURE_GOALS_TEMPLATES
        self.research_interests = RESEARCH_INTERESTS
        self.achievement_templates = ACHIEVEMENT_TEMPLATES

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
            compute_relevance_score(s, sch, self.rng, self.host_countries_by_region)
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
                relevance = compute_relevance_score(student, scholarship, self.rng, self.host_countries_by_region)

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
        for sid, schid, relevance in final_match + final_inbetween + final_notmatch:
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
    save_to_csv(students, scholarships, pairs, feedbacks, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)
    print(f"\nFiles saved to {OUTPUT_DIR}/:")
    for fname in ["scholarships.csv", "students.csv", "pairs.csv", "feedback.csv"]:
        print(f"  {fname}")


if __name__ == "__main__":
    main()
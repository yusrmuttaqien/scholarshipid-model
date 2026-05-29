"""
Feature engineering untuk Two-Tower recommendation system.

Semua encoding logic ada di sini — dipanggil oleh data_loader.py dan inference_engine.py.
"""
import ast
import json

import numpy as np

# ── Vocabulary (fixed dari synthetic data) ────────────────────────────────────

ALL_COUNTRIES = [
    "indonesia", "malaysia", "thailand", "philippines", "vietnam", "singapore",
    "japan", "south_korea", "china", "india",
    "france", "germany", "netherlands", "sweden", "uk", "switzerland",
    "canada", "usa", "argentina", "brazil", "chile",
    "egypt", "kenya", "morocco", "nigeria", "south_africa",
    "australia", "new_zealand",
]
ALL_TRACKS   = ["science", "social_studies", "languages", "religion", "vocational"]
ALL_FIELDS   = [
    "computer_science", "engineering", "medicine", "business", "economics",
    "law", "education", "arts_humanities", "social_sciences", "agriculture",
    "mathematics", "physics", "chemistry", "biology",
]
ALL_OLY_SUBJ = [
    "mathematics", "physics", "chemistry", "biology", "economics", "geography",
    "computer_science", "linguistics", "astronomy", "informatics",
    "history", "english_language", "business_studies",
]
ALL_TIERS    = ["excellence", "public_a", "private_a", "accredited_b",
                "accredited_c", "unaccredited", "unknown"]
ALL_INCOME   = ["very_low", "low", "middle", "upper_middle", "high"]
ALL_CAREERS  = ["academic", "industry", "government", "ngo_npo", "entrepreneurship", "public_service"]
ALL_OLY_LVL  = ["none", "school", "city", "provincial", "national", "international"]
ALL_LANG_TESTS = ["toefl", "ielts", "topik", "jlpt", "delf", "hsk"]
ALL_DEGREE   = ["high_school", "bachelors"]
ALL_REGIONS  = ["asia", "europe", "north_america", "south_america", "africa", "oceania"]

LANG_SCORE_MAX = {"toefl": 120, "ielts": 9, "topik": 300, "jlpt": 100, "delf": 100, "hsk": 300}

# ── Dimension constants ───────────────────────────────────────────────────────

STU_STRUCT_DIM = 122
SCH_STRUCT_DIM = 125
TEXT_EMB_DIM   = 384
STU_INPUT_DIM  = STU_STRUCT_DIM + TEXT_EMB_DIM  # 506
SCH_INPUT_DIM  = SCH_STRUCT_DIM + TEXT_EMB_DIM  # 509

SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

# ── JSON column normalization ────────────────────────────────────────────────


def normalize_json_columns(df, json_cols: list[str]):
    """Normalize JSON columns in a DataFrame by parsing string representations.

    When data is loaded from CSV via pd.read_csv(), JSON columns come back as
    strings (e.g., '["indonesia"]'). This function ensures all values are properly
    decoded Python objects (lists, dicts).

    Handles both standard JSON ('{"key": "value"}') and Python-style dicts
    with single quotes ("{'key': 'value'}").
    """
    for col in json_cols:
        if col not in df.columns:
            continue

        def _parse(val):
            if isinstance(val, str):
                val = val.strip()
                if val.startswith("["):
                    try:
                        return json.loads(val)
                    except (json.JSONDecodeError, ValueError):
                        pass
                elif val.startswith("{"):
                    try:
                        return json.loads(val)
                    except (json.JSONDecodeError, ValueError):
                        pass
                    try:
                        return ast.literal_eval(val)
                    except (ValueError, SyntaxError):
                        pass
            return val

        df[col] = df[col].apply(_parse)

    return df


# ── Encoding helpers ──────────────────────────────────────────────────────────

def one_hot(val, vocab):
    v = [0.0] * len(vocab)
    if val in vocab:
        v[vocab.index(val)] = 1.0
    return v


def multi_hot(vals, vocab):
    v = [0.0] * len(vocab)
    for x in (vals or []):
        if x in vocab:
            v[vocab.index(x)] = 1.0
    return v


def norm_clip(val, lo, hi):
    return float(np.clip((val - lo) / (hi - lo + 1e-9), 0.0, 1.0))


# ── Student encoder ───────────────────────────────────────────────────────────

def encode_student(row: dict) -> list:
    """Encode satu student row → flat float vector (122-dim).

    Expects JSON columns (language_proficiency, olympiad_subjects, target_countries)
    sudah di-parse jadi Python objects (list/dict), bukan string.
    """
    feats = []
    feats += one_hot(row["nationality"], ALL_COUNTRIES)                      # 28
    feats += [norm_clip(row["age"], 16, 18)]                                  # 1
    feats += one_hot(row["high_school_track"], ALL_TRACKS)                   # 5
    feats += [norm_clip(row["overall_report_card_average"], 0, 100)]         # 1
    feats += [norm_clip(row["math_score"], 0, 100)]                          # 1
    feats += [norm_clip(row["english_score"], 0, 100)]                       # 1
    feats += [norm_clip(row["major_subject_average"], 0, 100)]               # 1

    # language_proficiency: has_test + norm_score per test type              # 12
    lang_prof = row["language_proficiency"] or []
    lang_map = {
        lp.get("test_type"): lp.get("score", 0)
        for lp in lang_prof
        if isinstance(lp, dict) and lp.get("test_type")
    }
    for t in ALL_LANG_TESTS:
        if t in lang_map:
            feats += [1.0, norm_clip(lang_map[t], 0, LANG_SCORE_MAX[t])]
        else:
            feats += [0.0, 0.0]

    feats += one_hot(row["olympiad_level"], ALL_OLY_LVL)                    # 6
    feats += multi_hot(row["olympiad_subjects"] or [], ALL_OLY_SUBJ)        # 13
    feats += [norm_clip(row["leadership_experience_count"], 0, 10)]         # 1
    feats += [norm_clip(row["volunteer_experience_count"], 0, 15)]          # 1
    feats += [norm_clip(row["competition_wins_count"], 0, 10)]              # 1
    feats += one_hot(row["school_tier"], ALL_TIERS)                         # 7
    feats += one_hot(row["family_income_category"], ALL_INCOME)             # 5
    feats += [float(row["from_underrepresented_region"])]                    # 1
    feats += one_hot(row["intended_career_track"], ALL_CAREERS)             # 6
    feats += [float(row["willing_to_return_home"])]                          # 1
    feats += multi_hot(row["target_countries"] or [], ALL_COUNTRIES)        # 28
    feats += [float(row["needs_full_funding"])]                              # 1
    feats += [float(row["can_self_fund_living"])]                            # 1
    return feats                                                              # total: 122


# ── Scholarship encoder ───────────────────────────────────────────────────────

def encode_scholarship(row: dict) -> list:
    """Encode satu scholarship row → flat float vector (125-dim).

    Expects JSON columns sudah di-parse jadi Python objects.
    """
    feats = []
    feats += multi_hot(row["eligible_nationalities"] or [], ALL_COUNTRIES)   # 28
    feats += [norm_clip(row["min_age"], 14, 30),
              norm_clip(row["max_age"], 14, 30)]                             # 2
    feats += multi_hot(row["eligible_degree_levels"] or [], ALL_DEGREE)     # 2
    feats += multi_hot(row["eligible_high_school_tracks"] or [], ALL_TRACKS) # 5
    feats += multi_hot(row["eligible_fields"] or [], ALL_FIELDS)            # 14
    feats += one_hot(row["preferred_school_tier"], ALL_TIERS)               # 7
    feats += [norm_clip(row["min_report_card_average"], 0, 100),
              norm_clip(row["min_major_subject_average"], 0, 100)]          # 2

    # language requirements: min_score per test type                         # 6
    lang_reqs = row["language_requirements"] or []
    req_map = {
        lr.get("test_type"): lr.get("min_score", 0)
        for lr in lang_reqs
        if isinstance(lr, dict) and lr.get("test_type")
    }
    for t in ALL_LANG_TESTS:
        feats += [norm_clip(req_map.get(t, 0), 0, LANG_SCORE_MAX[t])]

    feats += [float(row["requires_financial_need"])]                         # 1
    feats += one_hot(row["max_family_income_category"], ALL_INCOME)         # 5
    feats += one_hot(row["host_country"], ALL_COUNTRIES)                    # 28
    feats += one_hot(row["host_region"], ALL_REGIONS)                       # 6

    sc = row["selection_criteria"] or {}
    feats += [sc.get("academic", 0.0), sc.get("leadership", 0.0),
              sc.get("olympiad", 0.0), sc.get("extracurricular", 0.0),
              sc.get("essay", 0.0)]                                          # 5

    feats += [float(row["funding_covers_tuition"]),
              float(row["funding_covers_living"]),
              float(row["funding_covers_airfare"]),
              float(row["funding_covers_insurance"])]                        # 4
    feats += [norm_clip(row["funding_monthly_stipend"], 0, 200_000)]        # 1
    feats += [float(row["funding_is_full_funding"])]                         # 1
    feats += [norm_clip(row["funding_coverage_count"], 0, 4)]               # 1
    feats += one_hot(row.get("career_track_preference") or "", ALL_CAREERS) # 6
    feats += [float(row["requires_return_home_country"])]                    # 1
    return feats                                                              # total: 125


# ── SBERT text encoder ────────────────────────────────────────────────────────

_sbert_model = None


def get_sbert_model():
    """Lazy singleton — load SBERT sekali saja."""
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
    return _sbert_model


def encode_text(texts: list) -> np.ndarray:
    """Encode list of strings → float32 array shape (N, 384)."""
    model = get_sbert_model()
    return model.encode(texts, batch_size=64, show_progress_bar=False,
                        convert_to_numpy=True).astype(np.float32)

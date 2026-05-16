"""Two-Tower Recommendation System — Feedback Capture & Retraining

Handles collection of implicit student feedback (apply, click, view, reject)
and prepares data for model retraining or fine-tuning.

Feedback Types and Weights:
    - apply:   3.0  (Strong positive — student applied)
    - click:   2.0  (Soft positive — student clicked)
    - view:    1.0  (Weak positive — student saw in recommendations)
    - reject: -1.0  (Explicit negative — student swiped away)
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Feedback Weight Constants
# ============================================================

FEEDBACK_WEIGHTS = {
    "apply": 3.0,
    "click": 2.0,
    "view": 1.0,
    "reject": -1.0,
}

FEEDBACK_TYPES = list(FEEDBACK_WEIGHTS.keys())


def get_feedback_weight(feedback_type: str) -> float:
    """Get training weight for a feedback type.

    Args:
        feedback_type: One of 'apply', 'click', 'view', 'reject'.

    Returns:
        Float weight value.
    """
    return FEEDBACK_WEIGHTS.get(feedback_type.lower(), 1.0)


# ============================================================
# Feedback Loading
# ============================================================


def load_feedback(
    feedback_csv: str = "./datasets_two_tower/feedback.csv",
) -> pd.DataFrame:
    """Load feedback data from CSV.

    Args:
        feedback_csv: Path to feedback CSV file.

    Returns:
        DataFrame with columns: student_id, scholarship_id, feedback_type, weight, timestamp
    """
    df = pd.read_csv(feedback_csv)

    # Ensure weight column exists
    if "weight" not in df.columns:
        df["weight"] = df["feedback_type"].apply(get_feedback_weight)

    return df


def filter_feedback_since(
    feedback_df: pd.DataFrame,
    since_timestamp: str,
) -> pd.DataFrame:
    """Filter feedback records since a given timestamp.

    Args:
        feedback_df: DataFrame with a 'timestamp' column.
        since_timestamp: ISO datetime string (inclusive).

    Returns:
        Filtered DataFrame.
    """
    return feedback_df[feedback_df["timestamp"] >= since_timestamp].copy()


# ============================================================
# Pair Generation from Feedback
# ============================================================


def generate_pairs_from_feedback(
    feedback_df: pd.DataFrame,
    students_df: pd.DataFrame,
    scholarships_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    feedback_influence: float = 0.3,
) -> pd.DataFrame:
    """Generate new training pairs from feedback data.

    For each feedback record, creates a new pair with a relevance score
    derived from the feedback weight combined with the current NN prediction.

    Relevance Score Formula:
        new_score = (1 - feedback_influence) * current_nn_score
                   + feedback_influence * normalized_feedback_weight

    Where normalized_feedback_weight maps:
        apply  (3.0)  -> 0.90
        click  (2.0)  -> 0.80
        view   (1.0)  -> 0.70
        reject (-1.0) -> 0.10

    Args:
        feedback_df: DataFrame with feedback records.
        students_df: DataFrame of all students (indexed by student_id).
        scholarships_df: DataFrame of all scholarships (indexed by scholarship_id).
        pairs_df: Existing training pairs DataFrame (to look up current NN score
                  as approximation of current prediction).
        feedback_influence: How much the feedback signal influences the new score
                            (0.0 = ignore feedback, 1.0 = ignore current score).

    Returns:
        DataFrame of new pairs with columns: student_id, scholarship_id,
        relevance_score, timestamp, weight.
    """
    # Normalize feedback weights to [0, 1]
    def normalize_weight(w: float) -> float:
        if w >= 3.0:
            return 0.90  # apply
        elif w >= 2.0:
            return 0.80  # click
        elif w >= 1.0:
            return 0.70  # view
        else:
            return 0.10  # reject

    # Build a lookup for existing pairs (student_id, scholarship_id) -> relevance_score
    pair_lookup = {}
    for _, row in pairs_df.iterrows():
        key = (row["student_id"], row["scholarship_id"])
        pair_lookup[key] = row["relevance_score"]

    new_pairs = []
    for _, fb_row in feedback_df.iterrows():
        sid = fb_row["student_id"]
        schid = fb_row["scholarship_id"]
        fb_weight = float(fb_row.get("weight", get_feedback_weight(fb_row["feedback_type"])))
        fb_normalized = normalize_weight(fb_weight)
        timestamp = fb_row.get(
            "timestamp",
            datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

        # Get current NN approximate score if available
        current_score = pair_lookup.get((sid, schid), 0.5)

        # Compute new score
        new_score = (1.0 - feedback_influence) * current_score + feedback_influence * fb_normalized
        new_score = max(0.0, min(1.0, new_score))  # Clamp to [0, 1]

        new_pairs.append(
            {
                "student_id": sid,
                "scholarship_id": schid,
                "relevance_score": round(new_score, 4),
                "timestamp": timestamp,
                "weight": round(fb_weight, 1),
            }
        )

    if not new_pairs:
        return pd.DataFrame(
            columns=["student_id", "scholarship_id", "relevance_score", "timestamp", "weight"]
        )

    return pd.DataFrame(new_pairs)


# ============================================================
# Dataset Merging for Retraining
# ============================================================


def merge_datasets(
    original_pairs_df: pd.DataFrame,
    new_feedback_pairs_df: pd.DataFrame,
    max_original_samples: Optional[int] = None,
) -> pd.DataFrame:
    """Merge original training pairs with new feedback-derived pairs.

    To prevent the model from overfitting on new feedback (which may be sparse),
    a subset of original pairs is retained.

    Args:
        original_pairs_df: Original training pairs (from generator).
        new_feedback_pairs_df: New pairs generated from feedback.
        max_original_samples: Max number of original pairs to retain.
            If None, keep all original pairs. If the new feedback pairs are
            few, consider downsampling originals to balance.

    Returns:
        Merged DataFrame ready for retraining.
    """
    if new_feedback_pairs_df.empty:
        return original_pairs_df

    # Sample original pairs if needed
    if max_original_samples is not None and len(original_pairs_df) > max_original_samples:
        original_pairs_df = original_pairs_df.sample(
            n=max_original_samples, random_state=42
        )

    # Combine
    merged = pd.concat(
        [original_pairs_df, new_feedback_pairs_df],
        ignore_index=True,
    )

    # Sort by timestamp for consistent time-based splitting
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    return merged


# ============================================================
# Retraining Preparation
# ============================================================


def prepare_retraining_data(
    feedback_csv: str = "./datasets_two_tower/feedback.csv",
    students_csv: str = "./datasets_two_tower/students.csv",
    scholarships_csv: str = "./datasets_two_tower/scholarships.csv",
    pairs_csv: str = "./datasets_two_tower/pairs.csv",
    since_timestamp: Optional[str] = None,
    feedback_influence: float = 0.3,
    output_pairs_csv: str = "./datasets_two_tower/pairs_retrain.csv",
    max_original_pairs: Optional[int] = 250_000,
) -> str:
    """Prepare retraining data by merging new feedback with original pairs.

    This is the main entry point for the feedback loop. Call this function
    when new feedback is available for retraining.

    Args:
        feedback_csv: Path to feedback CSV.
        students_csv: Path to students CSV (for validation).
        scholarships_csv: Path to scholarships CSV.
        pairs_csv: Path to original training pairs CSV.
        since_timestamp: Only consider feedback after this timestamp.
            If None, use all feedback.
        feedback_influence: How much feedback influences new relevance scores.
        output_pairs_csv: Where to save the merged pairs for retraining.
        max_original_pairs: Max original pairs to keep (to balance with new data).

    Returns:
        Path to the saved retraining pairs CSV.
    """
    print("=" * 60)
    print("Feedback Loop — Retraining Data Preparation")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    feedback_df = load_feedback(feedback_csv)
    students_df = pd.read_csv(students_csv)
    scholarships_df = pd.read_csv(scholarships_csv)
    pairs_df = pd.read_csv(pairs_csv)

    print(f"  Feedback records:    {len(feedback_df):,}")
    print(f"  Original pairs:      {len(pairs_df):,}")
    print(f"  Students:            {len(students_df):,}")
    print(f"  Scholarships:        {len(scholarships_df):,}")

    # Filter feedback since timestamp
    if since_timestamp:
        feedback_df = filter_feedback_since(feedback_df, since_timestamp)
        print(f"  Feedback since {since_timestamp}: {len(feedback_df):,}")

    if feedback_df.empty:
        print("  No new feedback to process. Returning original pairs.")
        pairs_df.to_csv(output_pairs_csv, index=False)
        return output_pairs_csv

    # Generate new pairs from feedback
    print("\nGenerating new pairs from feedback...")
    new_pairs_df = generate_pairs_from_feedback(
        feedback_df, students_df, scholarships_df, pairs_df, feedback_influence
    )
    print(f"  Generated {len(new_pairs_df):,} new pairs")

    # Count feedback type distribution in new pairs
    if "weight" in new_pairs_df.columns:
        print("\n  New pair source distribution:")
        for fb_type in FEEDBACK_TYPES:
            count = len(feedback_df[feedback_df["feedback_type"] == fb_type])
            if count > 0:
                print(f"    {fb_type}: {count:,} → weight {get_feedback_weight(fb_type):.1f}")

    # Merge with original pairs
    print("\nMerging with original pairs...")
    merged_df = merge_datasets(
        pairs_df, new_pairs_df, max_original_samples=max_original_pairs
    )
    print(f"  Total pairs for retraining: {len(merged_df):,}")

    # Show relevance distribution
    merged_scores = merged_df["relevance_score"]
    match = (merged_scores >= 0.7).sum()
    inbetween = ((merged_scores >= 0.3) & (merged_scores < 0.7)).sum()
    not_match = (merged_scores < 0.3).sum()
    print(f"\n  Relevance distribution in merged set:")
    print(f"    Match (>=0.7):    {match:,}")
    print(f"    In-Between (0.3-0.7): {inbetween:,}")
    print(f"    Not Match (<0.3): {not_match:,}")

    # Save
    merged_df.to_csv(output_pairs_csv, index=False)
    print(f"\nSaved retraining pairs to: {output_pairs_csv}")

    return output_pairs_csv


# ============================================================
# CLI Entry Point
# ============================================================


def main():
    """CLI entry point for feedback processing.

    Usage:
        python feedback.py --since 2024-06-01T00:00:00Z --influence 0.3
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Feedback Loop — Prepare Retraining Data"
    )
    parser.add_argument(
        "--since", type=str, default=None,
        help="ISO timestamp: only process feedback after this time",
    )
    parser.add_argument(
        "--influence", type=float, default=0.3,
        help="Feedback influence on new relevance scores (0.0-1.0)",
    )
    parser.add_argument(
        "--feedback", type=str, default="./datasets_two_tower/feedback.csv",
        help="Path to feedback CSV",
    )
    parser.add_argument(
        "--students", type=str, default="./datasets_two_tower/students.csv",
        help="Path to students CSV",
    )
    parser.add_argument(
        "--scholarships", type=str, default="./datasets_two_tower/scholarships.csv",
        help="Path to scholarships CSV",
    )
    parser.add_argument(
        "--pairs", type=str, default="./datasets_two_tower/pairs.csv",
        help="Path to original pairs CSV",
    )
    parser.add_argument(
        "--output", type=str, default="./datasets_two_tower/pairs_retrain.csv",
        help="Output path for retraining pairs CSV",
    )
    parser.add_argument(
        "--max-original", type=int, default=250_000,
        help="Max original pairs to retain (None = keep all)",
    )

    args = parser.parse_args()

    prepare_retraining_data(
        feedback_csv=args.feedback,
        students_csv=args.students,
        scholarships_csv=args.scholarships,
        pairs_csv=args.pairs,
        since_timestamp=args.since,
        feedback_influence=args.influence,
        output_pairs_csv=args.output,
        max_original_pairs=args.max_original if args.max_original > 0 else None,
    )


if __name__ == "__main__":
    main()
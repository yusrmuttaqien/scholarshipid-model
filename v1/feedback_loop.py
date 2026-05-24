"""Feedback-driven pair adjustment (v1).

Reads pairs.csv + feedback.csv, adjusts relevance_score based on user signals,
and writes a new CSV that can be used for retraining.

Usage:
    python v1/feedback_loop.py \
      --pairs-file v1/datasets/pairs.csv \
      --feedback-file v1/datasets/feedback.csv \
      --output-file v1/datasets/pairs_feedback.csv \
      --alpha 0.5
"""

import argparse
import json
from pathlib import Path

import pandas as pd


# Feedback weights — calibrated from generator's scoring system
FEEDBACK_WEIGHTS = {
    "view": 1.0,
    "click": 2.0,
    "apply": 3.0,
    "reject": -1.0,
}

# Min/max of aggregated feedback sum (all signals for one pair)
FEEDBACK_MIN = -1.0   # single reject
FEEDBACK_MAX = 9.0    # view + click + apply


def compute_feedback_signal(group: pd.DataFrame) -> float:
    """Return normalized feedback signal in [0, 1] for a (student, scholarship) group."""
    total = group["weight"].sum()
    if total == 0.0:
        return 0.5  # neutral — no strong preference either way
    normalized = (total - FEEDBACK_MIN) / (FEEDBACK_MAX - FEEDBACK_MIN)
    return float(round(normalized, 4))


def adjust_pairs(pairs_df: pd.DataFrame, feedback_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Adjust relevance_score in pairs using feedback signals.

    adjusted_score = alpha * original_score + (1 - alpha) * feedback_signal
    Only pairs with feedback get adjusted; others keep their original score.
    """
    # Aggregate feedback per (student, scholarship) pair
    fb_signals = feedback_df.groupby(["student_id", "scholarship_id"])["weight"].sum().reset_index()
    fb_signals.columns = ["student_id", "scholarship_id", "total_weight"]

    def normalize(total_w: float) -> float:
        if total_w == 0.0:
            return 0.5
        normalized = (total_w - FEEDBACK_MIN) / (FEEDBACK_MAX - FEEDBACK_MIN)
        return float(round(normalized, 4))

    fb_signals["feedback_signal"] = fb_signals["total_weight"].apply(normalize)
    fb_signals = fb_signals[["student_id", "scholarship_id", "feedback_signal"]]

    # Merge back into pairs
    adjusted = pairs_df.merge(fb_signals, on=["student_id", "scholarship_id"], how="left")

    # Compute adjusted score (NaN feedback_signal → no adjustment)
    has_feedback = adjusted["feedback_signal"].notna() & (adjusted["feedback_signal"] != 0.5)
    adjusted.loc[has_feedback, "adjusted_score"] = adjusted.loc[has_feedback].apply(
        lambda r: alpha * r["relevance_score"] + (1 - alpha) * r["feedback_signal"],
        axis=1,
    )
    # Pairs without feedback keep original score
    no_feedback = ~has_feedback
    adjusted.loc[no_feedback, "adjusted_score"] = adjusted.loc[no_feedback, "relevance_score"]

    # Clip to [0, 1] for safety
    adjusted["adjusted_score"] = adjusted["adjusted_score"].clip(0.0, 1.0)

    return adjusted


def main():
    parser = argparse.ArgumentParser(description="Adjust pairs.csv using feedback signals")
    parser.add_argument("--pairs-file", required=True, help="Path to pairs.csv")
    parser.add_argument("--feedback-file", required=True, help="Path to feedback.csv")
    parser.add_argument("--output-file", required=True, help="Output CSV path (e.g., pairs_feedback.csv)")
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Blend ratio: alpha * original + (1 - alpha) * feedback (default 0.5)",
    )
    args = parser.parse_args()

    print(f"Loading pairs from {args.pairs_file}...")
    pairs_df = pd.read_csv(args.pairs_file)
    print(f"Loading feedback from {args.feedback_file}...")
    feedback_df = pd.read_csv(args.feedback_file)

    adjusted_df = adjust_pairs(pairs_df, feedback_df, args.alpha)

    # Write output — only keep columns train.py needs
    out_cols = ["student_id", "scholarship_id", "relevance_score", "timestamp", "adjusted_score"]
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adjusted_df[out_cols].to_csv(output_path, index=False)
    print(f"Adjusted pairs written to {output_path} ({len(adjusted_df):,} rows)")

    # Print summary stats
    has_feedback = adjusted_df["feedback_signal"].notna() & (adjusted_df["feedback_signal"] != 0.5)
    n_adjusted = has_feedback.sum()
    print(f"\n=== Summary ===")
    print(f"Total pairs: {len(adjusted_df):,}")
    print(f"Pairs with feedback: {n_adjusted:,} ({100 * n_adjusted / len(adjusted_df):.1f}%)")

    # Score distributions
    for col in ["relevance_score", "adjusted_score"]:
        vals = adjusted_df[col]
        print(f"\n{col}:")
        print(f"  mean: {vals.mean():.4f}, std: {vals.std():.4f}")
        print(f"  min: {vals.min():.4f}, max: {vals.max():.4f}")
        print(f"  quartiles: Q1={vals.quantile(0.25):.4f} Q3={vals.quantile(0.75):.4f}")

    # Feedback type distribution for adjusted pairs only
    fb_adjusted = feedback_df[feedback_df["student_id"].isin(adjusted_df.loc[has_feedback, "student_id"])]
    print(f"\nFeedback types (for adjusted pairs):")
    counts = fb_adjusted["feedback_type"].value_counts()
    for t, c in counts.items():
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()

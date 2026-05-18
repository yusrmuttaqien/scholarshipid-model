"""Field-to-track and field-to-career inference utilities.

Used by src/data/scholarships.py to derive HighSchoolTrack and CareerTrack
values from a Scholarship's eligible_fields.
"""

from typing import Optional

FIELD_TO_TRACK_MAP: dict[str, list[str]] = {
    "computer_science": ["science"],
    "engineering": ["science", "vocational"],
    "medicine": ["science"],
    "mathematics": ["science"],
    "physics": ["science"],
    "chemistry": ["science"],
    "biology": ["science"],
    "agriculture": ["science", "vocational"],
    "business": ["social_studies"],
    "economics": ["social_studies"],
    "law": ["social_studies"],
    "social_sciences": ["social_studies"],
    "education": ["social_studies", "languages"],
    "arts_humanities": ["languages", "social_studies"],
}

FIELD_TO_CAREER_MAP: dict[str, str] = {
    "computer_science": "industry",
    "engineering": "industry",
    "medicine": "public_service",
    "business": "industry",
    "economics": "government",
    "law": "government",
    "education": "academic",
    "arts_humanities": "academic",
    "social_sciences": "ngo_npo",
    "agriculture": "government",
    "mathematics": "academic",
    "physics": "academic",
    "chemistry": "academic",
    "biology": "academic",
}


def infer_tracks_from_fields(fields: list[str]) -> list[str]:
    """Infer eligible HighSchoolTrack values from a list of MajorField values."""
    tracks = set()
    for f in fields:
        for track in FIELD_TO_TRACK_MAP.get(f, ["science"]):
            tracks.add(track)
    return sorted(tracks)


def infer_career_from_fields(fields: list[str]) -> Optional[str]:
    """Infer primary CareerTrack from a list of MajorField values."""
    for f in fields:
        career = FIELD_TO_CAREER_MAP.get(f)
        if career:
            return career
    return None

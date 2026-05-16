from dataclasses import dataclass


@dataclass
class Pair:
    student_id: str
    scholarship_id: str
    relevance_score: float  # 0.0–1.0; >=0.7=Match, 0.3–0.7=In-Between, <0.3=Not Match
    timestamp: str          # ISO 8601: "YYYY-MM-DDTHH:MM:SSZ"

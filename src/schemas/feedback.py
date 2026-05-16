from dataclasses import dataclass

_FEEDBACK_WEIGHTS = {
    "apply": 3.0,
    "click": 2.0,
    "view": 1.0,
    "reject": -1.0,
}


@dataclass
class Feedback:
    student_id: str
    scholarship_id: str
    feedback_type: str  # "apply" | "click" | "view" | "reject"
    timestamp: str      # ISO 8601: "YYYY-MM-DDTHH:MM:SSZ"

    @property
    def weight(self) -> float:
        return _FEEDBACK_WEIGHTS.get(self.feedback_type, 1.0)

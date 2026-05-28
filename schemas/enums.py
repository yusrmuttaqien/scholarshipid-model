from enum import Enum
from typing import Optional


class DegreeLevel(str, Enum):
    high_school = "high_school"
    bachelors = "bachelors"


class Country(str, Enum):
    # Asia
    china = "china"
    india = "india"
    indonesia = "indonesia"
    japan = "japan"
    malaysia = "malaysia"
    philippines = "philippines"
    singapore = "singapore"
    south_korea = "south_korea"
    thailand = "thailand"
    vietnam = "vietnam"
    # Europe
    france = "france"
    germany = "germany"
    netherlands = "netherlands"
    sweden = "sweden"
    uk = "uk"
    switzerland = "switzerland"
    # Americas
    canada = "canada"
    usa = "usa"
    argentina = "argentina"
    brazil = "brazil"
    chile = "chile"
    # Africa
    egypt = "egypt"
    kenya = "kenya"
    morocco = "morocco"
    nigeria = "nigeria"
    south_africa = "south_africa"
    # Oceania
    australia = "australia"
    new_zealand = "new_zealand"


class HighSchoolTrack(str, Enum):
    science = "science"
    social_studies = "social_studies"
    languages = "languages"
    religion = "religion"
    vocational = "vocational"


class OlympiadLevel(str, Enum):
    none = "none"
    school = "school"
    city = "city"
    provincial = "provincial"
    national = "national"
    international = "international"


class OlympiadSubject(str, Enum):
    mathematics = "mathematics"
    physics = "physics"
    chemistry = "chemistry"
    biology = "biology"
    economics = "economics"
    geography = "geography"
    computer_science = "computer_science"
    linguistics = "linguistics"
    astronomy = "astronomy"
    informatics = "informatics"
    history = "history"
    english_language = "english_language"
    business_studies = "business_studies"


class IncomeCategory(str, Enum):
    very_low = "very_low"
    low = "low"
    middle = "middle"
    upper_middle = "upper_middle"
    high = "high"


class SchoolTier(str, Enum):
    excellence = "excellence"
    public_a = "public_a"
    private_a = "private_a"
    accredited_b = "accredited_b"
    accredited_c = "accredited_c"
    unaccredited = "unaccredited"
    unknown = "unknown"


class MajorField(str, Enum):
    computer_science = "computer_science"
    engineering = "engineering"
    medicine = "medicine"
    business = "business"
    economics = "economics"
    law = "law"
    education = "education"
    arts_humanities = "arts_humanities"
    social_sciences = "social_sciences"
    agriculture = "agriculture"
    mathematics = "mathematics"
    physics = "physics"
    chemistry = "chemistry"
    biology = "biology"


class CareerTrack(str, Enum):
    academic = "academic"
    industry = "industry"
    government = "government"
    ngo_npo = "ngo_npo"
    entrepreneurship = "entrepreneurship"
    public_service = "public_service"


class LanguageTest(str, Enum):
    toefl = "toefl"
    ielts = "ielts"
    topik = "topik"
    jlpt = "jlpt"
    delf = "delf"
    hsk = "hsk"


class HostRegion(str, Enum):
    asia = "asia"
    europe = "europe"
    north_america = "north_america"
    south_america = "south_america"
    africa = "africa"
    oceania = "oceania"


# ──────────────────────────────────────────────
# Enum mapping dicts
# ──────────────────────────────────────────────

# Map common country name variants → Country enum values
COUNTRY_MAP: dict[str, str] = {
    # Indonesia
    "indonesia": "indonesia",
    # Japan
    "japan": "japan", "jepang": "japan",
    # South Korea
    "south korea": "south_korea", "korea": "south_korea", "republic of korea": "south_korea",
    "korea selatan": "south_korea",
    # Singapore
    "singapore": "singapore", "singapura": "singapore",
    # Malaysia
    "malaysia": "malaysia",
    # China
    "china": "china", "people's republic of china": "china", "prc": "china", "tiongkok": "china",
    # India
    "india": "india",
    # Thailand
    "thailand": "thailand",
    # Philippines
    "philippines": "philippines", "filipina": "philippines",
    # Vietnam
    "vietnam": "vietnam",
    # USA
    "united states": "usa", "usa": "usa", "us": "usa", "united states of america": "usa",
    "amerika serikat": "usa",
    # Canada
    "canada": "canada",
    # UK
    "united kingdom": "uk", "uk": "uk", "great britain": "uk", "inggris": "uk",
    # Germany
    "germany": "germany", "deutschland": "germany", "jerman": "germany",
    # France
    "france": "france", "prancis": "france",
    # Netherlands
    "netherlands": "netherlands", "holland": "netherlands", "belanda": "netherlands",
    # Sweden
    "sweden": "sweden", "swedia": "sweden",
    # Switzerland
    "switzerland": "switzerland", "swiss": "switzerland",
    # Australia
    "australia": "australia",
    # New Zealand
    "new zealand": "new_zealand",
    # Hungary
    "hungary": "hungary",
    # Turkey
    "turkey": "turkey", "türkiye": "turkey",
    # Taiwan
    "taiwan": "taiwan",
    # Russia
    "russia": "russia", "russian federation": "russia",
    # Brazil
    "brazil": "brazil", "brasil": "brazil",
    # Argentina
    "argentina": "argentina",
    # Chile
    "chile": "chile",
    # Egypt
    "egypt": "egypt", "mesir": "egypt",
    # Kenya
    "kenya": "kenya",
    # Morocco
    "morocco": "morocco", "maroko": "morocco",
    # Nigeria
    "nigeria": "nigeria",
    # South Africa
    "south africa": "south_africa", "afrika selatan": "south_africa",
}

# Map major field keywords → MajorField enum values
FIELD_MAP: dict[str, str] = {
    "computer science": "computer_science",
    "computing": "computer_science",
    "informatics": "computer_science",
    "information technology": "computer_science",
    "it": "computer_science",
    "engineering": "engineering",
    "teknik": "engineering",
    "medicine": "medicine",
    "medical": "medicine",
    "health sciences": "medicine",
    "kedokteran": "medicine",
    "business": "business",
    "management": "business",
    "administration": "business",
    "economics": "economics",
    "ekonomi": "economics",
    "law": "law",
    "hukum": "law",
    "legal": "law",
    "education": "education",
    "pendidikan": "education",
    "arts": "arts_humanities",
    "humanities": "arts_humanities",
    "seni": "arts_humanities",
    "social sciences": "social_sciences",
    "social": "social_sciences",
    "ilmu sosial": "social_sciences",
    "agriculture": "agriculture",
    "pertanian": "agriculture",
    "mathematics": "mathematics",
    "matematika": "mathematics",
    "math": "mathematics",
    "physics": "physics",
    "fisika": "physics",
    "chemistry": "chemistry",
    "kimia": "chemistry",
    "biology": "biology",
    "biologi": "biology",
    "life sciences": "biology",
}

# Map host country → HostRegion enum value
REGION_MAP: dict[str, str] = {
    "china": "asia",
    "india": "asia",
    "indonesia": "asia",
    "japan": "asia",
    "malaysia": "asia",
    "philippines": "asia",
    "singapore": "asia",
    "south_korea": "asia",
    "thailand": "asia",
    "vietnam": "asia",
    "france": "europe",
    "germany": "europe",
    "netherlands": "europe",
    "sweden": "europe",
    "uk": "europe",
    "switzerland": "europe",
    "canada": "north_america",
    "usa": "north_america",
    "argentina": "south_america",
    "brazil": "south_america",
    "chile": "south_america",
    "egypt": "africa",
    "kenya": "africa",
    "morocco": "africa",
    "nigeria": "africa",
    "south_africa": "africa",
    "australia": "oceania",
    "new_zealand": "oceania",
}


def normalize_country(raw: str) -> Optional[str]:
    """Normalize a raw country string to a Country enum value, or None if unknown."""
    key = raw.strip().lower()
    mapped = COUNTRY_MAP.get(key)
    if mapped is None:
        return None
    valid_values = {c.value for c in Country}
    return mapped if mapped in valid_values else None


def normalize_field(raw: str) -> Optional[str]:
    """Normalize a raw field string to a MajorField enum value, or None if unknown."""
    key = raw.strip().lower()
    for keyword, value in FIELD_MAP.items():
        if keyword in key:
            return value
    return None



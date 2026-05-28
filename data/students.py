"""Static seed data for synthetic student generation."""

COUNTRIES_BY_REGION = {
    "asia": [
        "indonesia", "malaysia", "thailand", "philippines", "vietnam",
        "singapore", "japan", "south_korea", "china", "india",
    ],
    "europe": ["france", "germany", "netherlands", "sweden", "uk", "switzerland"],
    "north_america": ["canada", "usa"],
    "south_america": ["argentina", "brazil", "chile"],
    "africa": ["egypt", "kenya", "morocco", "nigeria", "south_africa"],
    "oceania": ["australia", "new_zealand"],
}

ALL_COUNTRIES = [c for countries in COUNTRIES_BY_REGION.values() for c in countries]

SCHOOL_NAMES = {
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

PERSONAL_STATEMENT_TEMPLATES = [
    "I am passionate about {field} and aspire to make a significant contribution to {interest}.",
    "My journey in {field} began during high school where I discovered my love for {interest}.",
    "As a dedicated student of {track}, I have been actively involved in {activity}.",
    "Growing up in {country}, I witnessed firsthand the challenges in {interest} and want to address them.",
]

FUTURE_GOALS_TEMPLATES = [
    "After completing my studies, I plan to return home and contribute to {interest} in my community.",
    "My long-term goal is to become a leader in {field} and drive innovation in {region}.",
    "I aspire to establish a program focused on {interest} that benefits underrepresented communities.",
    "I want to bridge the gap between {field} and {interest} through research and practice.",
]

RESEARCH_INTERESTS = [
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

ACHIEVEMENT_TEMPLATES = [
    "Achieved {level} level in {subject} olympiad.",
    "Won {competition} competition at the {level} level.",
    "Led a team of {team_size} in a {subject} project.",
    "Volunteered {hours} hours for community service.",
    "Served as {position} in student council.",
]

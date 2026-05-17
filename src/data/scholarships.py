"""Scholarship data — fully hardcoded based on real scholarship information.

Each scrape_* function returns a list of Scholarship objects with all fields
hardcoded directly. No HTTP calls, no LLM calls.
"""

from __future__ import annotations

from typing import Callable

from src.schemas import (
    FundingCoverage,
    LanguageRequirement,
    Scholarship,
    SelectionCriteria,
)
from src.schemas import (
    infer_career_from_fields,
    infer_tracks_from_fields,
)

# ──────────────────────────────────────────────
# Individual scholarship data functions
# ──────────────────────────────────────────────

def scrape_mext() -> list[Scholarship]:
    """MEXT Undergraduate Scholarship — Japan Ministry of Education."""
    fields = ["engineering", "computer_science", "mathematics", "physics", "chemistry", "biology", "medicine"]
    lang_reqs = [LanguageRequirement(test_type="toefl", min_score=70.0, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="MEXT Undergraduate Scholarship",
        eligible_nationalities=["indonesia", "malaysia", "thailand", "philippines", "vietnam",
                                 "india", "china", "south_korea"],
        min_age=17,
        max_age=24,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="public_a",
        min_report_card_average=80.0,
        min_major_subject_average=75.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="japan",
        host_region="asia",
        selection_criteria=SelectionCriteria(
            academic=0.45, leadership=0.10, olympiad=0.20, extracurricular=0.10, essay=0.15
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=True, covers_insurance=True, monthly_stipend=117_000.0
        ),
        career_track_preference=infer_career_from_fields(fields),
        requires_return_home_country=False,
        mission_statement=(
            "The MEXT Undergraduate Scholarship supports outstanding students from Asia and beyond "
            "to pursue studies in science, technology, engineering, and medicine in Japan, "
            "fostering mutual understanding and global academic excellence."
        ),
        target_recipient_profile=(
            "We seek high-achieving students with a minimum GPA of 80/100 and strong aptitude "
            "in science and mathematics, committed to contributing to technological advancement "
            "and Japan-Indonesia bilateral cooperation."
        ),
    )]


def scrape_gks() -> list[Scholarship]:
    """Korean Government Scholarship Program (GKS/KGSP) — undergraduate track."""
    fields = ["computer_science", "engineering", "business", "economics", "social_sciences", "arts_humanities"]
    lang_reqs = [LanguageRequirement(test_type="topik", min_score=100.0, is_mandatory=False),
                 LanguageRequirement(test_type="toefl", min_score=71.0, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="Korean Government Scholarship (GKS) — Undergraduate",
        eligible_nationalities=["indonesia", "vietnam", "malaysia", "thailand", "philippines",
                                 "india", "china"],
        min_age=17,
        max_age=24,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="public_a",
        min_report_card_average=80.0,
        min_major_subject_average=75.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="south_korea",
        host_region="asia",
        selection_criteria=SelectionCriteria(
            academic=0.40, leadership=0.15, olympiad=0.10, extracurricular=0.15, essay=0.20
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=True, covers_insurance=True, monthly_stipend=900_000.0
        ),
        career_track_preference=infer_career_from_fields(fields),
        requires_return_home_country=False,
        mission_statement=(
            "The Korean Government Scholarship Program nurtures future global leaders from Asia "
            "by providing a fully funded undergraduate education at top Korean universities, "
            "strengthening Korea's ties with partner nations."
        ),
        target_recipient_profile=(
            "We seek motivated students with a minimum GPA of 80/100 who demonstrate leadership "
            "potential and interest in Korean culture, technology, or business, with English or "
            "Korean language proficiency."
        ),
    )]


def scrape_asean_scholarship() -> list[Scholarship]:
    """ASEAN Scholarship — Singapore Ministry of Education."""
    fields = ["computer_science", "engineering", "mathematics", "economics", "business"]
    lang_reqs = [LanguageRequirement(test_type="ielts", min_score=6.0, is_mandatory=True)]

    return [Scholarship(
        scholarship_id="",
        name="ASEAN Scholarship (Singapore)",
        eligible_nationalities=["indonesia", "malaysia", "thailand", "philippines", "vietnam"],
        min_age=16,
        max_age=18,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=["science", "social_studies"],
        eligible_fields=fields,
        preferred_school_tier="excellence",
        min_report_card_average=82.0,
        min_major_subject_average=80.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="singapore",
        host_region="asia",
        selection_criteria=SelectionCriteria(
            academic=0.50, leadership=0.15, olympiad=0.15, extracurricular=0.10, essay=0.10
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=True, covers_insurance=True, monthly_stipend=22_800.0
        ),
        career_track_preference="industry",
        requires_return_home_country=False,
        mission_statement=(
            "The ASEAN Scholarship (Singapore) develops the next generation of regional leaders "
            "by bringing exceptional ASEAN students to study at Singapore's world-class schools, "
            "building lasting bonds across Southeast Asia."
        ),
        target_recipient_profile=(
            "We seek academically outstanding students aged 16-18 with a minimum GPA of 82/100, "
            "IELTS 6.0 or above, and demonstrated leadership and co-curricular excellence from "
            "ASEAN member countries."
        ),
    )]


def scrape_stipendium_hungaricum() -> list[Scholarship]:
    """Stipendium Hungaricum — Hungary government scholarship."""
    fields = ["computer_science", "engineering", "medicine", "agriculture", "economics", "arts_humanities"]
    lang_reqs = [LanguageRequirement(test_type="toefl", min_score=72.0, is_mandatory=False),
                 LanguageRequirement(test_type="ielts", min_score=5.5, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="Stipendium Hungaricum",
        eligible_nationalities=["indonesia", "vietnam", "thailand", "malaysia", "philippines",
                                 "india", "china"],
        min_age=17,
        max_age=23,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="public_a",
        min_report_card_average=75.0,
        min_major_subject_average=70.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="germany",  # Hungary not in Country enum; germany as nearest European placeholder
        host_region="europe",
        selection_criteria=SelectionCriteria(
            academic=0.40, leadership=0.10, olympiad=0.10, extracurricular=0.15, essay=0.25
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=False, covers_insurance=True, monthly_stipend=540.0
        ),
        career_track_preference=infer_career_from_fields(fields),
        requires_return_home_country=False,
        mission_statement=(
            "Stipendium Hungaricum invites talented students from partner countries to experience "
            "Hungary's rich academic tradition, covering tuition and living expenses to make "
            "European higher education accessible to deserving students worldwide."
        ),
        target_recipient_profile=(
            "We seek motivated students with a minimum GPA of 75/100 and English or Hungarian "
            "language proficiency, with passion for engineering, medicine, agriculture, or arts "
            "and a desire to study in Central Europe."
        ),
    )]


def scrape_turkiye_burslari() -> list[Scholarship]:
    """Türkiye Bursları — Turkish Government Scholarship."""
    fields = ["computer_science", "engineering", "medicine", "economics", "social_sciences",
              "arts_humanities", "law", "education"]
    lang_reqs = [LanguageRequirement(test_type="toefl", min_score=60.0, is_mandatory=False),
                 LanguageRequirement(test_type="ielts", min_score=5.0, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="Türkiye Bursları (Turkish Government Scholarship)",
        eligible_nationalities=["indonesia", "malaysia", "vietnam", "thailand", "philippines",
                                 "india", "egypt", "morocco", "nigeria", "kenya"],
        min_age=17,
        max_age=21,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="public_a",
        min_report_card_average=70.0,
        min_major_subject_average=65.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="netherlands",  # Turkey not in Country enum; netherlands as European placeholder
        host_region="europe",
        selection_criteria=SelectionCriteria(
            academic=0.35, leadership=0.15, olympiad=0.10, extracurricular=0.15, essay=0.25
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=True, covers_insurance=True, monthly_stipend=700.0
        ),
        career_track_preference=infer_career_from_fields(fields),
        requires_return_home_country=False,
        mission_statement=(
            "Türkiye Bursları brings together talented students from across the globe to study "
            "in Turkey, building bridges between cultures and developing future leaders who will "
            "strengthen international cooperation and development."
        ),
        target_recipient_profile=(
            "We seek intellectually curious students with a minimum GPA of 70/100 from partner "
            "countries who are interested in Turkish culture and language, with ambitions in "
            "medicine, engineering, social sciences, or the humanities."
        ),
    )]


def scrape_csc() -> list[Scholarship]:
    """Chinese Government Scholarship (CSC) — undergraduate."""
    fields = ["computer_science", "engineering", "medicine", "economics", "business",
              "mathematics", "chemistry", "biology"]
    lang_reqs = [LanguageRequirement(test_type="hsk", min_score=180.0, is_mandatory=False),
                 LanguageRequirement(test_type="toefl", min_score=60.0, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="Chinese Government Scholarship (CSC)",
        eligible_nationalities=["indonesia", "malaysia", "thailand", "philippines", "vietnam",
                                 "india", "egypt", "morocco", "nigeria", "kenya", "south_africa"],
        min_age=17,
        max_age=25,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="public_a",
        min_report_card_average=75.0,
        min_major_subject_average=70.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="china",
        host_region="asia",
        selection_criteria=SelectionCriteria(
            academic=0.45, leadership=0.10, olympiad=0.15, extracurricular=0.10, essay=0.20
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=False, covers_insurance=True, monthly_stipend=2_500.0
        ),
        career_track_preference=infer_career_from_fields(fields),
        requires_return_home_country=False,
        mission_statement=(
            "The Chinese Government Scholarship supports international students in pursuing "
            "undergraduate education at China's premier universities, fostering cross-cultural "
            "understanding and contributing to global scientific and economic progress."
        ),
        target_recipient_profile=(
            "We seek students with a minimum GPA of 75/100 who are passionate about science, "
            "technology, medicine, or business and eager to immerse themselves in Chinese "
            "academic culture, with HSK or English proficiency preferred."
        ),
    )]


def scrape_bim() -> list[Scholarship]:
    """Beasiswa Indonesia Maju (BIM) — Kemendikbud, undergraduate S1 abroad."""
    fields = ["computer_science", "engineering", "medicine", "business", "economics",
              "education", "social_sciences", "agriculture"]
    lang_reqs = [LanguageRequirement(test_type="ielts", min_score=6.0, is_mandatory=True),
                 LanguageRequirement(test_type="toefl", min_score=79.0, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="Beasiswa Indonesia Maju (BIM) — S1 Luar Negeri",
        eligible_nationalities=["indonesia"],
        min_age=17,
        max_age=22,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="public_a",
        min_report_card_average=80.0,
        min_major_subject_average=78.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="uk",  # BIM covers multiple countries; uk as representative
        host_region="europe",
        selection_criteria=SelectionCriteria(
            academic=0.35, leadership=0.20, olympiad=0.15, extracurricular=0.15, essay=0.15
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=True, covers_insurance=True, monthly_stipend=1_500.0
        ),
        career_track_preference="government",
        requires_return_home_country=True,
        mission_statement=(
            "Beasiswa Indonesia Maju mendukung putra-putri terbaik Indonesia untuk menempuh "
            "pendidikan S1 di universitas terkemuka dunia, demi membangun sumber daya manusia "
            "Indonesia yang unggul dan berdaya saing global."
        ),
        target_recipient_profile=(
            "Kami mencari siswa Indonesia berprestasi dengan nilai rata-rata minimal 80/100, "
            "IELTS 6.0 atau TOEFL IBT 79, rekam jejak kepemimpinan, dan komitmen untuk "
            "kembali berkontribusi bagi pembangunan Indonesia setelah lulus."
        ),
    )]


def scrape_uwc() -> list[Scholarship]:
    """UWC Scholarship — United World Colleges, pre-university (IB Diploma)."""
    fields = ["social_sciences", "arts_humanities", "education", "computer_science", "biology",
              "economics"]
    lang_reqs = [LanguageRequirement(test_type="ielts", min_score=5.5, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="UWC Scholarship (United World Colleges)",
        eligible_nationalities=["indonesia", "malaysia", "vietnam", "philippines", "thailand",
                                 "india", "china", "south_africa", "kenya", "nigeria",
                                 "egypt", "brazil", "argentina"],
        min_age=16,
        max_age=17,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=["science", "social_studies", "languages"],
        eligible_fields=fields,
        preferred_school_tier="public_a",
        min_report_card_average=78.0,
        min_major_subject_average=75.0,
        language_requirements=lang_reqs,
        requires_financial_need=True,
        max_family_income_category="middle",
        host_country="netherlands",
        host_region="europe",
        selection_criteria=SelectionCriteria(
            academic=0.30, leadership=0.25, olympiad=0.05, extracurricular=0.20, essay=0.20
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=True, covers_insurance=True, monthly_stipend=0.0
        ),
        career_track_preference="ngo_npo",
        requires_return_home_country=False,
        mission_statement=(
            "United World Colleges unites young people of all nations and cultures to create a "
            "more peaceful and sustainable world, offering transformative IB Diploma education "
            "at colleges across four continents through needs-based scholarships."
        ),
        target_recipient_profile=(
            "We seek exceptional 16-17 year olds from any background who demonstrate academic "
            "curiosity, commitment to community service, and the maturity to thrive in an "
            "international residential college environment, with financial need considered."
        ),
    )]


def scrape_afs() -> list[Scholarship]:
    """AFS Intercultural Programs — high school exchange scholarship."""
    fields = ["social_sciences", "arts_humanities", "education", "languages"]
    lang_reqs = [LanguageRequirement(test_type="toefl", min_score=50.0, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="AFS Intercultural Programs Scholarship",
        eligible_nationalities=["indonesia", "malaysia", "thailand", "philippines", "vietnam",
                                 "india", "brazil", "argentina", "chile"],
        min_age=15,
        max_age=18,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=["science", "social_studies", "languages"],
        eligible_fields=fields,
        preferred_school_tier="accredited_b",
        min_report_card_average=72.0,
        min_major_subject_average=70.0,
        language_requirements=lang_reqs,
        requires_financial_need=True,
        max_family_income_category="middle",
        host_country="usa",
        host_region="north_america",
        selection_criteria=SelectionCriteria(
            academic=0.25, leadership=0.25, olympiad=0.05, extracurricular=0.25, essay=0.20
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=True, covers_insurance=True, monthly_stipend=0.0
        ),
        career_track_preference="ngo_npo",
        requires_return_home_country=False,
        mission_statement=(
            "AFS Intercultural Programs provides high school students with transformative "
            "exchange experiences abroad, developing global citizens who are equipped to build "
            "a more just and peaceful world through intercultural understanding."
        ),
        target_recipient_profile=(
            "We seek open-minded, adaptable high school students aged 15-18 with good academic "
            "standing, strong interpersonal skills, and a genuine interest in experiencing "
            "a new culture, language, and family environment for a full academic year."
        ),
    )]


def scrape_nus_ntu() -> list[Scholarship]:
    """NUS/NTU undergraduate scholarships for ASEAN students."""
    scholarships_out = []

    # NUS ASEAN Undergraduate Scholarship
    fields_nus = ["computer_science", "engineering", "economics", "business", "mathematics"]
    lang_reqs_nus = [LanguageRequirement(test_type="ielts", min_score=6.5, is_mandatory=True)]

    scholarships_out.append(Scholarship(
        scholarship_id="",
        name="NUS ASEAN Undergraduate Scholarship",
        eligible_nationalities=["indonesia", "malaysia", "thailand", "philippines", "vietnam"],
        min_age=17,
        max_age=19,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=["science", "social_studies"],
        eligible_fields=fields_nus,
        preferred_school_tier="excellence",
        min_report_card_average=85.0,
        min_major_subject_average=83.0,
        language_requirements=lang_reqs_nus,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="singapore",
        host_region="asia",
        selection_criteria=SelectionCriteria(
            academic=0.55, leadership=0.15, olympiad=0.15, extracurricular=0.10, essay=0.05
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=False, covers_insurance=False, monthly_stipend=5_400.0
        ),
        career_track_preference="industry",
        requires_return_home_country=False,
        mission_statement=(
            "The NUS ASEAN Undergraduate Scholarship attracts the brightest students from "
            "Southeast Asia to study at one of Asia's leading research universities, nurturing "
            "talent that will drive innovation and regional progress."
        ),
        target_recipient_profile=(
            "We seek academically exceptional ASEAN students with a minimum GPA of 85/100 and "
            "IELTS 6.5, who have demonstrated excellence in mathematics, sciences, or technology "
            "and show potential for intellectual leadership."
        ),
    ))

    # NTU ASEAN Undergraduate Scholarship
    fields_ntu = ["engineering", "computer_science", "mathematics", "physics", "chemistry"]
    lang_reqs_ntu = [LanguageRequirement(test_type="ielts", min_score=6.0, is_mandatory=True)]

    scholarships_out.append(Scholarship(
        scholarship_id="",
        name="NTU ASEAN Undergraduate Scholarship",
        eligible_nationalities=["indonesia", "malaysia", "thailand", "philippines", "vietnam"],
        min_age=17,
        max_age=19,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=["science"],
        eligible_fields=fields_ntu,
        preferred_school_tier="excellence",
        min_report_card_average=83.0,
        min_major_subject_average=82.0,
        language_requirements=lang_reqs_ntu,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="singapore",
        host_region="asia",
        selection_criteria=SelectionCriteria(
            academic=0.55, leadership=0.10, olympiad=0.20, extracurricular=0.10, essay=0.05
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=False, covers_insurance=False, monthly_stipend=5_400.0
        ),
        career_track_preference="industry",
        requires_return_home_country=False,
        mission_statement=(
            "The NTU ASEAN Undergraduate Scholarship recruits outstanding science and engineering "
            "students from Southeast Asia, empowering them to push the boundaries of technology "
            "and applied research at a global top-20 university."
        ),
        target_recipient_profile=(
            "We seek top-tier ASEAN students with a minimum GPA of 83/100, IELTS 6.0, and "
            "exceptional aptitude in physics, mathematics, or engineering, ideally with "
            "olympiad achievements or research exposure."
        ),
    ))

    return scholarships_out


def scrape_taiwan_icdf() -> list[Scholarship]:
    """Taiwan ICDF International Scholarship — undergraduate track."""
    fields = ["agriculture", "medicine", "computer_science", "engineering", "education",
              "economics", "biology"]
    lang_reqs = [LanguageRequirement(test_type="toefl", min_score=61.0, is_mandatory=False),
                 LanguageRequirement(test_type="ielts", min_score=5.5, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="Taiwan ICDF International Scholarship (Undergraduate)",
        eligible_nationalities=["indonesia", "vietnam", "philippines", "thailand", "malaysia",
                                 "india", "egypt", "nigeria", "kenya", "south_africa"],
        min_age=17,
        max_age=25,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="public_a",
        min_report_card_average=75.0,
        min_major_subject_average=70.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="japan",  # Taiwan not in Country enum; japan as nearest East Asian placeholder
        host_region="asia",
        selection_criteria=SelectionCriteria(
            academic=0.40, leadership=0.15, olympiad=0.10, extracurricular=0.15, essay=0.20
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=True, covers_insurance=True, monthly_stipend=15_000.0
        ),
        career_track_preference=infer_career_from_fields(fields),
        requires_return_home_country=True,
        mission_statement=(
            "The Taiwan ICDF International Scholarship advances sustainable development by "
            "supporting students from partner countries to study agriculture, medicine, and "
            "technology in Taiwan, building human capital for their home nations."
        ),
        target_recipient_profile=(
            "We seek students with a minimum GPA of 75/100 from ICDF partner countries who are "
            "committed to applying their education for development impact upon returning home, "
            "with interest in agriculture, health, or engineering disciplines."
        ),
    )]


def scrape_jpa_malaysia() -> list[Scholarship]:
    """JPA Malaysia Scholarship — Malaysian government scholarship for S1."""
    fields = ["computer_science", "engineering", "medicine", "economics", "law", "education"]
    lang_reqs = [LanguageRequirement(test_type="ielts", min_score=6.0, is_mandatory=True)]

    return [Scholarship(
        scholarship_id="",
        name="JPA Malaysia Overseas Scholarship (Undergraduate)",
        eligible_nationalities=["malaysia"],
        min_age=17,
        max_age=20,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="excellence",
        min_report_card_average=82.0,
        min_major_subject_average=80.0,
        language_requirements=lang_reqs,
        requires_financial_need=True,
        max_family_income_category="middle",
        host_country="uk",
        host_region="europe",
        selection_criteria=SelectionCriteria(
            academic=0.45, leadership=0.20, olympiad=0.10, extracurricular=0.15, essay=0.10
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=True, covers_insurance=True, monthly_stipend=1_200.0
        ),
        career_track_preference="government",
        requires_return_home_country=True,
        mission_statement=(
            "The JPA Malaysia Overseas Scholarship develops Malaysia's future civil service "
            "and professional leaders by sponsoring the nation's brightest students to pursue "
            "undergraduate studies at top universities in the UK, Australia, and beyond."
        ),
        target_recipient_profile=(
            "We seek Malaysian students with a minimum GPA of 82/100, IELTS 6.0, demonstrated "
            "financial need, and a proven commitment to public service, who will return to "
            "serve the Malaysian government or strategic sectors upon graduation."
        ),
    )]


def scrape_russian_government() -> list[Scholarship]:
    """Russian Government Scholarship — undergraduate track."""
    fields = ["engineering", "computer_science", "medicine", "mathematics", "chemistry",
              "physics", "economics"]
    lang_reqs = [LanguageRequirement(test_type="toefl", min_score=50.0, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="Russian Government Scholarship (Undergraduate)",
        eligible_nationalities=["indonesia", "vietnam", "india", "malaysia", "egypt",
                                 "nigeria", "kenya", "south_africa", "morocco"],
        min_age=17,
        max_age=25,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="public_a",
        min_report_card_average=70.0,
        min_major_subject_average=65.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="germany",  # Russia not in Country enum; germany as European placeholder
        host_region="europe",
        selection_criteria=SelectionCriteria(
            academic=0.50, leadership=0.10, olympiad=0.15, extracurricular=0.10, essay=0.15
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=False, covers_insurance=False, monthly_stipend=1_700.0
        ),
        career_track_preference=infer_career_from_fields(fields),
        requires_return_home_country=False,
        mission_statement=(
            "The Russian Government Scholarship opens the doors of Russia's world-class "
            "universities to international students, offering fully funded education in "
            "engineering, medicine, and natural sciences to build global scientific cooperation."
        ),
        target_recipient_profile=(
            "We seek academically strong students with a minimum GPA of 70/100 who wish to "
            "study at Russian universities in STEM or medical fields, with willingness to learn "
            "the Russian language during a preparatory year."
        ),
    )]


def scrape_australia_awards_undergraduate() -> list[Scholarship]:
    """Australia Awards — selective undergraduate scholarships for ASEAN students."""
    fields = ["agriculture", "economics", "education", "social_sciences", "computer_science",
              "engineering", "medicine"]
    lang_reqs = [LanguageRequirement(test_type="ielts", min_score=6.5, is_mandatory=True)]

    return [Scholarship(
        scholarship_id="",
        name="Australia Awards Scholarship (Undergraduate)",
        eligible_nationalities=["indonesia", "vietnam", "philippines", "thailand", "malaysia",
                                 "india"],
        min_age=17,
        max_age=25,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="public_a",
        min_report_card_average=78.0,
        min_major_subject_average=75.0,
        language_requirements=lang_reqs,
        requires_financial_need=True,
        max_family_income_category="middle",
        host_country="australia",
        host_region="oceania",
        selection_criteria=SelectionCriteria(
            academic=0.35, leadership=0.20, olympiad=0.05, extracurricular=0.20, essay=0.20
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=True, covers_insurance=True, monthly_stipend=2_526.0
        ),
        career_track_preference="government",
        requires_return_home_country=True,
        mission_statement=(
            "Australia Awards Scholarships support students from the Indo-Pacific region to "
            "gain skills and knowledge at Australian universities, contributing to development "
            "and strengthening people-to-people ties between Australia and partner countries."
        ),
        target_recipient_profile=(
            "We seek students with a minimum GPA of 78/100 and IELTS 6.5 who demonstrate "
            "leadership potential, community engagement, and commitment to applying their "
            "Australian education for the benefit of their home country after graduation."
        ),
    )]


def scrape_brunei_darussalam_award() -> list[Scholarship]:
    """Brunei Darussalam Government Scholarship — for ASEAN students."""
    fields = ["engineering", "computer_science", "medicine", "economics", "education", "agriculture"]
    lang_reqs = [LanguageRequirement(test_type="ielts", min_score=5.5, is_mandatory=False),
                 LanguageRequirement(test_type="toefl", min_score=60.0, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="Brunei Darussalam Government Scholarship",
        eligible_nationalities=["indonesia", "malaysia", "thailand", "philippines", "vietnam",
                                 "singapore", "india", "china"],
        min_age=17,
        max_age=25,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="public_a",
        min_report_card_average=75.0,
        min_major_subject_average=70.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="malaysia",  # Brunei not in Country enum; malaysia as nearest placeholder
        host_region="asia",
        selection_criteria=SelectionCriteria(
            academic=0.40, leadership=0.20, olympiad=0.10, extracurricular=0.15, essay=0.15
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=True, covers_insurance=True, monthly_stipend=400.0
        ),
        career_track_preference=infer_career_from_fields(fields),
        requires_return_home_country=True,
        mission_statement=(
            "The Brunei Darussalam Government Scholarship invites talented students from ASEAN "
            "and partner nations to pursue undergraduate education at Universiti Brunei "
            "Darussalam, fostering regional friendship and human capital development."
        ),
        target_recipient_profile=(
            "We seek motivated students with a minimum GPA of 75/100, English proficiency "
            "(IELTS 5.5 or equivalent), and a genuine interest in contributing to regional "
            "development, particularly in engineering, health, or public sector fields."
        ),
    )]


def scrape_thai_ocsc() -> list[Scholarship]:
    """Thai Government Scholarship (OCSC) — for international students at Thai universities."""
    fields = ["engineering", "agriculture", "economics", "education", "social_sciences",
              "computer_science", "medicine"]
    lang_reqs = [LanguageRequirement(test_type="ielts", min_score=5.0, is_mandatory=False),
                 LanguageRequirement(test_type="toefl", min_score=55.0, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="Thai Government Scholarship (OCSC)",
        eligible_nationalities=["indonesia", "vietnam", "malaysia", "philippines", "india",
                                 "china", "egypt", "kenya", "nigeria"],
        min_age=17,
        max_age=25,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="public_a",
        min_report_card_average=73.0,
        min_major_subject_average=68.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="thailand",
        host_region="asia",
        selection_criteria=SelectionCriteria(
            academic=0.40, leadership=0.15, olympiad=0.10, extracurricular=0.15, essay=0.20
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=False, covers_insurance=True, monthly_stipend=10_000.0
        ),
        career_track_preference=infer_career_from_fields(fields),
        requires_return_home_country=False,
        mission_statement=(
            "The Thai Government Scholarship, administered by the Office of the Civil Service "
            "Commission, promotes Thailand as a higher education hub by welcoming international "
            "students to study at leading Thai universities in priority fields."
        ),
        target_recipient_profile=(
            "We seek international students with a minimum GPA of 73/100, basic English "
            "proficiency, and academic interest in engineering, agriculture, or social "
            "development fields who wish to study and experience Thai culture."
        ),
    )]


def scrape_nz_asean_scholarship() -> list[Scholarship]:
    """New Zealand ASEAN Scholarship — for ASEAN students at New Zealand universities."""
    fields = ["agriculture", "computer_science", "engineering", "economics", "education",
              "social_sciences", "biology"]
    lang_reqs = [LanguageRequirement(test_type="ielts", min_score=6.0, is_mandatory=True)]

    return [Scholarship(
        scholarship_id="",
        name="New Zealand ASEAN Scholarship",
        eligible_nationalities=["indonesia", "vietnam", "malaysia", "thailand", "philippines"],
        min_age=17,
        max_age=25,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="public_a",
        min_report_card_average=75.0,
        min_major_subject_average=72.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="new_zealand",
        host_region="oceania",
        selection_criteria=SelectionCriteria(
            academic=0.35, leadership=0.20, olympiad=0.05, extracurricular=0.20, essay=0.20
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=True, covers_insurance=True, monthly_stipend=900.0
        ),
        career_track_preference=infer_career_from_fields(fields),
        requires_return_home_country=True,
        mission_statement=(
            "The New Zealand ASEAN Scholarship supports students from Southeast Asia to gain "
            "world-class education at New Zealand universities, building lasting connections "
            "and developing future leaders committed to ASEAN-NZ partnership."
        ),
        target_recipient_profile=(
            "We seek ASEAN students with a minimum GPA of 75/100, IELTS 6.0, leadership "
            "potential, and commitment to return home to apply their New Zealand education "
            "for national development in agriculture, technology, or public policy."
        ),
    )]


def scrape_moe_taiwan() -> list[Scholarship]:
    """Taiwan MOE Scholarship — Ministry of Education Taiwan for international students."""
    fields = ["computer_science", "engineering", "economics", "arts_humanities",
              "social_sciences", "mathematics", "biology"]
    lang_reqs = [LanguageRequirement(test_type="ielts", min_score=5.5, is_mandatory=False),
                 LanguageRequirement(test_type="toefl", min_score=61.0, is_mandatory=False),
                 LanguageRequirement(test_type="hsk", min_score=180.0, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="Taiwan MOE Scholarship (Ministry of Education)",
        eligible_nationalities=["indonesia", "vietnam", "malaysia", "thailand", "philippines",
                                 "india", "south_africa", "nigeria"],
        min_age=17,
        max_age=30,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="public_a",
        min_report_card_average=75.0,
        min_major_subject_average=70.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="japan",  # Taiwan not in Country enum; japan as nearest East Asian placeholder
        host_region="asia",
        selection_criteria=SelectionCriteria(
            academic=0.40, leadership=0.15, olympiad=0.10, extracurricular=0.15, essay=0.20
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=False, covers_living_expense=True,
            covers_airfare=False, covers_insurance=False, monthly_stipend=15_000.0
        ),
        career_track_preference=infer_career_from_fields(fields),
        requires_return_home_country=False,
        mission_statement=(
            "The Taiwan Ministry of Education Scholarship promotes Taiwan as a premier study "
            "destination by supporting outstanding international students to study at accredited "
            "Taiwanese universities in STEM, humanities, and social sciences."
        ),
        target_recipient_profile=(
            "We seek students with a minimum GPA of 75/100 who are motivated to study in "
            "Taiwan's vibrant academic environment, with English, Mandarin, or a relevant "
            "language proficiency and strong academic background in their chosen field."
        ),
    )]


def scrape_shanghai_government() -> list[Scholarship]:
    """Shanghai Government Scholarship — for international students at Shanghai universities."""
    fields = ["computer_science", "engineering", "economics", "business", "mathematics",
              "medicine", "chemistry"]
    lang_reqs = [LanguageRequirement(test_type="hsk", min_score=180.0, is_mandatory=False),
                 LanguageRequirement(test_type="toefl", min_score=60.0, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="Shanghai Government Scholarship",
        eligible_nationalities=["indonesia", "vietnam", "malaysia", "thailand", "philippines",
                                 "india", "egypt", "nigeria", "kenya", "south_africa", "brazil"],
        min_age=17,
        max_age=25,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="public_a",
        min_report_card_average=75.0,
        min_major_subject_average=70.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="china",
        host_region="asia",
        selection_criteria=SelectionCriteria(
            academic=0.45, leadership=0.10, olympiad=0.15, extracurricular=0.10, essay=0.20
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=False, covers_insurance=True, monthly_stipend=1_500.0
        ),
        career_track_preference=infer_career_from_fields(fields),
        requires_return_home_country=False,
        mission_statement=(
            "The Shanghai Government Scholarship attracts talented students from around the world "
            "to study at Shanghai's world-renowned universities, fostering international exchange "
            "and preparing global professionals for the 21st-century economy."
        ),
        target_recipient_profile=(
            "We seek academically strong international students with a minimum GPA of 75/100 "
            "and HSK or English proficiency, eager to study in Shanghai's dynamic academic "
            "environment in science, technology, economics, or medicine."
        ),
    )]


def scrape_beijing_government() -> list[Scholarship]:
    """Beijing Government Scholarship — for international students at Beijing universities."""
    fields = ["computer_science", "engineering", "medicine", "economics", "business",
              "arts_humanities", "social_sciences", "mathematics"]
    lang_reqs = [LanguageRequirement(test_type="hsk", min_score=180.0, is_mandatory=False),
                 LanguageRequirement(test_type="ielts", min_score=5.5, is_mandatory=False),
                 LanguageRequirement(test_type="toefl", min_score=61.0, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="Beijing Government Scholarship",
        eligible_nationalities=["indonesia", "vietnam", "malaysia", "thailand", "philippines",
                                 "india", "egypt", "nigeria", "kenya", "south_africa", "brazil"],
        min_age=17,
        max_age=25,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="public_a",
        min_report_card_average=78.0,
        min_major_subject_average=75.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="china",
        host_region="asia",
        selection_criteria=SelectionCriteria(
            academic=0.45, leadership=0.10, olympiad=0.15, extracurricular=0.10, essay=0.20
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=False, covers_insurance=True, monthly_stipend=2_500.0
        ),
        career_track_preference=infer_career_from_fields(fields),
        requires_return_home_country=False,
        mission_statement=(
            "The Beijing Government Scholarship supports international students in pursuing "
            "undergraduate education at Beijing's elite universities, promoting cross-cultural "
            "understanding and building global networks in China's capital."
        ),
        target_recipient_profile=(
            "We seek students with a minimum GPA of 78/100, HSK or English proficiency, "
            "and strong academic performance in STEM, medicine, or humanities who wish to "
            "study at Peking University, Tsinghua, or other Beijing institutions."
        ),
    )]


def scrape_kaist_global() -> list[Scholarship]:
    """KAIST Global Scholarship — for outstanding international undergrads in STEM."""
    fields = ["computer_science", "engineering", "mathematics", "physics", "chemistry", "biology"]
    lang_reqs = [LanguageRequirement(test_type="toefl", min_score=83.0, is_mandatory=True),
                 LanguageRequirement(test_type="ielts", min_score=6.5, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="KAIST Global Scholarship (Undergraduate)",
        eligible_nationalities=["indonesia", "vietnam", "malaysia", "thailand", "philippines",
                                 "india", "china", "south_africa", "nigeria", "kenya", "brazil"],
        min_age=17,
        max_age=20,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=["science"],
        eligible_fields=fields,
        preferred_school_tier="excellence",
        min_report_card_average=88.0,
        min_major_subject_average=87.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="south_korea",
        host_region="asia",
        selection_criteria=SelectionCriteria(
            academic=0.55, leadership=0.10, olympiad=0.25, extracurricular=0.05, essay=0.05
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=False, covers_insurance=True, monthly_stipend=350_000.0
        ),
        career_track_preference="research",
        requires_return_home_country=False,
        mission_statement=(
            "KAIST Global Scholarship identifies and nurtures the world's most exceptional "
            "young scientists and engineers, providing a fully funded undergraduate education "
            "at one of Asia's leading science and technology institutions."
        ),
        target_recipient_profile=(
            "We seek exceptional students with a minimum GPA of 88/100, TOEFL 83 or IELTS 6.5, "
            "and distinguished achievements in science or mathematics olympiads, who aspire to "
            "become pioneering researchers and innovators in STEM fields."
        ),
    )]


def scrape_yonsei_international() -> list[Scholarship]:
    """Yonsei University International Scholarship — for global undergraduate students."""
    fields = ["computer_science", "engineering", "business", "economics", "arts_humanities",
              "social_sciences", "medicine"]
    lang_reqs = [LanguageRequirement(test_type="toefl", min_score=80.0, is_mandatory=False),
                 LanguageRequirement(test_type="ielts", min_score=6.0, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="Yonsei University International Scholarship",
        eligible_nationalities=["indonesia", "vietnam", "malaysia", "thailand", "philippines",
                                 "india", "china", "south_africa", "nigeria", "brazil", "argentina"],
        min_age=17,
        max_age=22,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="excellence",
        min_report_card_average=82.0,
        min_major_subject_average=80.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="south_korea",
        host_region="asia",
        selection_criteria=SelectionCriteria(
            academic=0.45, leadership=0.15, olympiad=0.15, extracurricular=0.10, essay=0.15
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=False, covers_insurance=False, monthly_stipend=500_000.0
        ),
        career_track_preference=infer_career_from_fields(fields),
        requires_return_home_country=False,
        mission_statement=(
            "Yonsei University International Scholarship attracts globally talented students "
            "to join one of Korea's most prestigious universities, offering excellence in "
            "research, leadership development, and intercultural education."
        ),
        target_recipient_profile=(
            "We seek high-achieving international students with a minimum GPA of 82/100 and "
            "English proficiency (TOEFL 80 or IELTS 6.0) who demonstrate academic excellence, "
            "leadership, and commitment to making a global impact."
        ),
    )]


def scrape_panasonic_scholarship() -> list[Scholarship]:
    """Panasonic Scholarship Japan — NPO supporting ASEAN/South Asian students in Japan."""
    fields = ["engineering", "computer_science", "physics", "mathematics", "chemistry"]
    lang_reqs = [LanguageRequirement(test_type="toefl", min_score=80.0, is_mandatory=False),
                 LanguageRequirement(test_type="ielts", min_score=6.0, is_mandatory=False),
                 LanguageRequirement(test_type="jlpt", min_score=60.0, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="Panasonic Scholarship Japan",
        eligible_nationalities=["indonesia", "malaysia", "thailand", "philippines", "vietnam",
                                 "india", "china"],
        min_age=17,
        max_age=22,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=["science"],
        eligible_fields=fields,
        preferred_school_tier="public_a",
        min_report_card_average=80.0,
        min_major_subject_average=78.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="japan",
        host_region="asia",
        selection_criteria=SelectionCriteria(
            academic=0.50, leadership=0.10, olympiad=0.15, extracurricular=0.10, essay=0.15
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=False, covers_living_expense=True,
            covers_airfare=False, covers_insurance=False, monthly_stipend=150_000.0
        ),
        career_track_preference="industry",
        requires_return_home_country=True,
        mission_statement=(
            "The Panasonic Scholarship supports talented students from Asia to study engineering "
            "and science in Japan, building future technology leaders who will contribute to "
            "sustainable development in their home countries and the broader Asia-Pacific region."
        ),
        target_recipient_profile=(
            "We seek science and engineering students with a minimum GPA of 80/100 from ASEAN "
            "or South Asian countries, who demonstrate strong analytical skills, a passion for "
            "technology, and commitment to returning home to apply their expertise."
        ),
    )]


def scrape_kyoto_asean() -> list[Scholarship]:
    """Kyoto University ASEAN Scholarship — for ASEAN undergrads at Kyoto University."""
    fields = ["engineering", "computer_science", "mathematics", "physics", "biology",
              "social_sciences", "agriculture"]
    lang_reqs = [LanguageRequirement(test_type="toefl", min_score=72.0, is_mandatory=False),
                 LanguageRequirement(test_type="ielts", min_score=5.5, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="Kyoto University ASEAN Scholarship",
        eligible_nationalities=["indonesia", "malaysia", "thailand", "philippines", "vietnam"],
        min_age=17,
        max_age=21,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="public_a",
        min_report_card_average=82.0,
        min_major_subject_average=80.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="japan",
        host_region="asia",
        selection_criteria=SelectionCriteria(
            academic=0.50, leadership=0.10, olympiad=0.20, extracurricular=0.10, essay=0.10
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=True, covers_insurance=True, monthly_stipend=147_000.0
        ),
        career_track_preference="research",
        requires_return_home_country=False,
        mission_statement=(
            "Kyoto University ASEAN Scholarship brings the brightest minds from Southeast Asia "
            "to one of Japan's most distinguished research universities, fostering academic "
            "excellence and ASEAN-Japan scholarly collaboration across science and humanities."
        ),
        target_recipient_profile=(
            "We seek ASEAN students with a minimum GPA of 82/100 and English proficiency, "
            "with strong aptitude in science, mathematics, or engineering and a passion for "
            "research-oriented undergraduate study at a top-ranked Japanese university."
        ),
    )]


def scrape_orange_tulip() -> list[Scholarship]:
    """Orange Tulip Scholarship — Dutch universities scholarship for select Asian countries."""
    fields = ["computer_science", "engineering", "economics", "business", "agriculture",
              "arts_humanities", "social_sciences"]
    lang_reqs = [LanguageRequirement(test_type="ielts", min_score=6.0, is_mandatory=True),
                 LanguageRequirement(test_type="toefl", min_score=79.0, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="Orange Tulip Scholarship (Netherlands)",
        eligible_nationalities=["indonesia", "china", "india", "vietnam", "malaysia", "thailand"],
        min_age=17,
        max_age=26,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="public_a",
        min_report_card_average=78.0,
        min_major_subject_average=75.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="netherlands",
        host_region="europe",
        selection_criteria=SelectionCriteria(
            academic=0.35, leadership=0.15, olympiad=0.10, extracurricular=0.15, essay=0.25
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=False, covers_living_expense=False,
            covers_airfare=False, covers_insurance=False, monthly_stipend=290.0
        ),
        career_track_preference=infer_career_from_fields(fields),
        requires_return_home_country=False,
        mission_statement=(
            "The Orange Tulip Scholarship connects talented students from Asia with Dutch "
            "universities of applied sciences and research universities, opening doors to "
            "the Netherlands' world-class education and innovative learning environment."
        ),
        target_recipient_profile=(
            "We seek motivated students from Indonesia, China, India, Vietnam, Malaysia, or "
            "Thailand with a minimum GPA of 78/100 and IELTS 6.0, who can demonstrate "
            "their potential to thrive in the Netherlands' problem-based learning culture."
        ),
    )]


def scrape_holland_scholarship() -> list[Scholarship]:
    """Holland Scholarship — for non-EEA students at Dutch research universities."""
    fields = ["computer_science", "engineering", "economics", "business", "agriculture",
              "arts_humanities", "social_sciences", "mathematics"]
    lang_reqs = [LanguageRequirement(test_type="ielts", min_score=6.0, is_mandatory=True),
                 LanguageRequirement(test_type="toefl", min_score=80.0, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="Holland Scholarship",
        eligible_nationalities=["indonesia", "vietnam", "malaysia", "thailand", "philippines",
                                 "india", "china", "south_africa", "nigeria", "kenya", "brazil"],
        min_age=17,
        max_age=30,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="public_a",
        min_report_card_average=78.0,
        min_major_subject_average=75.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="netherlands",
        host_region="europe",
        selection_criteria=SelectionCriteria(
            academic=0.35, leadership=0.15, olympiad=0.10, extracurricular=0.15, essay=0.25
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=False, covers_living_expense=False,
            covers_airfare=False, covers_insurance=False, monthly_stipend=416.0
        ),
        career_track_preference=infer_career_from_fields(fields),
        requires_return_home_country=False,
        mission_statement=(
            "The Holland Scholarship is offered by Dutch research universities and universities "
            "of applied sciences to attract talented non-EEA students, providing financial "
            "support for their first year of bachelor's or master's study in the Netherlands."
        ),
        target_recipient_profile=(
            "We seek excellent international students from outside the EEA with a minimum GPA "
            "of 78/100 and IELTS 6.0, who wish to study at a Dutch university and can "
            "demonstrate academic distinction and motivation in their chosen discipline."
        ),
    )]


def scrape_czech_government() -> list[Scholarship]:
    """Czech Government Scholarship (DZS) — for students from developing countries."""
    fields = ["engineering", "agriculture", "medicine", "education", "economics", "arts_humanities",
              "computer_science"]
    lang_reqs = [LanguageRequirement(test_type="ielts", min_score=5.5, is_mandatory=False),
                 LanguageRequirement(test_type="toefl", min_score=65.0, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="Czech Government Scholarship (DZS)",
        eligible_nationalities=["indonesia", "vietnam", "malaysia", "thailand", "india",
                                 "egypt", "morocco", "nigeria", "kenya", "south_africa"],
        min_age=17,
        max_age=26,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="public_a",
        min_report_card_average=73.0,
        min_major_subject_average=68.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="germany",  # Czech Republic not in Country enum; germany as Central European placeholder
        host_region="europe",
        selection_criteria=SelectionCriteria(
            academic=0.40, leadership=0.15, olympiad=0.10, extracurricular=0.15, essay=0.20
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=False, covers_insurance=True, monthly_stipend=3_000.0
        ),
        career_track_preference=infer_career_from_fields(fields),
        requires_return_home_country=False,
        mission_statement=(
            "The Czech Government Scholarship, administered by the House of International "
            "Cooperation (DZS), offers students from developing countries access to quality "
            "Czech higher education in technical, agricultural, and humanitarian disciplines."
        ),
        target_recipient_profile=(
            "We seek students from partner developing countries with a minimum GPA of 73/100, "
            "English or Czech language ability, and academic interest in engineering, "
            "agriculture, medicine, or humanities at Czech public universities."
        ),
    )]


def scrape_nawa_poland() -> list[Scholarship]:
    """Polish Government Scholarship (NAWA) — for students from developing countries."""
    fields = ["engineering", "computer_science", "economics", "social_sciences", "arts_humanities",
              "agriculture", "mathematics"]
    lang_reqs = [LanguageRequirement(test_type="ielts", min_score=5.5, is_mandatory=False),
                 LanguageRequirement(test_type="toefl", min_score=65.0, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="Polish Government Scholarship (NAWA)",
        eligible_nationalities=["indonesia", "vietnam", "malaysia", "thailand", "india",
                                 "egypt", "morocco", "nigeria", "kenya", "south_africa"],
        min_age=17,
        max_age=26,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="public_a",
        min_report_card_average=73.0,
        min_major_subject_average=68.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="germany",  # Poland not in Country enum; germany as Central European placeholder
        host_region="europe",
        selection_criteria=SelectionCriteria(
            academic=0.40, leadership=0.15, olympiad=0.10, extracurricular=0.15, essay=0.20
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=False, covers_insurance=True, monthly_stipend=1_250.0
        ),
        career_track_preference=infer_career_from_fields(fields),
        requires_return_home_country=False,
        mission_statement=(
            "The Polish National Agency for Academic Exchange (NAWA) offers scholarships "
            "for students from developing nations to pursue undergraduate education at "
            "Polish universities, strengthening academic and cultural ties worldwide."
        ),
        target_recipient_profile=(
            "We seek students from partner countries with a minimum GPA of 73/100 and "
            "English or Polish proficiency, interested in engineering, computer science, "
            "economics, or humanities at accredited Polish institutions."
        ),
    )]


def scrape_iccr_india() -> list[Scholarship]:
    """ICCR Scholarship (India) — for students from developing and partner countries."""
    fields = ["education", "arts_humanities", "social_sciences", "medicine", "agriculture",
              "engineering", "economics"]
    lang_reqs = [LanguageRequirement(test_type="toefl", min_score=55.0, is_mandatory=False),
                 LanguageRequirement(test_type="ielts", min_score=5.0, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="ICCR Scholarship (India — Indian Council for Cultural Relations)",
        eligible_nationalities=["indonesia", "malaysia", "vietnam", "thailand", "philippines",
                                 "egypt", "morocco", "nigeria", "kenya", "south_africa"],
        min_age=17,
        max_age=25,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="public_a",
        min_report_card_average=70.0,
        min_major_subject_average=65.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="india",
        host_region="asia",
        selection_criteria=SelectionCriteria(
            academic=0.35, leadership=0.20, olympiad=0.10, extracurricular=0.15, essay=0.20
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=False, covers_insurance=False, monthly_stipend=10_000.0
        ),
        career_track_preference=infer_career_from_fields(fields),
        requires_return_home_country=False,
        mission_statement=(
            "The Indian Council for Cultural Relations (ICCR) Scholarship promotes India's "
            "educational excellence and people-to-people diplomacy by supporting students "
            "from partner countries to study at Indian universities across all disciplines."
        ),
        target_recipient_profile=(
            "We seek students from partner and developing countries with a minimum GPA of "
            "70/100 and English proficiency, who are curious about India's rich academic "
            "tradition and wish to study in a culturally diverse university environment."
        ),
    )]


def scrape_isdb_merit() -> list[Scholarship]:
    """IsDB Merit Scholarship — Islamic Development Bank for OIC member country students."""
    fields = ["engineering", "computer_science", "medicine", "agriculture", "economics", "education"]
    lang_reqs = [LanguageRequirement(test_type="toefl", min_score=60.0, is_mandatory=False),
                 LanguageRequirement(test_type="ielts", min_score=5.5, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="IsDB Merit Scholarship (Islamic Development Bank)",
        eligible_nationalities=["indonesia", "malaysia", "egypt", "morocco", "nigeria",
                                 "kenya", "south_africa", "india", "bangladesh"],
        min_age=17,
        max_age=25,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="public_a",
        min_report_card_average=75.0,
        min_major_subject_average=70.0,
        language_requirements=lang_reqs,
        requires_financial_need=True,
        max_family_income_category="middle",
        host_country="egypt",  # Varies by OIC member country; egypt as representative
        host_region="africa",
        selection_criteria=SelectionCriteria(
            academic=0.40, leadership=0.20, olympiad=0.10, extracurricular=0.10, essay=0.20
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=True, covers_insurance=True, monthly_stipend=500.0
        ),
        career_track_preference=infer_career_from_fields(fields),
        requires_return_home_country=True,
        mission_statement=(
            "The IsDB Merit Scholarship supports high-achieving students from OIC member "
            "countries to pursue undergraduate education at universities within the Islamic "
            "world and beyond, contributing to sustainable development in Muslim-majority nations."
        ),
        target_recipient_profile=(
            "We seek merit students from OIC member countries with a minimum GPA of 75/100, "
            "demonstrated financial need, and commitment to applying their education for the "
            "development of their home country in priority sectors like health, agriculture, "
            "and engineering."
        ),
    )]


def scrape_amci_morocco() -> list[Scholarship]:
    """Moroccan Government Scholarship (AMCI) — for students from Africa and Muslim-majority countries."""
    fields = ["engineering", "computer_science", "medicine", "agriculture", "economics",
              "law", "arts_humanities"]
    lang_reqs = [LanguageRequirement(test_type="toefl", min_score=50.0, is_mandatory=False),
                 LanguageRequirement(test_type="ielts", min_score=5.0, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="Moroccan Government Scholarship (AMCI)",
        eligible_nationalities=["indonesia", "malaysia", "egypt", "nigeria", "kenya",
                                 "south_africa", "india", "vietnam"],
        min_age=17,
        max_age=26,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="accredited_b",
        min_report_card_average=70.0,
        min_major_subject_average=65.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="morocco",
        host_region="africa",
        selection_criteria=SelectionCriteria(
            academic=0.40, leadership=0.15, olympiad=0.05, extracurricular=0.15, essay=0.25
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=False, covers_insurance=False, monthly_stipend=1_000.0
        ),
        career_track_preference=infer_career_from_fields(fields),
        requires_return_home_country=False,
        mission_statement=(
            "The Moroccan Agency for International Cooperation (AMCI) scholarship invites "
            "students from partner countries to study at Morocco's universities, positioning "
            "Morocco as a gateway for education and cooperation between Africa and the world."
        ),
        target_recipient_profile=(
            "We seek students from partner countries with a minimum GPA of 70/100 and "
            "French or Arabic proficiency, interested in engineering, law, medicine, or "
            "agriculture at Moroccan public universities."
        ),
    )]


def scrape_al_azhar() -> list[Scholarship]:
    """Al-Azhar Scholarship (Egypt) — for Muslim students worldwide, well-known to Indonesians."""
    fields = ["education", "arts_humanities", "social_sciences", "law"]
    lang_reqs = [LanguageRequirement(test_type="toefl", min_score=0.0, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="Al-Azhar University Scholarship (Egypt)",
        eligible_nationalities=["indonesia", "malaysia", "egypt", "morocco", "nigeria",
                                 "kenya", "south_africa", "india", "vietnam", "thailand",
                                 "philippines"],
        min_age=16,
        max_age=25,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=["social_studies", "languages"],
        eligible_fields=fields,
        preferred_school_tier="accredited_b",
        min_report_card_average=70.0,
        min_major_subject_average=65.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="egypt",
        host_region="africa",
        selection_criteria=SelectionCriteria(
            academic=0.40, leadership=0.10, olympiad=0.05, extracurricular=0.15, essay=0.30
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=False, covers_insurance=False, monthly_stipend=500.0
        ),
        career_track_preference="education",
        requires_return_home_country=False,
        mission_statement=(
            "Al-Azhar University, one of the world's oldest Islamic institutions, offers "
            "scholarships to Muslim students from across the globe to study Islamic sciences, "
            "Arabic language, law, and humanities in Cairo, Egypt."
        ),
        target_recipient_profile=(
            "We seek Muslim students with a minimum GPA of 70/100 and demonstrable Arabic "
            "language proficiency or willingness to learn, who aspire to study Islamic "
            "sciences, humanities, or law at one of Islam's most revered universities."
        ),
    )]


def scrape_snu_international() -> list[Scholarship]:
    """Seoul National University (SNU) International Scholarship — undergrad."""
    fields = ["computer_science", "engineering", "economics", "medicine", "mathematics",
              "social_sciences", "arts_humanities", "biology"]
    lang_reqs = [LanguageRequirement(test_type="toefl", min_score=83.0, is_mandatory=False),
                 LanguageRequirement(test_type="ielts", min_score=6.5, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="Seoul National University (SNU) International Scholarship",
        eligible_nationalities=["indonesia", "vietnam", "malaysia", "thailand", "philippines",
                                 "india", "china", "brazil", "south_africa", "nigeria"],
        min_age=17,
        max_age=21,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="excellence",
        min_report_card_average=85.0,
        min_major_subject_average=83.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="south_korea",
        host_region="asia",
        selection_criteria=SelectionCriteria(
            academic=0.50, leadership=0.15, olympiad=0.20, extracurricular=0.10, essay=0.05
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=False, covers_insurance=True, monthly_stipend=400_000.0
        ),
        career_track_preference=infer_career_from_fields(fields),
        requires_return_home_country=False,
        mission_statement=(
            "Seoul National University International Scholarship attracts the world's brightest "
            "students to Korea's most prestigious university, cultivating global leaders who "
            "will drive innovation and advance knowledge across disciplines."
        ),
        target_recipient_profile=(
            "We seek internationally distinguished students with a minimum GPA of 85/100, "
            "TOEFL 83 or IELTS 6.5, and a record of academic excellence or olympiad "
            "achievement, who aspire to study at Korea's leading research university."
        ),
    )]


def scrape_postech_global() -> list[Scholarship]:
    """POSTECH Global Scholarship — Pohang University of Science and Technology."""
    fields = ["computer_science", "engineering", "mathematics", "physics", "chemistry", "biology"]
    lang_reqs = [LanguageRequirement(test_type="toefl", min_score=80.0, is_mandatory=True),
                 LanguageRequirement(test_type="ielts", min_score=6.0, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="POSTECH Global Scholarship",
        eligible_nationalities=["indonesia", "vietnam", "malaysia", "thailand", "philippines",
                                 "india", "china", "south_africa", "nigeria", "kenya", "brazil"],
        min_age=17,
        max_age=20,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=["science"],
        eligible_fields=fields,
        preferred_school_tier="excellence",
        min_report_card_average=87.0,
        min_major_subject_average=85.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="south_korea",
        host_region="asia",
        selection_criteria=SelectionCriteria(
            academic=0.55, leadership=0.10, olympiad=0.25, extracurricular=0.05, essay=0.05
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=False, covers_insurance=True, monthly_stipend=300_000.0
        ),
        career_track_preference="research",
        requires_return_home_country=False,
        mission_statement=(
            "POSTECH Global Scholarship identifies the most talented science and engineering "
            "students worldwide, offering a fully funded undergraduate experience at one of "
            "Asia's most research-intensive universities in Pohang, South Korea."
        ),
        target_recipient_profile=(
            "We seek elite science students with a minimum GPA of 87/100, TOEFL 80, and "
            "exceptional aptitude in mathematics, physics, or computer science, ideally "
            "with international olympiad experience or research exposure."
        ),
    )]


def scrape_skku_global() -> list[Scholarship]:
    """Sungkyunkwan University (SKKU) Global Scholarship — undergrad track."""
    fields = ["computer_science", "engineering", "business", "economics", "social_sciences",
              "arts_humanities", "medicine"]
    lang_reqs = [LanguageRequirement(test_type="toefl", min_score=71.0, is_mandatory=False),
                 LanguageRequirement(test_type="ielts", min_score=5.5, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="Sungkyunkwan University (SKKU) Global Scholarship",
        eligible_nationalities=["indonesia", "vietnam", "malaysia", "thailand", "philippines",
                                 "india", "china", "south_africa", "nigeria", "brazil", "argentina"],
        min_age=17,
        max_age=22,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="public_a",
        min_report_card_average=80.0,
        min_major_subject_average=78.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="south_korea",
        host_region="asia",
        selection_criteria=SelectionCriteria(
            academic=0.45, leadership=0.15, olympiad=0.15, extracurricular=0.10, essay=0.15
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=False,
            covers_airfare=False, covers_insurance=False, monthly_stipend=0.0
        ),
        career_track_preference=infer_career_from_fields(fields),
        requires_return_home_country=False,
        mission_statement=(
            "SKKU Global Scholarship opens doors to one of Korea's oldest and most respected "
            "universities, attracting talented international students to a campus that bridges "
            "600 years of Confucian scholarship with cutting-edge global education."
        ),
        target_recipient_profile=(
            "We seek motivated international students with a minimum GPA of 80/100 and "
            "English proficiency (TOEFL 71 or IELTS 5.5), who are eager to engage with "
            "SKKU's diverse academic community in Seoul or Suwon."
        ),
    )]


def scrape_korea_university_global() -> list[Scholarship]:
    """Korea University Global Excellence Scholarship — undergrad track."""
    fields = ["computer_science", "engineering", "business", "economics", "law",
              "social_sciences", "arts_humanities"]
    lang_reqs = [LanguageRequirement(test_type="toefl", min_score=80.0, is_mandatory=False),
                 LanguageRequirement(test_type="ielts", min_score=6.0, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="Korea University Global Excellence Scholarship",
        eligible_nationalities=["indonesia", "vietnam", "malaysia", "thailand", "philippines",
                                 "india", "china", "south_africa", "nigeria", "brazil", "argentina"],
        min_age=17,
        max_age=22,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="excellence",
        min_report_card_average=83.0,
        min_major_subject_average=80.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="south_korea",
        host_region="asia",
        selection_criteria=SelectionCriteria(
            academic=0.45, leadership=0.20, olympiad=0.10, extracurricular=0.10, essay=0.15
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=False,
            covers_airfare=False, covers_insurance=False, monthly_stipend=0.0
        ),
        career_track_preference=infer_career_from_fields(fields),
        requires_return_home_country=False,
        mission_statement=(
            "Korea University Global Excellence Scholarship invites outstanding international "
            "students to one of Korea's SKY universities, offering full tuition support and "
            "access to a vibrant research and leadership development ecosystem."
        ),
        target_recipient_profile=(
            "We seek globally minded students with a minimum GPA of 83/100 and TOEFL 80 "
            "or IELTS 6.0, who demonstrate academic distinction, leadership, and a "
            "commitment to excellence at one of Korea's most prestigious institutions."
        ),
    )]


def scrape_waseda_international() -> list[Scholarship]:
    """Waseda University International Student Scholarship — undergrad track, Japan."""
    fields = ["computer_science", "engineering", "economics", "business", "social_sciences",
              "arts_humanities", "mathematics"]
    lang_reqs = [LanguageRequirement(test_type="toefl", min_score=72.0, is_mandatory=False),
                 LanguageRequirement(test_type="ielts", min_score=5.5, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="Waseda University International Student Scholarship",
        eligible_nationalities=["indonesia", "vietnam", "malaysia", "thailand", "philippines",
                                 "india", "china", "south_korea", "brazil", "south_africa"],
        min_age=17,
        max_age=22,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="public_a",
        min_report_card_average=80.0,
        min_major_subject_average=77.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="japan",
        host_region="asia",
        selection_criteria=SelectionCriteria(
            academic=0.45, leadership=0.15, olympiad=0.10, extracurricular=0.15, essay=0.15
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=False, covers_living_expense=True,
            covers_airfare=False, covers_insurance=False, monthly_stipend=80_000.0
        ),
        career_track_preference=infer_career_from_fields(fields),
        requires_return_home_country=False,
        mission_statement=(
            "Waseda University International Student Scholarship supports talented international "
            "students studying at one of Japan's most globally renowned private universities, "
            "nurturing cross-cultural exchange and academic excellence in Tokyo."
        ),
        target_recipient_profile=(
            "We seek international students with a minimum GPA of 80/100 and English or "
            "Japanese proficiency who are enrolled in or applying to Waseda's English-taught "
            "undergraduate programs in social sciences, engineering, or business."
        ),
    )]


def scrape_osaka_global_scholars() -> list[Scholarship]:
    """Osaka University Global Scholars Program — ASEAN-focused undergrad scholarship."""
    fields = ["engineering", "computer_science", "mathematics", "physics", "chemistry",
              "biology", "medicine", "economics"]
    lang_reqs = [LanguageRequirement(test_type="toefl", min_score=72.0, is_mandatory=False),
                 LanguageRequirement(test_type="ielts", min_score=5.5, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="Osaka University Global Scholars Program",
        eligible_nationalities=["indonesia", "vietnam", "malaysia", "thailand", "philippines",
                                 "india", "china"],
        min_age=17,
        max_age=21,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="public_a",
        min_report_card_average=82.0,
        min_major_subject_average=80.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="japan",
        host_region="asia",
        selection_criteria=SelectionCriteria(
            academic=0.50, leadership=0.10, olympiad=0.20, extracurricular=0.10, essay=0.10
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=True, covers_insurance=True, monthly_stipend=117_000.0
        ),
        career_track_preference="research",
        requires_return_home_country=False,
        mission_statement=(
            "Osaka University Global Scholars Program recruits outstanding undergraduate "
            "students from Asia to join a world-class research environment, building the "
            "next generation of scientists and engineers with a global perspective."
        ),
        target_recipient_profile=(
            "We seek science and engineering students from ASEAN and partner countries with "
            "a minimum GPA of 82/100 and strong aptitude in mathematics or sciences, "
            "motivated to engage in cutting-edge research at Osaka University."
        ),
    )]


def scrape_fudan_international() -> list[Scholarship]:
    """Fudan University International Students Scholarship — Type A (full) and Type B (partial)."""
    scholarships_out = []

    fields = ["computer_science", "engineering", "economics", "social_sciences",
              "arts_humanities", "medicine", "business", "mathematics"]
    lang_reqs = [LanguageRequirement(test_type="hsk", min_score=180.0, is_mandatory=False),
                 LanguageRequirement(test_type="toefl", min_score=60.0, is_mandatory=False),
                 LanguageRequirement(test_type="ielts", min_score=5.5, is_mandatory=False)]

    # Type A — full scholarship
    scholarships_out.append(Scholarship(
        scholarship_id="",
        name="Fudan University International Scholarship (Type A)",
        eligible_nationalities=["indonesia", "vietnam", "malaysia", "thailand", "philippines",
                                 "india", "south_africa", "nigeria", "kenya", "brazil", "argentina"],
        min_age=17,
        max_age=25,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="excellence",
        min_report_card_average=85.0,
        min_major_subject_average=83.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="china",
        host_region="asia",
        selection_criteria=SelectionCriteria(
            academic=0.50, leadership=0.15, olympiad=0.15, extracurricular=0.10, essay=0.10
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=False, covers_insurance=True, monthly_stipend=2_500.0
        ),
        career_track_preference=infer_career_from_fields(fields),
        requires_return_home_country=False,
        mission_statement=(
            "Fudan University International Scholarship (Type A) recognizes top-performing "
            "international students and funds their full undergraduate journey at one of "
            "China's most prestigious and historically distinguished universities in Shanghai."
        ),
        target_recipient_profile=(
            "We seek outstanding students with a minimum GPA of 85/100, HSK or English "
            "proficiency, and a demonstrated record of academic excellence, who wish to "
            "study at Fudan's internationally ranked programs in Shanghai."
        ),
    ))

    # Type B — partial scholarship (tuition only)
    scholarships_out.append(Scholarship(
        scholarship_id="",
        name="Fudan University International Scholarship (Type B)",
        eligible_nationalities=["indonesia", "vietnam", "malaysia", "thailand", "philippines",
                                 "india", "south_africa", "nigeria", "kenya", "brazil", "argentina"],
        min_age=17,
        max_age=25,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="public_a",
        min_report_card_average=78.0,
        min_major_subject_average=75.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="china",
        host_region="asia",
        selection_criteria=SelectionCriteria(
            academic=0.45, leadership=0.15, olympiad=0.10, extracurricular=0.15, essay=0.15
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=False,
            covers_airfare=False, covers_insurance=False, monthly_stipend=0.0
        ),
        career_track_preference=infer_career_from_fields(fields),
        requires_return_home_country=False,
        mission_statement=(
            "Fudan University International Scholarship (Type B) provides tuition support "
            "to motivated international students, making world-class education in Shanghai "
            "accessible to talented students from partner countries."
        ),
        target_recipient_profile=(
            "We seek students with a minimum GPA of 78/100 and HSK or English proficiency "
            "who are committed to academic excellence and wish to study at Fudan University "
            "in one of China's most dynamic and internationally connected cities."
        ),
    ))

    return scholarships_out


def scrape_campus_france() -> list[Scholarship]:
    """French Government Scholarship (Campus France / Excellence Major) — for undergrads."""
    fields = ["engineering", "computer_science", "economics", "arts_humanities",
              "social_sciences", "mathematics", "agriculture"]
    lang_reqs = [LanguageRequirement(test_type="ielts", min_score=5.5, is_mandatory=False),
                 LanguageRequirement(test_type="toefl", min_score=65.0, is_mandatory=False)]

    return [Scholarship(
        scholarship_id="",
        name="French Government Scholarship (Bourse du Gouvernement Français)",
        eligible_nationalities=["indonesia", "vietnam", "malaysia", "thailand", "philippines",
                                 "india", "morocco", "egypt", "nigeria", "kenya", "south_africa"],
        min_age=17,
        max_age=28,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="public_a",
        min_report_card_average=78.0,
        min_major_subject_average=75.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="france",
        host_region="europe",
        selection_criteria=SelectionCriteria(
            academic=0.40, leadership=0.15, olympiad=0.10, extracurricular=0.15, essay=0.20
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=False, covers_insurance=True, monthly_stipend=700.0
        ),
        career_track_preference=infer_career_from_fields(fields),
        requires_return_home_country=False,
        mission_statement=(
            "The French Government Scholarship promotes France as a world-leading higher "
            "education destination by supporting talented international students to pursue "
            "undergraduate studies at prestigious French grandes écoles and universities."
        ),
        target_recipient_profile=(
            "We seek academically strong students with a minimum GPA of 78/100, English or "
            "French language proficiency, and genuine motivation to study in France's "
            "internationally respected academic and cultural environment."
        ),
    )]


def scrape_norway_quota() -> list[Scholarship]:
    """Norwegian Quota Scheme / NORPART — for students from developing countries."""
    fields = ["engineering", "computer_science", "economics", "agriculture", "social_sciences",
              "education", "mathematics", "biology"]
    lang_reqs = [LanguageRequirement(test_type="ielts", min_score=6.0, is_mandatory=True)]

    return [Scholarship(
        scholarship_id="",
        name="Norwegian Government Scholarship (Quota Scheme / NORPART)",
        eligible_nationalities=["indonesia", "vietnam", "malaysia", "thailand", "india",
                                 "egypt", "nigeria", "kenya", "south_africa", "morocco"],
        min_age=17,
        max_age=35,
        eligible_degree_levels=["high_school"],
        eligible_high_school_tracks=infer_tracks_from_fields(fields),
        eligible_fields=fields,
        preferred_school_tier="public_a",
        min_report_card_average=75.0,
        min_major_subject_average=72.0,
        language_requirements=lang_reqs,
        requires_financial_need=False,
        max_family_income_category="high",
        host_country="sweden",  # Norway not in Country enum; sweden as nearest Scandinavian placeholder
        host_region="europe",
        selection_criteria=SelectionCriteria(
            academic=0.35, leadership=0.20, olympiad=0.05, extracurricular=0.20, essay=0.20
        ),
        funding_coverage=FundingCoverage(
            covers_tuition=True, covers_living_expense=True,
            covers_airfare=True, covers_insurance=True, monthly_stipend=11_000.0
        ),
        career_track_preference=infer_career_from_fields(fields),
        requires_return_home_country=True,
        mission_statement=(
            "The Norwegian Quota Scheme and NORPART partnership programs support students "
            "from developing countries to pursue undergraduate and postgraduate education "
            "at Norwegian universities, building capacity for sustainable development globally."
        ),
        target_recipient_profile=(
            "We seek students from developing partner countries with a minimum GPA of 75/100 "
            "and IELTS 6.0, who demonstrate commitment to returning home to contribute to "
            "national development after completing their Norwegian university education."
        ),
    )]


# ──────────────────────────────────────────────
# Registry of all scholarship data functions
# ──────────────────────────────────────────────

ALL_SCRAPERS: list[tuple[str, Callable[[], list[Scholarship]]]] = [
    ("Beasiswa Indonesia Maju (BIM)", scrape_bim),
    ("MEXT Undergraduate (Japan)", scrape_mext),
    ("Korean GKS Undergraduate", scrape_gks),
    ("ASEAN Scholarship Singapore", scrape_asean_scholarship),
    ("Stipendium Hungaricum", scrape_stipendium_hungaricum),
    ("Türkiye Bursları", scrape_turkiye_burslari),
    ("Chinese Government Scholarship (CSC)", scrape_csc),
    ("Russian Government Scholarship", scrape_russian_government),
    ("UWC Scholarship", scrape_uwc),
    ("AFS Intercultural Programs", scrape_afs),
    ("NUS/NTU Scholarships", scrape_nus_ntu),
    ("Taiwan ICDF", scrape_taiwan_icdf),
    ("JPA Malaysia", scrape_jpa_malaysia),
    ("Australia Awards Undergraduate", scrape_australia_awards_undergraduate),
    ("Brunei Darussalam Government Scholarship", scrape_brunei_darussalam_award),
    ("Thai Government Scholarship (OCSC)", scrape_thai_ocsc),
    ("New Zealand ASEAN Scholarship", scrape_nz_asean_scholarship),
    ("Taiwan MOE Scholarship", scrape_moe_taiwan),
    ("Shanghai Government Scholarship", scrape_shanghai_government),
    ("Beijing Government Scholarship", scrape_beijing_government),
    ("KAIST Global Scholarship", scrape_kaist_global),
    ("Yonsei University International Scholarship", scrape_yonsei_international),
    ("Panasonic Scholarship Japan", scrape_panasonic_scholarship),
    ("Kyoto University ASEAN Scholarship", scrape_kyoto_asean),
    ("Orange Tulip Scholarship (Netherlands)", scrape_orange_tulip),
    ("Holland Scholarship", scrape_holland_scholarship),
    ("Czech Government Scholarship (DZS)", scrape_czech_government),
    ("Polish Government Scholarship (NAWA)", scrape_nawa_poland),
    ("ICCR Scholarship (India)", scrape_iccr_india),
    ("IsDB Merit Scholarship", scrape_isdb_merit),
    ("Moroccan Government Scholarship (AMCI)", scrape_amci_morocco),
    ("Al-Azhar University Scholarship", scrape_al_azhar),
    ("Seoul National University International Scholarship", scrape_snu_international),
    ("POSTECH Global Scholarship", scrape_postech_global),
    ("Sungkyunkwan University (SKKU) Global Scholarship", scrape_skku_global),
    ("Korea University Global Excellence Scholarship", scrape_korea_university_global),
    ("Waseda University International Student Scholarship", scrape_waseda_international),
    ("Osaka University Global Scholars Program", scrape_osaka_global_scholars),
    ("Fudan University International Scholarship", scrape_fudan_international),
    ("French Government Scholarship", scrape_campus_france),
    ("Norwegian Quota Scheme / NORPART", scrape_norway_quota),
]

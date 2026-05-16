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
]

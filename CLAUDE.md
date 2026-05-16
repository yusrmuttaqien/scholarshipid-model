# scholarshipid-model — Panduan Proyek untuk Claude

## Ringkasan Proyek

Membangun dataset sintetis + real untuk melatih **Two-Tower Recommendation System** yang mencocokkan profil pelajar SMA dengan beasiswa luar negeri. Model menghasilkan continuous relevance score (0.0–1.0).

---

## Struktur Project

```
scholarshipid-model/
├── src/
│   ├── schemas/
│   │   ├── __init__.py          # Re-export semua dataclasses & enums
│   │   ├── student.py           # Student dataclass + enums terkait
│   │   ├── scholarship.py       # Scholarship dataclass + enums terkait
│   │   ├── pair.py              # Pair dataclass
│   │   └── feedback.py          # Feedback dataclass
│   └── scrapers/
│       ├── __init__.py
│       ├── base_scraper.py      # retry_request(), GeminiTextGenerator, enum maps
│       └── scholarship_scraper.py  # Scraper per sumber beasiswa
├── generator_two_tower.py       # Entry: baca scholarships.csv + generate student/pair/feedback
├── scrape_scholarships.py       # Entry: jalankan scraper → datasets_two_tower/scholarships.csv
├── datasets_two_tower/
│   ├── scholarships.csv         # Output scraping (data real)
│   ├── students.csv             # Output generator (sintetis)
│   ├── pairs.csv                # Output generator (computed relevance)
│   └── feedback.csv             # Output generator (implicit feedback)
├── CLAUDE.md                    # File ini
├── README.md                    # Spesifikasi teknis lengkap
└── POINTS_TO_REMEMBER.md        # Catatan penting pengembangan
```

---

## Versi Kode

### Versi 0 (`generator_two_tower.py` sebelum refactor)
- Semua data (student, scholarship, pair) **penuh sintetis**
- Satu file monolitik
- Digunakan sebagai baseline / referensi

### Versi 1 (target saat ini)
- Data **beasiswa = real** (scraped dari website resmi)
- Data student = sintetis dengan korelasi lebih realistis
- Kode direfactor ke `src/schemas/` + scraper terpisah
- Text fields (`mission_statement`, `target_recipient_profile`) di-generate via **Gemini API**

---

## Rencana Pengerjaan v1

### Grup 1: Data Beasiswa

**Phase 1 – Identifikasi beasiswa target**

Beasiswa SMA Indonesia untuk program S1 luar negeri:

| No | Nama | Negara | Website |
|----|------|--------|---------|
| 1 | Beasiswa Indonesia Maju (BIM) | Berbagai | bim.kemdikbud.go.id |
| 2 | MEXT Undergraduate | Jepang | mext.go.jp |
| 3 | Korean Government Scholarship (GKS) | Korea | studyinkorea.go.kr |
| 4 | ASEAN Scholarship | Singapura | moe.gov.sg |
| 5 | Stipendium Hungaricum | Hungaria | stipendiumhungaricum.hu |
| 6 | Türkiye Bursları | Turki | turkiyeburslari.gov.tr |
| 7 | Chinese Government Scholarship (CSC) | China | campuschina.org |
| 8 | Russian Government Scholarship | Rusia | russia.study |
| 9 | UWC Scholarship | International | uwc.org |
| 10 | AFS Intercultural Programs | Berbagai | afs.org |
| 11 | NUS/NTU Undergraduate Scholarship | Singapura | nus.edu.sg / ntu.edu.sg |
| 12 | Taiwan ICDF Scholarship | Taiwan | icdf.org.tw |
| 13 | Beasiswa Pemerintah Malaysia (JPA) | Malaysia | jpa.gov.my |

> Jika real scholarship < 800 → pakai apa adanya (tidak ada supplement sintetis).

**Phase 2 – Implementasi scraping & normalisasi**

- Library: `requests` + `BeautifulSoup4`
- Tiap sumber punya dedicated scraper function
- Field yang tidak tersedia di web: gunakan default / heuristic / Gemini API
- Field teks (`mission_statement`, `target_recipient_profile`): **Gemini API** dengan retry + RPM limiter

**Phase 3 – Test & run**

- Validasi semua field dan enum
- Verifikasi output CSV bisa dibaca generator

---

### Grup 2: Data Student

**Phase 1 – Tulis ulang student generator**

Perbaikan dari versi 0:
- Korelasi realistis: `school_tier` → skor akademis, `high_school_track` → `olympiad_subjects`
- Distribusi nationality: ~40–50% Indonesia, sisanya Asia Tenggara & negara lain
- Language proficiency terkait dengan `english_score`
- Text field lebih bervariasi

**Phase 2 – Test & run**

---

### Grup 3: Data Pair

**Phase 1 – Tulis ulang pair generator**

- `compute_relevance_score()`: pertahankan logika versi 0
- `generate_balanced_pairs()`: sesuaikan untuk jumlah scholarship real
- Pastikan ratio Match:In-Between:Not Match = 1:1:1

**Phase 2 – Test & run**

---

## Keputusan Teknis Penting

| Keputusan | Pilihan | Alasan |
|-----------|---------|--------|
| Jumlah beasiswa | Pakai apa adanya (tidak supplement sintetis) | Data real lebih valuable walau lebih sedikit |
| Text fields | Gemini API + fallback template | Teks lebih natural daripada template statis |
| RPM Gemini | Dikonfirmasi user sebelum Phase 2 | Bergantung pada quota API |
| Arsitektur file | `src/schemas/` + scraper terpisah | Separation of concerns: scraping ≠ generation |

---

## Skema Dataset

Lengkap di `README.md`. Ringkasan:

- **Student**: 20,000 profil pelajar SMA (sintetis), usia 16–18, target S1 luar negeri
- **Scholarship**: N beasiswa real (hasil scraping)
- **Pair**: ~250,000+ pasang (student, scholarship) dengan `relevance_score` 0.0–1.0
- **Feedback**: implicit feedback (apply/click/view/reject)

### Relevance Score

| Kategori | Range | Label |
|----------|-------|-------|
| Match | ≥ 0.7 | Strong alignment |
| In-Between | 0.3–0.7 | Partial alignment |
| Not Match | < 0.3 | No alignment |

Target distribusi pair: **1:1:1** (Match : In-Between : Not Match)

---

## Cara Menjalankan

```bash
# 1. Scrape beasiswa (butuh internet)
python scrape_scholarships.py

# 2. Generate dataset lengkap (baca scholarships.csv hasil scraping)
python generator_two_tower.py
```

Output tersimpan di `datasets_two_tower/`.

---

## Catatan untuk Claude

- Gunakan **Gemini API** (`google-generativeai`), bukan Claude API, untuk semua LLM enrichment
- Implementasi Gemini harus include: retry dengan exponential backoff + RPM rate limiter + fallback template
- `src/schemas/` hanya berisi dataclasses & enums, tidak ada logika bisnis
- Scraper harus handle `requests.exceptions.RequestException` dan skip jika gagal (return `None`)
- Setelah normalisasi, validasi semua enum values sebelum masuk CSV

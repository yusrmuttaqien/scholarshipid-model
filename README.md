# ScholarshipID — Two-Tower Recommendation Model

Sistem rekomendasi beasiswa menggunakan arsitektur **Two-Tower (Dual Encoder)** untuk mencocokkan profil siswa SMA dengan beasiswa S1 luar negeri, menghasilkan top-5 beasiswa paling relevan per siswa.

## Arsitektur

```
Student Tower                     Scholarship Tower
  Input(506)                         Input(509)
  Dense(256, relu)                   Dense(256, relu)
  Dense(128, relu)                   Dense(128, relu)
  L2Normalize                        L2Normalize
      │                                   │
      └──────── Dot Product ──────────────┘
                     │
               Top-5 Ranking
```

- **Student Tower**: concat(structured_features=122, text_emb=384) → 128-dim L2-normalized embedding
- **Scholarship Tower**: concat(structured_features=125, text_emb=384) → 128-dim L2-normalized embedding
- **Text Encoder**: Sentence-BERT `all-MiniLM-L6-v2` (384-dim, frozen, pre-computed)
- **Retrieval**: Brute-force dot product vs semua 44 scholarship
- **Loss**: Sampled softmax + in-batch negatives, temperature=0.1, sample weighting (accepted=5×, apply=2×, click=1×)
- **Metrics**: Recall@5, NDCG@5, MRR

## Struktur Folder

```
├── configs/
│   ├── default.yaml             # Hyperparameter & paths
│   └── serving.yaml             # Serving configuration (environment, models)
├── data/
│   ├── raw/                     # students.csv, scholarships.csv, feedback.csv
│   ├── processed/
│   └── features/
│       └── text_embeddings/     # Cache SBERT embeddings (.npy)
├── notebooks/
│   └── notebook_two_tower.ipynb # Referensi implementasi (TF/Keras)
├── outputs/
│   ├── checkpoints/             # student_tower_best.weights.h5, scholarship_tower_best.weights.h5
│   ├── embeddings/              # scholarship_emb.npy, scholarship_ids.npy
│   └── logs/
├── scripts/
│   ├── precompute_text_embeddings.py  # Step 1: cache SBERT
│   ├── train.py                       # Step 2: training
│   ├── evaluate.py                    # Step 3: evaluasi test set
│   ├── export_embeddings.py           # Step 4: export untuk serving
│   └── serve.py                       # Start FastAPI inference server
└── src/
    ├── models/
    │   ├── student_tower.py
    │   ├── scholarship_tower.py
    │   └── two_tower.py
    ├── serving/
    │   ├── inference_engine.py        # Inference engine (encode, retrieve)
    │   └── api.py                     # FastAPI endpoints
    ├── trainers/trainer.py
    ├── evaluators/evaluator.py
    ├── utils/
    │   ├── feature_engineering.py
    │   └── data_loader.py
```

## Setup

> **Windows**: pastikan [Microsoft Visual C++ Redistributable 2019](https://aka.ms/vs/17/release/vc_redist.x64.exe) sudah terinstall.

```bash
# Pastikan python di sini adalah Python sistem (bukan conda base). Minimal versi 3.11
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

## Quick Start

```bash
# Step 1 — Pre-compute text embeddings (sekali saja, ~5-10 menit)
python scripts/precompute_text_embeddings.py # or python -m scripts.precompute_text_embeddings

# Step 2 — Train model
python scripts/train.py --config configs/default.yaml # or python -m scripts.train --config configs/default.yaml

# Step 3 — Evaluasi pada test set
python scripts/evaluate.py \  # or python -m scripts.evaluate
  --config configs/default.yaml \
  --student_checkpoint outputs/checkpoints/student_tower_best.weights.h5 \
  --scholarship_checkpoint outputs/checkpoints/scholarship_tower_best.weights.h5

# Step 4 — Export scholarship embeddings untuk serving
python scripts/export_embeddings.py \  # or python -m scripts.export_embeddings
  --scholarship_checkpoint outputs/checkpoints/scholarship_tower_best.weights.h5
```

## Data

| File | Rows | Keterangan |
|---|---|---|
| `students.csv` | 20.000 | Profil siswa SMA |
| `scholarships.csv` | 43 | Beasiswa S1 luar negeri |
| `feedback.csv` | 100.000 | Interaksi: click / apply / accepted |

## Serving (FastAPI)

After training, start the inference server:

```bash
# Start the serving server
python scripts/serve.py
```

Server runs on `http://localhost:8001` with the following endpoints:

### GET `/docs` — Swagger docs

### GET `/health` — Health check

### Configuration

Edit `configs/serving.yaml` to configure:
- **Model paths**: Student & scholarship tower checkpoint locations
- **Data source**: CSV path for scholarship refresh
- **Server settings**: Host, port, CORS origins
- **Environment**: `local` (development) vs `production` modes

## Performance (test set)

| Metric | Score |
|---|---|
| Recall@5 | ~0.32 |
| NDCG@5 | ~0.22 |
| MRR | ~0.21 |

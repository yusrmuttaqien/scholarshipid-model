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
│   └── default.yaml             # All config: hyperparameters, model checkpoints, server settings
├── data/
│   ├── raw/                     # students.csv, scholarships.csv, feedback.csv
│   ├── processed/
│   └── features/
│       └── text_embeddings/     # Cache SBERT embeddings (.npy)
├── notebooks/
│   └── notebook_two_tower.ipynb # Referensi implementasi (TF/Keras)
├── outputs/
│   ├── checkpoints/             # student_tower_best.keras, scholarship_tower_best.keras
│   ├── embeddings/              # scholarship_emb.npy, scholarship_ids.npy
│   └── logs/                    # TensorBoard logs (tb_{experiment_name}/)
├── scripts/
│   ├── hf_sync.py               # HuggingFace artifact sync (pull/push)
│   ├── precompute_text_embeddings.py  # Step 1: cache SBERT
│   ├── train.py                        # Step 2: training
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
python -m venv venv # or uv venv venv -p 3.11

# Windows
.\venv\Scripts\Activate.ps1
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt # or use yusr-requirements.txt for CPU only compute
pip install -e .

# If you using yusr-requirements.txt to install the packages, install this too
pip install torch==2.2.2+cpu --index-url https://download.pytorch.org/whl/cpu

# Optionally, if Tensorboard failing to launch
pip install 'setuptools<75'
```

### HuggingFace Setup (Artifact Sync)

To sync model artifacts and data with HuggingFace:

```bash
# 1. Copy example env file and fill in your token
cp .env.example .env
# Edit .env with your HF_TOKEN from https://huggingface.co/settings/tokens
```

## Quick Start

```bash
# Step 1 — Pre-compute text embeddings (sekali saja, ~5-10 menit)
python scripts/precompute_text_embeddings.py # or python -m scripts.precompute_text_embeddings

# Step 2 — Train model
python scripts/train.py --config configs/default.yaml # or python -m scripts.train --config configs/default.yaml

# Step 3 — Evaluasi pada test set (checkpoint paths default to configs/default.yaml)
python scripts/evaluate.py \
  --config configs/default.yaml

# Override checkpoint paths if needed:
python scripts/evaluate.py \
  --config configs/default.yaml \
  --student_checkpoint outputs/checkpoints/student_tower_best.keras \
  --scholarship_checkpoint outputs/checkpoints/scholarship_tower_best.keras

# Step 4 — Export scholarship embeddings untuk serving (checkpoint path defaults to config)
python scripts/export_embeddings.py \
  --config configs/default.yaml

# Override checkpoint path if needed:
python scripts/export_embeddings.py \
  --scholarship_checkpoint outputs/checkpoints/scholarship_tower_best.keras
```

## HuggingFace Artifact Sync

Two separate repos are used for syncing:

| Repo | Contents | Type |
|---|---|---|
| `ydmhmhm/scholarshipid-data` | `data/raw/`, `outputs/logs/` | Dataset |
| `ydmhmhm/scholarshipid-model` | `checkpoints/`, `embeddings/` | Model |

**CLI commands:**

```bash
# Pull data + model from HuggingFace (before starting serving)
python scripts/hf_sync.py pull-data --config configs/default.yaml
python scripts/hf_sync.py pull-model --config configs/default.yaml

# Push data + model to HuggingFace (after retraining/refreshing)
python scripts/hf_sync.py push-data --config configs/default.yaml --message "New data"
python scripts/hf_sync.py push-model --config configs/default.yaml --message "Retrained"
```

**Auto-integration:**
- `scripts/serve.py` — pulls both repos before FastAPI starts
- `scripts/retrain.py` — pushes both repos after retraining
- `src/serving/api.py` `/retrain` endpoint — pushes both repos (data + model) on API retrain
- `src/serving/api.py` `/refresh` endpoint — pushes data only after refreshing scholarship cache

## Docker Deployment (HuggingFace Spaces)

The project includes a Dockerfile for deploying on HuggingFace Spaces with Docker runtime.

```bash
# Build locally
docker build -t scholarshipid-model .

# Run locally (sets SERVER_PORT=7860 to match HF Spaces)
docker run -p 7860:7860 \
  --name scholarship-id
  -e HF_TOKEN=your_token_here \
  -e SERVER_PORT=7860 \
  scholarshipid-model
```

**How it works:**
1. Container starts → `serve.py` runs automatically
2. Pulls models/data from HuggingFace repos (configured in `.env`)
3. Starts FastAPI on port 7860

**Deploy to HF Spaces:**
1. Push your repo to GitHub
2. Create a new Space → select **Docker** runtime
3. Connect your GitHub repo and give it a name like `scholarshipid-api`
4. Set `HF_TOKEN` as a secret in the Space settings (Settings → Secrets and variables → Actions)
5. The API will be live at `https://your-space-name.hf.space`

## Data

| File | Rows | Keterangan |
|---|---|---|
| `students.csv` | 20.000 | Profil siswa SMA |
| `scholarships.csv` | 43 | Beasiswa S1 luar negeri |
| `feedback.csv` | 100.000 | Interaksi: click / apply / accepted |

## Monitoring (TensorBoard)

TensorBoard logs are written to `outputs/logs/tb_{experiment_name}/`.

```bash
tensorboard --logdir outputs/logs/ --bind_all
```

## Serving (FastAPI)

After training, start the inference server:

```bash
# Start the serving server
python scripts/serve.py # or python -m scripts.serve
```

Server runs on `http://localhost:<PORT_DEFINED_AT_CONFIG>` with the following endpoints:

### GET `/docs` — Swagger docs

### Configuration

All configuration is in `configs/default.yaml`:
- **Model checkpoints**: `models.student_tower`, `models.scholarship_tower`
- **Server settings**: `server.host`, `server.port`, `server.cors_origins`
- **Auth**: `server.auth_required`, `server.auth_token` (set for production)
- **Retraining**: `retraining.holdout_fraction` (0.0 = use all data)

## Performance (test set)

| Metric | Score |
|---|---|
| Recall@5 | ~0.32 |
| NDCG@5 | ~0.22 |
| MRR | ~0.21 |
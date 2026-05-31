"""HuggingFace artifact synchronization.

Syncs two separate HuggingFace repos:
  - Data repo : data/raw/ and outputs/logs/
  - Model repo: outputs/checkpoints/ and outputs/embeddings/

Usage:
    python scripts/hf_sync.py pull-data --config configs/default.yaml
    python scripts/hf_sync.py push-data --config configs/default.yaml --message "New data"
    python scripts/hf_sync.py pull-model --config configs/default.yaml
    python scripts/hf_sync.py push-model --config configs/default.yaml --message "Retrained"

The script automatically loads .env from the project root if present.
"""
import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Auto-load .env from project root (relative to this script's parent dir)
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path, override=True)
except ImportError:
    pass  # python-dotenv not installed, skip

import yaml
from huggingface_hub import HfApi, snapshot_download


def load_config(config_path="configs/default.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_hf_credentials(config):
    """Get HuggingFace credentials from config or environment.

    Priority:
    1. HF_TOKEN env var (always checked first)
    2. hf.token in config
    """
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    hf_config = config.get("hf", {})
    token_env = hf_config.get("token_env", "HF_TOKEN")
    token = os.environ.get(token_env) or hf_config.get("token")
    return token


def ensure_login(config):
    """Ensure user is authenticated with HuggingFace."""
    token = get_hf_credentials(config)
    if not token:
        print("Error: No HuggingFace token found.")
        print("Set HF_TOKEN environment variable or add 'hf.token' to config.")
        sys.exit(1)

    api = HfApi()
    try:
        user = api.whoami(token=token)
        print(f"Authenticated as {user['name']}")
    except Exception as e:
        print(f"Authentication failed: {e}")
        sys.exit(1)
    return token


# ── Data repo sync (data/raw/, logs/) ───────────────────────────────────

def pull_data_artifacts(config_path="configs/default.yaml", repo_id=None):
    """Pull data and logs from the data repo.

    Downloads to:
        <raw_path>          (from config)
        outputs/logs/
    """
    config = load_config(config_path)
    token = ensure_login(config)

    hf_config = config.get("hf", {})
    repo_id = repo_id or hf_config.get("repo_id_data")
    if not repo_id:
        print("Error: No 'repo_id_data' specified in config.")
        sys.exit(1)

    data_config = config.get("data", {})
    raw_path = Path(data_config.get("raw_path", "data/raw/"))

    log_dir = Path(config.get("output", {}).get("log_dir", "outputs/logs/"))

    print(f"Pulling data from {repo_id} ...")
    print(f"  Raw data:  {raw_path}")
    print(f"  Logs:      {log_dir}")

    download_dir = raw_path.parent / ".hf_temp_download_data"
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=download_dir,
            token=token,
            allow_patterns=["raw/**", "logs/**"],
        )

        # Copy raw data
        src_raw = download_dir / "raw"
        if src_raw.exists():
            raw_path.mkdir(parents=True, exist_ok=True)
            for f in src_raw.rglob("*"):
                if f.is_file() and f.name != ".gitkeep":
                    dest = raw_path / f.relative_to(src_raw)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(f.read_bytes())
                    print(f"  ✓ {dest.relative_to(raw_path)}")

        # Copy logs (skip .gitkeep)
        src_logs = download_dir / "logs"
        if src_logs.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
            for f in src_logs.rglob("*"):
                if f.is_file() and f.name != ".gitkeep":
                    dest = log_dir / f.relative_to(src_logs)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(f.read_bytes())
                    print(f"  ✓ {dest.relative_to(log_dir)}")

    finally:
        if download_dir.exists():
            shutil.rmtree(download_dir, ignore_errors=True)

    print("Data pull complete!")


def push_data_artifacts(config_path="configs/default.yaml", message=None, repo_id=None):
    """Push data and logs to the data repo."""
    config = load_config(config_path)
    token = ensure_login(config)

    hf_config = config.get("hf", {})
    repo_id = repo_id or hf_config.get("repo_id_data")
    if not repo_id:
        print("Error: No 'repo_id_data' specified in config.")
        sys.exit(1)

    message = message or f"Auto-push data at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    api = HfApi()
    token = get_hf_credentials(config)

    data_config = config.get("data", {})
    raw_path = Path(data_config.get("raw_path", "data/raw/"))
    log_dir = Path(config.get("output", {}).get("log_dir", "outputs/logs/"))

    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", token=token, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create dataset repo (may already exist): {e}")

    # Build staging dir with subfolder structure for upload
    staging = Path(config_path).parent / ".hf_staging_data"
    try:
        staging.mkdir(exist_ok=True)
        for dir_path, subfolder in [(raw_path, "raw"), (log_dir, "logs")]:
            src_sub = staging / subfolder
            if dir_path.exists() and any(dir_path.rglob("*")):
                print(f"Uploading {subfolder}/ ...")
                shutil.copytree(str(dir_path), str(src_sub), dirs_exist_ok=True)
            else:
                print(f"Skipping {subfolder}/ (empty or not found)")

        api.upload_folder(
            folder_path=str(staging),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=message,
            token=token,
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    print(f"\nPushed data to {repo_id} — {message}")


# ── Model repo sync (checkpoints, embeddings) ───────────────────────────

def pull_model_artifacts(config_path="configs/default.yaml", repo_id=None):
    """Pull checkpoints and embeddings from the model repo."""
    config = load_config(config_path)
    token = ensure_login(config)

    hf_config = config.get("hf", {})
    repo_id = repo_id or hf_config.get("repo_id_model")
    if not repo_id:
        print("Error: No 'repo_id_model' specified in config.")
        sys.exit(1)

    output_config = config.get("output", {})
    checkpoint_dir = Path(output_config.get("checkpoint_dir", "outputs/checkpoints/"))
    embedding_dir = Path(output_config.get("embedding_dir", "outputs/embeddings/"))

    print(f"Pulling model artifacts from {repo_id} ...")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  Embeddings:  {embedding_dir}")

    download_dir = checkpoint_dir.parent / ".hf_temp_download_model"
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=download_dir,
            token=token,
            allow_patterns=[
                "checkpoints/**",
                "embeddings/**",
            ],
        )

        # Copy checkpoints
        src_checkpoints = download_dir / "checkpoints"
        if src_checkpoints.exists():
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            for f in src_checkpoints.rglob("*"):
                if f.is_file():
                    dest = checkpoint_dir / f.relative_to(src_checkpoints)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(f.read_bytes())
                    print(f"  ✓ {dest.relative_to(checkpoint_dir)}")

        # Copy embeddings (skip .gitkeep)
        src_embeddings = download_dir / "embeddings"
        if src_embeddings.exists():
            embedding_dir.mkdir(parents=True, exist_ok=True)
            for f in src_embeddings.rglob("*"):
                if f.is_file() and f.name not in (".gitkeep",):
                    dest = embedding_dir / f.relative_to(src_embeddings)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(f.read_bytes())
                    print(f"  ✓ {dest.relative_to(embedding_dir)}")

    finally:
        if download_dir.exists():
            shutil.rmtree(download_dir, ignore_errors=True)

    print("Model pull complete!")


def push_model_artifacts(config_path="configs/default.yaml", message=None, repo_id=None):
    """Push checkpoints and embeddings to the model repo."""
    config = load_config(config_path)
    token = ensure_login(config)

    hf_config = config.get("hf", {})
    repo_id = repo_id or hf_config.get("repo_id_model")
    if not repo_id:
        print("Error: No 'repo_id_model' specified in config.")
        sys.exit(1)

    message = message or f"Auto-push model at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    api = HfApi()
    token = get_hf_credentials(config)

    output_config = config.get("output", {})
    checkpoint_dir = Path(output_config.get("checkpoint_dir", "outputs/checkpoints/"))
    embedding_dir = Path(output_config.get("embedding_dir", "outputs/embeddings/"))

    try:
        api.create_repo(repo_id=repo_id, repo_type="model", token=token, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create repo (may already exist): {e}")

    # Build staging dir with subfolder structure for upload
    staging = Path(config_path).parent / ".hf_staging_model"
    try:
        staging.mkdir(exist_ok=True)
        for dir_path, subfolder in [
            (checkpoint_dir, "checkpoints"),
            (embedding_dir, "embeddings"),
        ]:
            src_sub = staging / subfolder
            if dir_path.exists() and any(dir_path.rglob("*")):
                print(f"Uploading {subfolder}/ ...")
                shutil.copytree(str(dir_path), str(src_sub), dirs_exist_ok=True)
            else:
                print(f"Skipping {subfolder}/ (empty or not found)")

        api.upload_folder(
            folder_path=str(staging),
            repo_id=repo_id,
            repo_type="model",
            commit_message=message,
            token=token,
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    print(f"\nPushed model to {repo_id} — {message}")


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HuggingFace artifact sync")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Pull data
    dp = subparsers.add_parser("pull-data", help="Pull data from data repo")
    dp.add_argument("--config", default="configs/default.yaml")
    dp.add_argument("--repo-id", default=None, help="Override data repo ID")

    # Push data
    dpu = subparsers.add_parser("push-data", help="Push data to data repo")
    dpu.add_argument("--config", default="configs/default.yaml")
    dpu.add_argument("--message", default=None, help="Commit message")
    dpu.add_argument("--repo-id", default=None, help="Override data repo ID")

    # Pull model
    mp = subparsers.add_parser("pull-model", help="Pull model from model repo")
    mp.add_argument("--config", default="configs/default.yaml")
    mp.add_argument("--repo-id", default=None, help="Override model repo ID")

    # Push model
    mpu = subparsers.add_parser("push-model", help="Push model to model repo")
    mpu.add_argument("--config", default="configs/default.yaml")
    mpu.add_argument("--message", default=None, help="Commit message")
    mpu.add_argument("--repo-id", default=None, help="Override model repo ID")

    args = parser.parse_args()

    commands = {
        "pull-data": lambda: pull_data_artifacts(config_path=args.config, repo_id=args.repo_id),
        "push-data": lambda: push_data_artifacts(config_path=args.config, message=args.message, repo_id=args.repo_id),
        "pull-model": lambda: pull_model_artifacts(config_path=args.config, repo_id=args.repo_id),
        "push-model": lambda: push_model_artifacts(config_path=args.config, message=args.message, repo_id=args.repo_id),
    }

    if args.command in commands:
        commands[args.command]()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Log completed SciFact evaluation results to remote DagsHub MLflow."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import mlflow
from dotenv import load_dotenv


load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["baseline", "agent"],
    required=True,
)
args = parser.parse_args()

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
username = os.getenv("DAGSHUB_USER")
token = os.getenv("DAGSHUB_TOKEN")

if not tracking_uri or not tracking_uri.startswith("https://"):
    raise RuntimeError("Set an HTTPS MLFLOW_TRACKING_URI in .env")

if not username or not token:
    raise RuntimeError("DAGSHUB_USER and DAGSHUB_TOKEN are required")

os.environ["MLFLOW_TRACKING_USERNAME"] = username
os.environ["MLFLOW_TRACKING_PASSWORD"] = token

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(
    os.getenv("MLFLOW_EXPERIMENT_NAME", "scientific-claim-verification")
)

metrics_file = Path(f"reports/{args.mode}_end_to_end_metrics.json")
predictions_file = Path(f"reports/{args.mode}_predictions.json")
config_file = Path("configs/verification.yaml")

for path in (metrics_file, predictions_file, config_file):
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact missing: {path}")

metrics = json.loads(metrics_file.read_text(encoding="utf-8"))

if metrics.get("examples") != 188:
    raise ValueError(
        f"Expected 188 evaluated claims; got {metrics.get('examples')}"
    )

retrieval_method = "bm25" if args.mode == "baseline" else "hybrid"
workflow = "baseline" if args.mode == "baseline" else "langgraph"

try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
except (subprocess.CalledProcessError, FileNotFoundError):
    git_commit = "unknown"

try:
    dvc_lock = subprocess.check_output(
        ["git", "hash-object", "dvc.lock"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
except (subprocess.CalledProcessError, FileNotFoundError):
    dvc_lock = "unknown"

params = {
    "system": args.mode,
    "workflow": workflow,
    "retrieval_method": retrieval_method,
    "nli_model": "cross-encoder/nli-deberta-v3-small",
    "split": "dev",
    "examples": metrics["examples"],
    "git_commit": git_commit,
    "dvc_lock_git_blob": dvc_lock,
}

numeric_metrics = {
    key: float(value)
    for key, value in metrics.items()
    if isinstance(value, (int, float)) and not isinstance(value, bool)
}

with mlflow.start_run(run_name=f"{args.mode}-dev-188") as run:
    mlflow.log_params(params)
    mlflow.log_metrics(numeric_metrics)

    mlflow.log_artifact(str(metrics_file), artifact_path="evaluation")
    mlflow.log_artifact(str(predictions_file), artifact_path="predictions")
    mlflow.log_artifact(str(config_file), artifact_path="configuration")

    print("Successfully logged MLflow run")
    print("Mode:", args.mode)
    print("Run ID:", run.info.run_id)
    print("Tracking URI:", tracking_uri)

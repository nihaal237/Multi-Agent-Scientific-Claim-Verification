"""Register the complete SciFact verification pipeline as an MLflow pyfunc model.
Run from repository root: python scripts/register_pipeline.py
"""
import json
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
import yaml
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from mlflow.tracking import MlflowClient

ROOT = Path.cwd().resolve()
MODEL_NAME = "ScientificClaimVerifier"

@contextmanager
def in_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)

class ScientificClaimPipeline(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # The existing project uses repository-relative paths. Recreate that
        # layout in a private, writable directory for the local Qdrant client.
        self._workspace = tempfile.TemporaryDirectory(prefix="scifact_pyfunc_")
        self.root = Path(self._workspace.name)
        (self.root / "data/processed").mkdir(parents=True)
        (self.root / "artifacts/index").mkdir(parents=True)
        (self.root / "configs").mkdir(parents=True)
        shutil.copy2(context.artifacts["documents"], self.root / "data/processed/documents.jsonl")
        shutil.copytree(context.artifacts["qdrant"], self.root / "artifacts/index/qdrant")
        for name in ("verification", "retrieval", "hybrid"):
            shutil.copy2(context.artifacts[f"{name}_config"], self.root / f"configs/{name}.yaml")
        # Replace model identifiers with bundled, versioned model snapshots.
        for cfg_name, key, artifact_key in (
            ("verification", "nli_model", "nli_model"),
            ("retrieval", "embedding_model", "embedding_model"),
            ("hybrid", "cross_encoder_model", "reranker_model"),
        ):
            cfg_path = self.root / f"configs/{cfg_name}.yaml"
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            cfg[key] = context.artifacts[artifact_key]
            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        import sys

        code_root = Path(__file__).resolve().parent.parent

        if str(code_root) not in sys.path:
            sys.path.insert(0, str(code_root))   
        from src.agents.workflow import Verifier
        with in_directory(self.root):
            self.verifier = Verifier(mode="agent")

    def close(self):
        if getattr(self, "verifier", None) is not None:
            # Release the Qdrant database lock before removing files.
            retriever = getattr(self.verifier, "retriever", None)

            if retriever is not None:
                qdrant = getattr(retriever, "qdrant", None)
                if qdrant is not None:
                    qdrant.close()

        if getattr(self, "_workspace", None) is not None:
            self._workspace.cleanup()

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            if "claim" not in model_input.columns:
                raise ValueError("Input DataFrame must contain a 'claim' column")
            claims = model_input["claim"].tolist()
        elif isinstance(model_input, dict) and "claim" in model_input:
            value = model_input["claim"]
            claims = value if isinstance(value, list) else [value]
        else:
            raise ValueError("Provide a DataFrame or dictionary containing 'claim'")
        rows = []
        with in_directory(self.root):
            for claim in claims:
                if not isinstance(claim, str) or not claim.strip():
                    raise ValueError("Each claim must be a nonempty string")
                result = self.verifier.run(claim)
                rows.append({
                    "claim": claim,
                    "verdict": result["verdict"],
                    "confidence": float(result["confidence"]),
                    "paper_ids": json.dumps(result["paper_ids"]),
                    "evidence_sentences": json.dumps(result["evidence"]),
                    "report_json": json.dumps(result),
                })
        return pd.DataFrame(rows)


def main():
    load_dotenv(ROOT / ".env")
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    user = os.environ.get("DAGSHUB_USER")
    token = os.environ.get("DAGSHUB_TOKEN")
    if not uri.startswith("https://") or not user or not token:
        raise RuntimeError("Configure HTTPS MLFLOW_TRACKING_URI, DAGSHUB_USER and DAGSHUB_TOKEN in .env")
    os.environ["MLFLOW_TRACKING_USERNAME"] = user
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    for required in (
        "src/agents/workflow.py", "src/retrieval/hybrid.py",
        "data/processed/documents.jsonl", "artifacts/index/qdrant",
        "configs/verification.yaml", "configs/hybrid.yaml", "configs/retrieval.yaml",
    ):
        if not (ROOT / required).exists():
            raise FileNotFoundError(f"Missing: {required}. Run dvc pull first.")

    cfgs = {name: yaml.safe_load((ROOT / f"configs/{name}.yaml").read_text(encoding="utf-8"))
            for name in ("verification", "retrieval", "hybrid")}
    model_ids = {
        "nli_model": cfgs["verification"]["nli_model"],
        "embedding_model": cfgs["retrieval"]["embedding_model"],
        "reranker_model": cfgs["hybrid"]["cross_encoder_model"],
    }
    # Stage real copies of the required model files: HF snapshots may contain
    # symlinks/junctions that MLflow correctly rejects as escaping artifact roots.
    # A context manager keeps the staging files alive until upload completes.
    staging = tempfile.TemporaryDirectory(prefix="scifact_registry_bundle_")
    stage_root = Path(staging.name)
    snapshots = {}
    allowed = [
        "config.json", "config_sentence_transformers.json", "modules.json",
        "sentence_bert_config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "added_tokens.json", "vocab.txt",
        "merges.txt", "spm.model", "sentencepiece.bpe.model",
        "model.safetensors",
        "1_Pooling/config.json", "0_Transformer/config.json",
    ]
    try:
        print("Preparing minimal PyTorch model bundles...", flush=True)
        for key, model_id in model_ids.items():
            snapshot = Path(snapshot_download(
                repo_id=model_id,
                allow_patterns=allowed,
            ))
            destination = stage_root / "models" / key
            destination.mkdir(parents=True, exist_ok=True)
            for source in snapshot.rglob("*"):
                if not source.is_file():
                    continue

                relative = source.relative_to(snapshot)

                # Copy only explicitly permitted model files.
                if relative.as_posix() not in allowed:
                    continue

                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target, follow_symlinks=True)
            snapshots[key] = str(destination)
            print(f"  {key}: {model_id}", flush=True)

        # Preserve 'src' as a package at the root of MLflow's code path.
        code_root = stage_root / "code"
        shutil.copytree(ROOT / "src", code_root / "src",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        artifacts = {
            "documents": str(ROOT / "data/processed/documents.jsonl"),
            "qdrant": str(ROOT / "artifacts/index/qdrant"),
            "verification_config": str(ROOT / "configs/verification.yaml"),
            "retrieval_config": str(ROOT / "configs/retrieval.yaml"),
            "hybrid_config": str(ROOT / "configs/hybrid.yaml"),
            **snapshots,
        }
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "scientific-claim-verification"))
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        client = MlflowClient()
        with mlflow.start_run(run_name="register-complete-verification-pipeline") as run:
            mlflow.log_params({
                "pipeline": "hybrid-retrieval-evidence-nli-langgraph",
                "git_commit": commit,
                **model_ids,
            })
            # MLflow 3.x uses `name`; code_paths makes project imports available.
            info = mlflow.pyfunc.log_model(
                name="scientific_claim_pipeline",
                python_model=ScientificClaimPipeline(),
                artifacts=artifacts,
                code_paths=[str(code_root)],
                pip_requirements=[
                    "mlflow>=3,<4", "pandas>=2,<3", "PyYAML>=6,<7",
                    "torch>=2.2,<3", "transformers>=4.40,<5",
                    "sentence-transformers>=3,<6", "rank-bm25==0.2.2",
                    "qdrant-client>=1.12,<2", "langgraph>=0.2,<2",
                    "numpy>=1.26,<3", "sentencepiece>=0.2,<1",
                ],
                metadata={"pipeline_type": "full_scientific_claim_verification", "git_commit": commit},
            )
            print("Logged model:", info.model_uri, flush=True)
            registered = mlflow.register_model(model_uri=info.model_uri, name=MODEL_NAME)
            print("Registered version:", registered.version, flush=True)
            client.transition_model_version_stage(
                name=MODEL_NAME, version=registered.version, stage="Staging", archive_existing_versions=False
            )
            print("SUCCESS: ScientificClaimVerifier version", registered.version, "-> Staging")
            print("MLflow run:", run.info.run_id)
    finally:
        staging.cleanup()

if __name__ == "__main__":
    main()

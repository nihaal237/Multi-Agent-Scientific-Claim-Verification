# Multi-Agent Scientific Claim Verification

An MLOps-based scientific claim verification system that retrieves relevant research papers using keyword-based, semantic, and hybrid retrieval techniques.

The goal is to develop a multi-agent AI system capable of determining whether a scientific claim is **SUPPORTED**, **CONTRADICTED**, or has **INSUFFICIENT EVIDENCE**, with explanations grounded in scientific research papers.

The project uses the **SciFact dataset** and follows a reproducible ML pipeline using **DVC and DagsHub**.

> **Current Development Stage (13 November 2026):** Dataset ingestion, validation, preprocessing, vector indexing, and retrieval evaluation are complete. Multi-agent verification and MLflow integration remain planned.

---

## 1. Problem Overview

Scientific claims are often repeated without verifying whether published research supports the exact statement.

Manually searching scientific papers, identifying relevant evidence, and determining whether the evidence supports or contradicts a claim can be time-consuming.

This project aims to automate scientific claim verification through a multi-agent AI system.

The complete system will:

1. Accept a scientific claim from a user.
2. Search a collection of scientific research papers.
3. Retrieve the most relevant papers.
4. Identify evidence sentences within those papers.
5. Determine whether the evidence supports, contradicts, or is insufficient to verify the claim.
6. Generate an explanation with supporting paper citations.

The system evaluates claims against available evidence rather than establishing definitive scientific truth.

---

## 2. Dataset

We use the **SciFact dataset**, which contains scientific claims, research paper abstracts, and evidence annotations.

**Dataset:** https://huggingface.co/datasets/allenai/scifact

**Original Repository:** https://github.com/allenai/scifact

**Project Repository:** https://github.com/nihaal237/Multi-Agent-Scientific-Claim-Verification

### Dataset Statistics

| Component | Count |
|-----------|------:|
| Research paper abstracts | 5,183 |
| Training claims | 809 |
| Development claims | 300 |
| Test claims | 300 |
| Preprocessed sentence chunks | 45,952 |

Each research paper contains:

- Document ID
- Title
- Abstract (split into sentences)

Each labeled scientific claim contains:

- Claim ID
- Scientific claim text
- Evidence document IDs
- Evidence sentence indices
- SUPPORT or CONTRADICT labels

The official test split does not contain publicly available evidence labels.

### Data Ingestion

The dataset is downloaded automatically using:

```bash
python -m src.ingest
```

The ingestion script:
- Downloads SciFact from its original source.
- Extracts the required JSONL files.
- Records the source SHA-256 checksum.
- Ensures reproducible dataset ingestion.

### Data Validation

Run:

```bash
python -m src.validate
```

Validation checks:

- Required dataset files exist.
- Document and claim IDs are unique.
- Paper titles and abstracts are valid.
- Claims contain nonempty text.
- Evidence document references exist.
- Evidence sentence indices are valid.
- Evidence labels contain valid values.

**Validation result: PASS — 0 errors.**

---

## 3. Data Preprocessing

The SciFact corpus contains 5,183 research abstracts.

Each abstract is divided into individual sentence chunks.

Run:

```bash
python -m src.preprocess
```

### Preprocessing Output

| Output | Description |
|--------|-------------|
| `documents.jsonl` | Full paper titles, abstracts, and sentences |
| `chunks.jsonl` | Sentence-level chunks with metadata |
| `claims_train.jsonl` | Training claims and annotations |
| `claims_dev.jsonl` | Development claims and annotations |
| `claims_test.jsonl` | Test claims |

**Total generated chunks: 45,952**

Each chunk includes:

```json
{
  "chunk_id": "12345_2",
  "doc_id": "12345",
  "sentence_id": 2,
  "title": "Example Scientific Paper",
  "text": "Example evidence sentence.",
  "embedding_text": "Example Scientific Paper. Example evidence sentence."
}
```

Chunking enables sentence-level retrieval and evidence extraction.

---

## 4. Retrieval Systems

We implemented three retrieval approaches.

### 4.1 BM25 Keyword Retrieval

BM25 ranks scientific papers according to matching keywords.

It is particularly useful when scientific claims contain exact terminology, such as drug names, diseases, genes, and technical terms.

**Implementation:** `src/retrieval/bm25.py`

Run:

```bash
python -m src.retrieval.bm25
```

### 4.2 Sentence-BERT Semantic Retrieval

We use:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model converts each sentence chunk into a dense vector embedding.

These embeddings are stored in a persistent **Qdrant vector database**.

A scientific claim is also converted into an embedding and compared with the indexed sentence vectors.

**Implementation:**

- `src/build_index.py`
- `src/retrieval/search.py`

Generate embeddings and build the Qdrant index:

```bash
python -m src.build_index
```

Run semantic retrieval:

```bash
python -m src.retrieval.search
```

The Qdrant index stores **45,952 sentence embeddings**.

### 4.3 Hybrid Retrieval + Cross-Encoder Reranking

The hybrid system combines BM25 and Sentence-BERT retrieval.

Its pipeline is:

```text
Scientific Claim
       |
       +--------------------+
       |                    |
       v                    v
  BM25 Search       Sentence-BERT Search
       |                    |
       v                    v
 Top 30 Papers        Top 30 Papers
       |                    |
       +----------+---------+
                  |
                  v
       Merge and Deduplicate
                  |
                  v
       Cross-Encoder Reranking
                  |
                  v
          Top 10 Papers
```

The cross-encoder model is:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

It evaluates each claim–paper pair jointly and ranks candidate papers according to relevance.

**Implementation:**

- `src/retrieval/hybrid.py`
- `configs/hybrid.yaml`

Run:

```bash
python -m src.retrieval.hybrid
```

---

## 5. Retrieval Evaluation

Retrieval systems are evaluated using **SciFact's labeled development claims**.

Of the 300 development claims, **188 contain annotated evidence documents** and are included in the retrieval evaluation.

### Evaluation Metrics

**Recall@5:** Fraction of annotated evidence documents retrieved within the top five papers.

**Recall@10:** Fraction of annotated evidence documents retrieved within the top ten papers.

**MRR@10:** Mean reciprocal rank of the first correctly retrieved evidence document.

### Experimental Results

| Metric | BM25 | Sentence-BERT | Hybrid + Cross-Encoder |
|--------|------|---------------|------------------------|
| Recall@5 | 0.8848 | 0.9006 | **0.9288** |
| Recall@10 | 0.9158 | 0.9431 | **0.9578** |
| MRR@10 | 0.7822 | 0.7945 | **0.8526** |
| Evaluation Time | 9.87 s | 35.75 s | 1982.15 s |

### Observations

- Sentence-BERT achieved higher retrieval recall than BM25.
- Hybrid retrieval achieved the highest Recall@5, Recall@10, and MRR@10.
- Hybrid retrieval required significantly more computation because of cross-encoder reranking.
- The results demonstrate a trade-off between retrieval quality and computational cost.

These comparisons are preliminary because BM25 searches complete abstracts, while semantic retrieval searches sentence chunks and aggregates results by document ID.

### Run Evaluations

BM25:

```bash
python -m src.evaluation.evaluate_bm25
```

Sentence-BERT:

```bash
python -m src.evaluation.evaluate_retrieval
```

Hybrid:

```bash
python -m src.evaluation.evaluate_hybrid
```

Evaluation metrics are stored as JSON files in the `reports/` directory.

---

## 6. DVC Pipeline and Data Versioning

DVC is used to make the data pipeline reproducible.

The project has used **DagsHub for remote DVC storage**. The supplied repository snapshot has an empty `.dvc/config`, so new clones must configure the remote before `dvc pull` or `dvc push`. Never commit credentials.

The pipeline contains the following stages:

| Stage | Purpose |
|-------|---------|
| `ingest` | Download SciFact |
| `validate` | Validate dataset structure and annotations |
| `preprocess` | Generate documents and sentence chunks |
| `build_index` | Generate embeddings and build Qdrant index |
| `evaluate_bm25` | Evaluate BM25 retrieval |
| `evaluate_semantic` | Evaluate Sentence-BERT retrieval |
| `evaluate_hybrid` | Evaluate hybrid retrieval |

### Execute the Pipeline

```bash
dvc repro
```

DVC automatically executes stages in dependency order.

### View Pipeline DAG

```bash
dvc dag
```

### Upload Versioned Artifacts

```bash
dvc push
```

### Download Versioned Artifacts

```bash
dvc pull
```

The configured DVC remote stores datasets, generated embeddings, Qdrant index files, and other tracked outputs.

Generated data and model artifacts are not committed directly to GitHub.

---

## 7. Installation and Setup

### Prerequisites

- Python 3.12 (recommended; verify compatibility with pinned dependencies)
- Git
- DVC
- Internet connection
- Access to the project's DagsHub remote storage

### Clone Repository

```bash
git clone https://github.com/nihaal237/Multi-Agent-Scientific-Claim-Verification.git

cd Multi-Agent-Scientific-Claim-Verification
```

### Create Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Download Data and Artifacts

After configuring your own DVC remote and authentication (the repository snapshot contains an empty `.dvc/config`; obtain the remote URL from the project owner):

```bash
dvc pull
```

This downloads the SciFact data, processed documents, and Qdrant index.

You do not need to regenerate the 45,952 embeddings if the versioned index has already been downloaded.

### Verify Pipeline

```bash
dvc status
```

### Test Retrieval

```bash
python -m src.retrieval.bm25
python -m src.retrieval.search
python -m src.retrieval.hybrid
```

---

## 8. Repository Structure

```text
Multi-Agent-Scientific-Claim-Verification/
|
├── configs/
│   ├── retrieval.yaml
│   └── hybrid.yaml
|
├── src/
│   ├── ingest.py
│   ├── validate.py
│   ├── preprocess.py
│   ├── build_index.py
│   |
│   ├── retrieval/
│   │   ├── bm25.py
│   │   ├── search.py
│   │   └── hybrid.py
│   |
│   ├── evaluation/
│   │   ├── evaluate_bm25.py
│   │   ├── evaluate_retrieval.py
│   │   └── evaluate_hybrid.py
│   |
│   ├── agents/
│   └── utils/
|
├── data/
│   ├── raw/
│   └── processed/
|
├── artifacts/
│   └── index/
│       └── qdrant/
|
├── reports/
|
├── .dvc/
├── dvc.yaml
├── dvc.lock
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 9. Planned Multi-Agent Architecture

The next development stage will implement five agents using LangGraph.

| Agent | Purpose |
|-------|---------|
| Claim Analyzer Agent | Identifies the important parts of a scientific claim |
| Retrieval Agent | Finds relevant research papers using hybrid retrieval |
| Evidence Agent | Extracts evidence sentences from retrieved papers |
| Verification Agent | Determines whether evidence supports or contradicts the claim |
| Report Agent | Generates a final verdict with explanations and citations |

The verification system will use a pretrained NLI model and a hosted instruction LLM.

The system will be evaluated using SciFact's annotated evidence sentences and SUPPORT/CONTRADICT labels.

---

## 10. Planned MLOps Components

The project will integrate:

- **LangGraph:** Multi-agent orchestration.
- **MLflow:** Remote experiment tracking, evaluation metrics, and model registry.
- **Langfuse:** Agent traces, tool calls, and token usage.
- **GitHub Actions:** CI/CD automation.
- **Docker + Docker Compose:** Application containerization.
- **AWS EC2 + S3:** Cloud deployment and artifact storage.
- **FastAPI:** Backend API.
- **React:** User interface.

Docker, frontend development, cloud deployment, and monitoring will be addressed afterwards.

---


## 11. Reproducibility, Status, and Known Limitations

**Verified development status (13 November 2026):** Data ingestion, validation, preprocessing, local Qdrant indexing, three retrieval methods, and development-set retrieval evaluations are implemented. Validation passed with zero errors. The reported metrics are from the project handover; they have not been independently rerun in this README update.

**Not yet implemented in this repository snapshot:** An end-to-end support/contradiction verifier, the five LangGraph agents, evidence-level and verdict-level evaluation, remote MLflow experiments and registry registration, a web interface, Docker, cloud deployment, CI/CD, and monitoring. The `src/agents/` directory is currently a placeholder. Installing packages such as FastAPI or MLflow does not mean those services are operational.

**Reproducibility notes:**

- Run all commands from the repository root. `dvc repro` runs the declared seven stages; it does **not** yet reproduce a complete multi-agent verification workflow.
- `python -m src.build_index` deletes and rebuilds `artifacts/index/qdrant`; use `dvc pull` when an existing index is available. Avoid concurrent access to the same local Qdrant directory.
- The dataset download is protected by a fixed SHA-256 checksum; if the upstream `latest` archive changes, ingestion intentionally fails.
- The provided `requirements.txt` is a large environment-specific pinned dependency export (including Windows-specific packages). Installation on other platforms may require a curated dependency file.
- The SciFact test claims have no public evidence labels in this project, so the reported retrieval evaluation uses the annotated development subset.
- The original assignment requires a baseline **and** an end-to-end main system, two remote MLflow runs, a registered artifact, and reproducible pipeline stages; these are not yet all satisfied.

### Next implementation milestones

1. Add BM25 + pretrained NLI baseline with evidence sentence extraction and verdict metrics.
2. Implement claim analyzer, retrieval, evidence, verification, and report agents with LangGraph.
3. Add fixed-split end-to-end evaluations, including evidence F1, verdict Macro-F1, latency, and failure rates.
4. Track reproducible runs on remote MLflow and register the relevant model or system artifact.
5. Version agent prompts/configuration and evaluation outputs in DVC, then verify `dvc pull` and `dvc repro` from a clean clone.
6. Subsequently implement FastAPI, React, Docker, AWS deployment, and observability.

### Credential safety

Keep DagsHub tokens, MLflow credentials, AWS keys, and `.env` files out of Git. Configure access using local environment variables or credential storage. Do not put usernames with passwords or tokens inside tracked `.dvc/config`, README examples, or committed URLs.

---

## 12. References

- SciFact: https://github.com/allenai/scifact
- SciFact Dataset on Hugging Face: https://huggingface.co/datasets/allenai/scifact
- Sentence-Transformers: https://www.sbert.net/
- Qdrant: https://qdrant.tech/
- DVC: https://dvc.org/
- DagsHub: https://dagshub.com/
- MLflow: https://mlflow.org/
- LangGraph: https://github.com/langchain-ai/langgraph

---

## 13. Academic Use

This project is developed for the MLOps course, Fall 2026, at FAST-NUCES Lahore.

Team members:

- Nihaal Durrani (23I-0831)
- Roshni Waheed (23L-0919)

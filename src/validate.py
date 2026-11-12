"""Before using the dataset for retrieval or model evaluation, we need to check that it's correct and complete. Our validation script will check:
1. All four dataset files exist and contain records.
2. Every research paper has a unique ID, title, and abstract.
3. Every scientific claim has an ID and nonempty text.
4. Evidence labels are either SUPPORT or CONTRADICT.
5. Evidence references point to existing papers.
6. Evidence sentence indices are valid."""

import json
from pathlib import Path
from collections import Counter

DATA_DIR = Path("data/raw")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

FILES = [
    "corpus.jsonl",
    "claims_train.jsonl",
    "claims_dev.jsonl",
    "claims_test.jsonl",
]

VALID_LABELS = {"SUPPORT", "CONTRADICT"}


def load_jsonl(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

    with path.open(encoding="utf-8") as f:
        records = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    if not records:
        raise ValueError(f"Empty dataset: {filename}")

    return records


def validate():
    errors = []
    stats = {}

    corpus = load_jsonl("corpus.jsonl")
    paper_ids = {}
    seen = set()

    for paper in corpus:
        pid = str(paper.get("doc_id", ""))

        if not pid or pid == "None":
            errors.append("Paper missing doc_id")
            continue

        if pid in seen:
            errors.append(f"Duplicate paper ID: {pid}")

        seen.add(pid)
        paper_ids[pid] = paper

        if not paper.get("title"):
            errors.append(f"Paper {pid}: missing title")

        abstract = paper.get("abstract")
        if not isinstance(abstract, list) or not abstract:
            errors.append(f"Paper {pid}: invalid abstract")

    stats["papers"] = len(corpus)

    all_claim_ids = set()

    for filename in FILES[1:]:
        claims = load_jsonl(filename)
        stats[filename] = len(claims)

        label_counts = Counter()

        for claim in claims:
            cid = str(claim.get("id", ""))

            if not cid or cid == "None":
                errors.append(f"{filename}: missing claim ID")

            if cid in all_claim_ids:
                errors.append(f"Duplicate claim ID: {cid}")

            all_claim_ids.add(cid)

            if not claim.get("claim"):
                errors.append(f"Claim {cid}: empty text")

            evidence = claim.get("evidence", {})

            for doc_id, entries in evidence.items():
                paper = paper_ids.get(str(doc_id))

                if paper is None:
                    errors.append(
                        f"Claim {cid}: unknown paper {doc_id}"
                    )
                    continue

                for entry in entries:
                    label = entry.get("label")
                    label_counts[label] += 1

                    if label not in VALID_LABELS:
                        errors.append(
                            f"Claim {cid}: invalid label {label}"
                        )

                    for idx in entry.get("sentences", []):
                        if (
                            not isinstance(idx, int)
                            or idx < 0
                            or idx >= len(paper["abstract"])
                        ):
                            errors.append(
                                f"Claim {cid}: invalid sentence "
                                f"{idx} in paper {doc_id}"
                            )

        stats[f"{filename}_labels"] = dict(label_counts)

    report = {
        "status": "PASS" if not errors else "FAIL",
        "statistics": stats,
        "error_count": len(errors),
        "errors": errors[:100]
    }

    output = REPORT_DIR / "validation.json"

    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(report, indent=2))

    if errors:
        raise ValueError(
            f"Dataset validation failed: {len(errors)} errors"
        )

    print("\nDataset validation PASSED!")


if __name__ == "__main__":
    validate()

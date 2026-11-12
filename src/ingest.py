"""Python script that downloads SciFact automatically, without requiring Bash or manual file downloads."""

import hashlib
import io
import json
import tarfile
from pathlib import Path

import requests

DATA_URL = (
    "https://scifact.s3-us-west-2.amazonaws.com/"
    "release/latest/data.tar.gz"
)

EXPECTED_SHA256 = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"

OUTPUT_DIR = Path("data/raw")
MANIFEST_PATH = OUTPUT_DIR / "source_manifest.json"

REQUIRED_FILES = [
    "corpus.jsonl",
    "claims_train.jsonl",
    "claims_dev.jsonl",
    "claims_test.jsonl",
]


def download_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading SciFact dataset...")

    response = requests.get(DATA_URL, timeout=120)
    response.raise_for_status()

    archive_bytes = response.content
    checksum = hashlib.sha256(archive_bytes).hexdigest()

    #ensures that if SciFact's latest download changes, the pipeline fails rather than silently switching datasets.

    if checksum != EXPECTED_SHA256: 
     raise ValueError(
        "SciFact source archive has changed. "
        "Expected a different SHA-256 checksum."
    )

    with tarfile.open(
        fileobj=io.BytesIO(archive_bytes),
        mode="r:gz"
    ) as archive:

        members = {
            Path(member.name).name: member
            for member in archive.getmembers()
            if member.isfile()
        }

        for filename in REQUIRED_FILES:
            member = members.get(filename)

            if member is None:
                raise FileNotFoundError(
                    f"Missing dataset file: {filename}"
                )

            source = archive.extractfile(member)

            if source is None:
                raise RuntimeError(
                    f"Cannot read: {filename}"
                )

            output_path = OUTPUT_DIR / filename

            with source, output_path.open("wb") as output:
                output.write(source.read())

            print(f"Saved: {output_path}")

    manifest = {
        "source": DATA_URL,
        "sha256": checksum,
        "files": REQUIRED_FILES
    }

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8"
    )

    print("\nDataset downloaded successfully!")
    print("Source SHA256:", checksum)


if __name__ == "__main__":
    download_dataset()

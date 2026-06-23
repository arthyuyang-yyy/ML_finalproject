"""Scan a local audio dataset and create a project JSONL manifest."""

import argparse
from pathlib import Path

from src.datasets.discovery import discover_audio_records
from src.datasets.manifest import write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a local audio-dataset manifest without copying data.")
    parser.add_argument("root", help="Extracted dataset directory")
    parser.add_argument("--dataset", required=True, help="Dataset name, for example alimeeting or ami")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--language", required=True, help="Language code such as zh or en")
    parser.add_argument("--annotation-root", help="Optional separate annotation directory")
    parser.add_argument("--output", required=True, help="Target JSONL manifest")
    args = parser.parse_args()

    records = discover_audio_records(
        args.root,
        dataset=args.dataset,
        split=args.split,
        language=args.language,
        annotation_root=args.annotation_root,
    )
    write_manifest(Path(args.output), records)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()

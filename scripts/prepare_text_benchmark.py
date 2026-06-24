"""Convert QMSum or VCSum into project-local evaluation JSONL."""

import argparse

from src.datasets.text_benchmarks import prepare_qmsum, prepare_vcsum


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a text-only meeting evaluation benchmark.")
    parser.add_argument("dataset", choices=["qmsum", "vcsum"])
    parser.add_argument("root", help="Cloned dataset repository root")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    converter = prepare_qmsum if args.dataset == "qmsum" else prepare_vcsum
    count = converter(args.root, args.output, split=args.split)
    print(f"Wrote {count} {args.dataset} records to {args.output}")


if __name__ == "__main__":
    main()

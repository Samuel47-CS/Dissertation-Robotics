"""Script to rename occurrences of a target identifier inside model metadata files."""

import argparse
from pathlib import Path

import pandas as pd
from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_MODELS_DIR = Path("models")
TARGET = "wrist.right"
REPLACEMENT = "right_wrist"


def rename_string(value, target=TARGET, replacement=REPLACEMENT):
    if isinstance(value, str) and target in value:
        return value.replace(target, replacement)
    return value


def process_json(file_path):
    file_path = Path(file_path)
    text = file_path.read_text(encoding="utf-8")
    if TARGET in text:
        file_path.write_text(text.replace(TARGET, REPLACEMENT), encoding="utf-8")


def process_parquet(file_path):
    file_path = Path(file_path)
    df = pd.read_parquet(file_path)
    df = df.applymap(rename_string)
    df.columns = [rename_string(col) for col in df.columns]
    df.to_parquet(file_path, index=False)


def process_safetensor(file_path):
    file_path = Path(file_path)
    with safe_open(file_path, framework="pt", device="cpu") as source:
        tensors = {name.replace(TARGET, REPLACEMENT): source.get_tensor(name) for name in source.keys()}
        metadata = source.metadata()

    new_metadata = None
    if metadata:
        new_metadata = {
            rename_string(key): rename_string(value)
            for key, value in metadata.items()
        }

    save_file(tensors, file_path, metadata=new_metadata)


def process_model_files(models_dir=DEFAULT_MODELS_DIR):
    models_dir = Path(models_dir)
    for model_dir in sorted(models_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        print(f"Processing model directory: {model_dir.name}")
        for file_path in model_dir.rglob("*"):
            if file_path.suffix == ".json":
                process_json(file_path)
            elif file_path.suffix == ".parquet":
                process_parquet(file_path)
            elif file_path.suffix == ".safetensors":
                process_safetensor(file_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rename target identifiers in model metadata files."
    )
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_MODELS_DIR),
        help="Directory containing model subdirectories to process.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Renaming occurrences of '{TARGET}' to '{REPLACEMENT}' in {args.models_dir}")
    process_model_files(args.models_dir)


if __name__ == "__main__":
    main()
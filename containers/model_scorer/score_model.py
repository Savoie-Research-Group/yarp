import argparse
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd


MODEL_DIR = Path("/opt/yarp_models")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Score YARP conformer-pair indicators.")
    parser.add_argument("--model", choices=["poor_model", "rich_model"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model_path = MODEL_DIR / f"{args.model}.sav"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")

    features = pd.read_csv(args.input)
    with model_path.open("rb") as f:
        model = pickle.load(f)

    payload = {
        "model": args.model,
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
        "n_rows": int(len(features)),
        "proba": model.predict_proba(features).tolist(),
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

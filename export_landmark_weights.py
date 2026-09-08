"""
Re-exports browser weight JSON from a trained landmark .pkl.

The .pkl files are the source of truth; web/*_landmarks_weights.json drift out
of sync whenever a model is retrained without re-running the full training
script. This rebuilds the JSON from the pkl's own coefs_/intercepts_ without
touching the dataset.

    python export_landmark_weights.py            # rewrite web/*.json in place
    python export_landmark_weights.py --out /tmp # dry run somewhere harmless
    python export_landmark_weights.py --only isl
"""

import argparse
import json
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
WEB_DIR = BASE_DIR / "web"

MODELS = {
    "asl": (MODELS_DIR / "asl_landmark_model.pkl", "asl_landmarks_weights.json"),
    "isl": (MODELS_DIR / "isl_landmark_model.pkl", "isl_landmarks_weights.json"),
}


def export(pkl_path: Path, out_path: Path):
    with open(pkl_path, "rb") as f:
        pkg = pickle.load(f)

    clf = pkg["model"]
    class_names = [str(c) for c in pkg["classes"]]

    if len(class_names) != clf.coefs_[-1].shape[1]:
        raise SystemExit(
            f"[!] {pkl_path.name}: {len(class_names)} class names but "
            f"{clf.coefs_[-1].shape[1]} output units"
        )

    weights = {"classes": class_names}
    for i, (w, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
        weights[f"w{i}"] = w.tolist()
        weights[f"b{i}"] = b.tolist()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(weights, f)

    shapes = " -> ".join(str(c.shape[0]) for c in clf.coefs_) + f" -> {clf.coefs_[-1].shape[1]}"
    print(f"[+] {pkl_path.name}: {shapes} | {len(class_names)} classes -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(WEB_DIR), help="output directory (default: web/)")
    ap.add_argument("--only", choices=sorted(MODELS), help="export just one model")
    args = ap.parse_args()

    out_dir = Path(args.out)
    names = [args.only] if args.only else sorted(MODELS)

    for name in names:
        pkl_path, json_name = MODELS[name]
        if not pkl_path.exists():
            print(f"[!] skipping {name}: {pkl_path} not found")
            continue
        export(pkl_path, out_dir / json_name)


if __name__ == "__main__":
    main()

"""
Run this script once from any directory to generate ONNX model files and a
metadata JSON consumed by the web app.

    pip install skl2onnx onnx joblib
    python js-app/export_models.py

Outputs are written to js-app/models/:
    full.onnx
    reduced_z.onnx
    reduced_raw.onnx
    reduced_raw_z.onnx
    reduced_demo_z.onnx
    reduced_demo_raw.onnx
    reduced_demo_raw_z.onnx
    metadata.json          (thresholds, feature lists, training-set means)

The original .pkl files in the repo root are never modified.
"""

import json
import os
import sys

import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = SCRIPT_DIR
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
except ImportError:
    sys.exit("Install skl2onnx first:  pip install skl2onnx onnx")

# All model suffixes: 1 full + 6 reduced combinations
SUFFIXES = [
    "full",
    "reduced_z",
    "reduced_raw",
    "reduced_raw_z",
    "reduced_demo_z",
    "reduced_demo_raw",
    "reduced_demo_raw_z",
]

metadata = {}

for suffix in SUFFIXES:
    model     = joblib.load(os.path.join(REPO_DIR, f"model_{suffix}.pkl"))
    threshold = joblib.load(os.path.join(REPO_DIR, f"threshold_{suffix}.pkl"))
    features  = joblib.load(os.path.join(REPO_DIR, f"features_{suffix}.pkl"))
    means     = joblib.load(os.path.join(REPO_DIR, f"feature_means_{suffix}.pkl"))

    n = len(features)
    initial_type = [("float_input", FloatTensorType([None, n]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=12,
        options={type(model): {"zipmap": False}},
    )

    onnx_path = os.path.join(MODELS_DIR, f"{suffix}.onnx")
    with open(onnx_path, "wb") as fh:
        fh.write(onnx_model.SerializeToString())

    metadata[suffix] = {
        "threshold": float(threshold),
        "features":  list(features),
        "means":     {k: float(v) for k, v in means.items()},
    }
    print(f"{suffix:25s}  features={n}  threshold={float(threshold):.4f}  -> {onnx_path}")

with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as fh:
    json.dump(metadata, fh, indent=2)

print("\nDone. Serve the app locally with:")
print("  python -m http.server 8080 --directory js-app/")
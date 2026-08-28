"""JSON manifests written by ``generate`` and consumed by ``collect`` / ``fit``.

They carry every parameter of the generation step (strain set, magnitudes,
model settings, directory names) so that nothing has to be re-typed -- the
original strainIFCcoupling scripts did not check that ``-smag`` agreed
between the two steps.
"""

import datetime
import json
import os

import numpy as np

IFC_MANIFEST = "strainifc_manifest.json"
ELASTIC_MANIFEST = "elastic_manifest.json"
VERSION = 1


def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def save_manifest(data, path):
    data = dict(data)
    data.setdefault("version", VERSION)
    data.setdefault("created", datetime.datetime.now().isoformat(timespec="seconds"))
    with open(path, "w") as f:
        json.dump(_to_jsonable(data), f, indent=2)


def load_manifest(path, expect_tool=None):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"manifest {path} not found; run the 'generate' step first (in the same --outdir)"
        )
    with open(path) as f:
        data = json.load(f)
    if data.get("version") != VERSION:
        raise ValueError(f"{path}: unsupported manifest version {data.get('version')}")
    if expect_tool is not None and data.get("tool") != expect_tool:
        raise ValueError(
            f"{path}: manifest was written by {data.get('tool')!r}, expected {expect_tool!r}"
        )
    return data

# src/ML/stress_test.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np

def find_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p/"models").exists(): return p
    return here.parents[2]
ROOT = find_root()

import sys
sys.path.append(str(ROOT))
from src.ML.predict_parisian import predict_out_price

BASE = dict(S0=100.0, K=100.0, T=1.0, r=0.02, q=0.0, sigma=0.2, B=80.0, D=0.25)

def run(model_dir=None, blend=False):
    out = {}
    def add(name, **kw):
        v_out, v_in = predict_out_price(model_dir=model_dir, blend=blend, **kw)
        out[name] = dict(OUT=v_out, IN=v_in)
    add("base", **BASE)
    add("S0->0", **{**BASE, "S0": 1e-6})
    add("sigma->0", **{**BASE, "sigma": 1e-6})
    add("T->0", **{**BASE, "T": 1e-6})
    add("barrier->0", **{**BASE, "B": 1e-6})
    add("deep_ITM", **{**BASE, "S0": 200.0})
    add("deep_OTM", **{**BASE, "S0": 20.0})
    return out

if __name__ == "__main__":
    res = run()
    print(json.dumps(res, indent=2))

"""Run by the user: imports the unmodified public estimator and compares it with the submission's function
and my transcription on controlled inputs. Usage: python u1_run_public_code.py
Default path is the local clone at commit bcee82c of the public MI-Peaks repository."""
import os
import ast
import importlib.util
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
PUBLIC = os.environ.get("MI_PEAKS_ROOT", "MI-Peaks") + "/src/mi_estimators.py"
SUBMITTED = Path(__file__).resolve().parents[2] / "original_experiments" / "autonomous_harmed_mi_search.py"
spec = importlib.util.spec_from_file_location("public_mi", PUBLIC)
public = importlib.util.module_from_spec(spec)
spec.loader.exec_module(public)
ns = {"np": np, "torch": torch}
for node in ast.parse(SUBMITTED.read_text(encoding="utf-8")).body:
    if isinstance(node, ast.FunctionDef) and node.name in {"distmat_t", "kernelmat", "hsic_normalized_cca"}:
        exec(compile(ast.Module([node], []), "submitted", "exec"), ns)
rng = np.random.default_rng(20261030)
for s in [0.5, 1, 2, 5, 10]:
    pub, sub = [], []
    for _ in range(100):
        x = torch.from_numpy(rng.normal(0, s, 128).astype(np.float32))
        y = torch.from_numpy(rng.normal(0, s, 128).astype(np.float32))
        pub.append(float(public.estimate_mi_hsic(x, y)))
        sub.append(ns["hsic_normalized_cca"](x, y, 50.0))
    pub, sub = np.array(pub), np.array(sub)
    print(f"s={s}: public negative {(pub < 0).mean():.1%}, min {pub.min():.4f}; "
          f"submitted negative {(sub < 0).mean():.1%}; mean |public-submitted| {np.abs(pub - sub).mean():.2e}")

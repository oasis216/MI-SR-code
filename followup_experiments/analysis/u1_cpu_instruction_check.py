"""Rerun the submission's estimate_mi_curve on fixed inputs under one environment; save curve.
args: path of submission .py, inputs .npy, output .npy, n_threads."""
import ast, sys
import numpy as np, torch
src, inp, out, th = sys.argv[1:5]
torch.set_num_threads(int(th))
ns = {"np": np, "torch": torch}
for node in ast.parse(open(src, encoding="utf-8").read()).body:
    if isinstance(node, ast.FunctionDef) and node.name == "estimate_mi_curve":
        exec(compile(ast.Module([node], []), "submitted", "exec"), ns)
a = np.load(inp)
np.save(out, ns["estimate_mi_curve"](a[:-1], a[-1], 128, 50.0, 64))

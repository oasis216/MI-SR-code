"""Read pickle opcodes as data only; never unpickle or execute GLOBAL/REDUCE.

The archive's curve store is known from audited source to store five float32
ndarrays per case, in A-E order. Validate that shape/count assumption and compare
the reference q0170_b6 series with independently supplied CSV before reporting.
"""
import os
from pathlib import Path
from zipfile import ZipFile
import csv, json, pickletools, re
import numpy as np

ROOT = Path(__file__).resolve().parent
ARCHIVE = Path(os.environ.get("MI_SR_ARCHIVE_ZIP", "raw_archive-20260729.zip"))
MODELS = ['qwen3_14b_reference', 'deepseek_r1_qwen14b', 'mistral7b_instruct_v03']
reference = {}
with (ROOT/'explanatory_case_q0170_b6_curves.csv').open(encoding='utf-8-sig',newline='') as f:
    for row in csv.DictReader(f):
        reference.setdefault(row['candidate'],[]).append(float(row['hsic_dependence_proxy']))

output=[]
with ZipFile(ARCHIVE) as z:
    for model in MODELS:
        member=f'output_mi_sr_full_targeted_v5/{model}/all_biases/all_candidate_curve_store.pkl'
        data=z.read(member)
        current=None
        buffers=[]
        cases={}
        dtype_strings=set()
        for op,arg,pos in pickletools.genops(data):
            if op.name in {'SHORT_BINUNICODE','BINUNICODE','BINUNICODE8'}:
                if isinstance(arg,str) and re.fullmatch(r'q\d+_b\d+',arg):
                    if current is not None:
                        assert len(buffers)==5,(current,len(buffers))
                        cases[current]=buffers
                    current=arg
                    buffers=[]
                if isinstance(arg,str) and re.fullmatch(r'[fiucb][0-9]+',arg):
                    dtype_strings.add(arg)
            elif op.name in {'BINBYTES','SHORT_BINBYTES','BINBYTES8','BYTEARRAY8'} and len(arg)>4:
                assert current is not None
                assert len(arg)%4==0 and len(arg)<=4*256, (current,len(arg))
                buffers.append(np.frombuffer(arg,dtype='<f4'))
        assert len(buffers)==5,(current,len(buffers))
        cases[current]=buffers
        assert dtype_strings=={'f4'},dtype_strings
        if model==MODELS[0]:
            for letter,values in zip('ABCDE',cases['q0170_b6']):
                assert np.array_equal(values.astype(float),reference[letter]),letter
        arrays=[v for case in cases.values() for v in case]
        total=sum(v.size for v in arrays)
        negative=sum(int(np.count_nonzero(v<0)) for v in arrays)
        out=dict(model=model,archive_member=member,cases=len(cases),curves=len(arrays),
                 curve_values=total,negative_values=negative,negative_fraction=negative/total,
                 curves_with_negative=sum(bool(np.any(v<0)) for v in arrays),
                 cases_with_negative=sum(any(np.any(v<0) for v in c) for c in cases.values()),
                 min=min(float(v.min()) for v in arrays),max=max(float(v.max()) for v in arrays),
                 all_finite=all(bool(np.isfinite(v).all()) for v in arrays),
                 dtype_strings=sorted(dtype_strings))
        output.append(out)
        print(json.dumps(out),flush=True)
(ROOT/'static_curve_buffer_summary.json').write_text(json.dumps(output,indent=2),encoding='utf-8')

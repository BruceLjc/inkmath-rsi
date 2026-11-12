"""文件职责：复现基础符号顺序，构建不含人工答案的邻近笔迹上下文通道。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,random
import numpy as np
from inkmath.ink import context_raster
ROOT=Path(__file__).resolve().parents[1];random.seed(42);labels=json.loads((ROOT/'data/processed/labels.json').read_text());contexts=[]
for k,line in enumerate((ROOT/'data/processed/expressions.jsonl').open()):
    r=json.loads(line);sym=r['symbols']
    for si,s in enumerate(sym):
        if s['label'] not in labels:continue
        contexts.append(context_raster(r['strokes'],s['indices'],max_context=100000))
        if si+1<len(sym) and random.random()<.55:contexts.append(context_raster(r['strokes'],s['indices']+sym[si+1]['indices'],max_context=100000))
    if k%1000==0:print('processed',k,flush=True)
assert len(contexts)==len(np.load(ROOT/'data/processed/symbols.npz')['images'])
np.save(ROOT/'data/processed/contexts.npy',np.array(contexts));print('done',len(contexts),flush=True)

"""文件职责：复现原符号样本顺序，生成方向与抬笔特征，不改变已有数据划分。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,random
import numpy as np
from inkmath.ink import trajectory,raster
ROOT=Path(__file__).resolve().parents[1];random.seed(42)
labels=json.loads((ROOT/'data/processed/labels.json').read_text());traces=[]
for k,line in enumerate((ROOT/'data/processed/expressions.jsonl').open()):
    r=json.loads(line);sym=r['symbols']
    for si,s in enumerate(sym):
        if s['label'] not in labels:continue
        ss=[r['strokes'][j] for j in s['indices']];traces.append(trajectory(ss))
        if si+1<len(sym) and random.random()<.55:traces.append(trajectory(ss+[r['strokes'][j] for j in sym[si+1]['indices']]))
    if k%1000==0:print('processed',k,flush=True)
base=np.load(ROOT/'data/processed/symbols.npz');assert len(traces)==len(base['images'])
np.save(ROOT/'data/processed/trajectories.npy',np.array(traces));print('done',len(traces),flush=True)

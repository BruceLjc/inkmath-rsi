"""文件职责：仅在训练划分挖掘模型误接受的笔画组，加入拒识负例，验证/测试真值不参与训练。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,time,collections
import numpy as np
from inkmath.recognizer import Recognizer
from inkmath.ink import raster,trajectory,context_raster
ROOT=Path(__file__).resolve().parents[1]
rec=Recognizer(ROOT/'artifacts/v2.pt');labels=rec.labels;reject=labels.index('<reject>');ims=[];gs=[];fingerprints=[];traces=[];contexts=[];count=0;tick=time.monotonic()
for line in (ROOT/'data/processed/expressions.jsonl').open():
    r=json.loads(line)
    if r['split']!='train' or len(r['strokes'])>128:continue
    groups=rec.candidates(r['strokes']);gold={tuple(sorted(s['indices'])) for s in r['symbols']};p,b=rec.classify(r['strokes'],groups)
    scores=p.copy();scores[:,reject]=0;conf=scores.max(-1);candidates=[(float(conf[i]),i) for i,g in enumerate(groups) if tuple(g) not in gold]
    # 覆盖高置信错合并与不完整单笔；每条公式最多采十个，防止长式主导训练。
    for score,i in sorted(candidates,reverse=True)[:10]:
        if score<.25:continue
        im,g=raster([r['strokes'][j] for j in groups[i]]);ims.append(im);gs.append(g);fingerprints.append(r['fingerprint']);traces.append(trajectory([r['strokes'][j] for j in groups[i]]));contexts.append(context_raster(r['strokes'],groups[i],max_context=100000))
    count+=1
    if count%500==0:print('formulas',count,'negatives',len(ims),'seconds',time.monotonic()-tick,flush=True)
np.savez_compressed(ROOT/'data/processed/hard-negatives.npz',images=np.array(ims),geometry=np.array(gs),targets=np.full(len(ims),reject,dtype=np.int64),splits=np.zeros(len(ims),dtype=np.int64))
np.save(ROOT/'data/processed/hard-trajectories.npy',np.array(traces));np.save(ROOT/'data/processed/hard-contexts.npy',np.array(contexts))
(ROOT/'reports/hard-negative-provenance.json').write_text(json.dumps({'split':'train','formulas':count,'examples':len(ims),'unique_source_fingerprints':sorted(set(fingerprints)),'checkpoint':'v2.pt','seconds':time.monotonic()-tick},indent=2))
print('done',len(ims),flush=True)

"""文件职责：从公开 CROHME 研究归档构建去重、分组隔离的符号训练集与公式评测集。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import collections, hashlib, json, zipfile, random
import numpy as np
from inkmath.ink import read_inkml, raster
ROOT=Path(__file__).resolve().parents[1]
random.seed(42)
z=zipfile.ZipFile(ROOT/'data/raw/CROHME_full_v2.zip')
records=[]; seen=set(); counts=collections.Counter(); errors=[]
for name in z.namelist():
    if '/CROHME2013_data/TrainINKML/' not in name or not name.endswith('.inkml'): continue
    try:
        r=read_inkml(z.read(name))
        if not r['symbols'] or not r['mathml']: continue
        fingerprint=hashlib.sha256(json.dumps(r['strokes'],separators=(',',':')).encode()).hexdigest()
        if fingerprint in seen: continue
        seen.add(fingerprint)
        family=name.split('/TrainINKML/')[1].split('/')[0]
        # 已有作者 ID 则严格按作者分组；无 ID 的样本按来源会话代理分组，报告单独说明局限。
        stem=Path(name).stem
        proxy=stem.split('-')[0] if family=='MathBrush' else stem.split('_')[0] if family=='KAIST' else stem
        group=family+':'+(r['writer'] or proxy)
        val=int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%100
        split='test' if val<12 else 'val' if val<24 else 'train'
        r.update(id=Path(name).as_posix(),fingerprint=fingerprint,split=split,group=group,has_writer=bool(r['writer']))
        del r['writer']; records.append(r)
        counts.update(s['label'] for s in r['symbols'])
    except Exception as e: errors.append({'source':name,'error':str(e)})
labels=sorted(k for k,v in counts.items() if v>=20)+['<reject>']; idx={v:i for i,v in enumerate(labels)}
images=[];geometry=[];targets=[];splits=[]
for k,r in enumerate(records):
    sym=r['symbols']; owner={j:i for i,s in enumerate(sym) for j in s['indices']}
    for si,s in enumerate(sym):
        if s['label'] not in idx: continue
        ss=[r['strokes'][j] for j in s['indices']]
        im,g=raster(ss);images.append(im);geometry.append(g);targets.append(idx[s['label']]);splits.append(['train','val','test'].index(r['split']))
        # 相邻但不属于同一字符的合并作为拒识负例，防止模型总偏好多笔合并。
        if si+1<len(sym) and random.random()<0.55:
            other=sym[si+1]
            im,g=raster(ss+[r['strokes'][j] for j in other['indices']]);images.append(im);geometry.append(g);targets.append(idx['<reject>']);splits.append(['train','val','test'].index(r['split']))
    if k%1000==0: print('processed',k,flush=True)
out=ROOT/'data/processed'
np.savez_compressed(out/'symbols.npz',images=np.array(images),geometry=np.array(geometry),targets=np.array(targets),splits=np.array(splits))
(out/'expressions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
(out/'labels.json').write_text(json.dumps(labels,ensure_ascii=False,indent=2))
summary={'expressions':len(records),'symbols_with_negatives':len(images),'classes':len(labels),'splits':dict(collections.Counter(r['split'] for r in records)),'known_writer_expressions':sum(r['has_writer'] for r in records),'class_counts':dict(counts),'errors':errors,'fingerprint_duplicates_removed':len([n for n in z.namelist() if '/TrainINKML/' in n and n.endswith('.inkml')])-len(records)}
(out/'manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ['class_counts','errors']},indent=2))

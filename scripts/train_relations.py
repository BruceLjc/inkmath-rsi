"""文件职责：按固定作者/会话划分训练二维关系模型，记录验证指标和最优检查点。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,random,time
import numpy as np
import torch
from torch.nn import functional as F
from inkmath.relations import RelationNet,gold_edges,pair_features,RELATIONS
ROOT=Path(__file__).resolve().parents[1];random.seed(42);torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(3)
labels=json.loads((ROOT/'data/processed/labels.json').read_text());lookup={v:i for i,v in enumerate(labels)};features=[];pairs=[];targets=[];split=[];errors=0
for line in (ROOT/'data/processed/expressions.jsonl').open():
    r=json.loads(line)
    if r['split']=='test':continue
    try:edges,root=gold_edges(r)
    except Exception:errors+=1;continue
    sym=r['symbols'];truth={(i,j):k for i,j,k in edges};n=len(sym);med=float(np.median([max(s['box'][3]-s['box'][1],1e-6) for s in sym]));neg=[(i,j) for i in range(n) for j in range(n) if i!=j and (i,j) not in truth];random.shuffle(neg)
    chosen=list(truth)+neg[:max(len(truth)*3,3)]
    for i,j in chosen:
        if sym[i]['label'] not in lookup or sym[j]['label'] not in lookup:continue
        features.append(pair_features(sym[i]['box'],sym[j]['box'],med));pairs.append([lookup[sym[i]['label']],lookup[sym[j]['label']]]);targets.append(truth.get((i,j),0));split.append(r['split']=='val')
print('relation examples',len(features),'parse errors',errors,flush=True)
x=torch.from_numpy(np.array(features));p=torch.tensor(pairs);y=torch.tensor(targets);split=np.array(split);tr=np.where(~split)[0];va=np.where(split)[0]
device='mps' if torch.backends.mps.is_available() else 'cpu';net=RelationNet(len(labels)).to(device);optim=torch.optim.AdamW(net.parameters(),lr=.002);best=0;logs=[]
for epoch in range(30):
    tick=time.monotonic();net.train();np.random.shuffle(tr);losses=[]
    for off in range(0,len(tr),2048):
        ix=tr[off:off+2048];optim.zero_grad(set_to_none=True);logits=net(x[ix].to(device),p[ix].to(device));loss=F.cross_entropy(logits,y[ix].to(device));loss.backward();optim.step();losses.append(loss.item())
    net.eval();preds=[]
    with torch.inference_mode():
        for off in range(0,len(va),4096):
            ix=va[off:off+4096];preds.append(net(x[ix].to(device),p[ix].to(device)).argmax(-1).cpu())
    pred=torch.cat(preds);truth=y[va];mask=truth!=0;score=float((pred[mask]==truth[mask]).float().mean());acc=float((pred==truth).float().mean());row={'epoch':epoch+1,'val_positive_accuracy':score,'val_accuracy':acc,'loss':float(np.mean(losses)),'seconds':time.monotonic()-tick};logs.append(row);print(json.dumps(row),flush=True)
    if score>best:
        best=score;torch.save({'state':{k:v.cpu() for k,v in net.state_dict().items()},'labels':labels,'epoch':epoch+1,'positive_accuracy':score,'relations':RELATIONS},ROOT/'artifacts/relations.pt')
    (ROOT/'reports/train-relations.json').write_text(json.dumps(logs,indent=2))

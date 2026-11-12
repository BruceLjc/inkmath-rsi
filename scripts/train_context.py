"""文件职责：训练带局部上下文和笔顺的数学符号模型，针对误分组进行第三轮改进。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,time,random
import numpy as np
import torch
from torch.nn import functional as F
from inkmath.model import TrajectorySymbolNet,DualSymbolNet
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=35);p.add_argument('--name',default='v4');p.add_argument('--dual',action='store_true');a=p.parse_args()
torch.manual_seed(43);np.random.seed(43);torch.set_num_threads(4);device='mps' if torch.backends.mps.is_available() else 'cpu'
d=dict(np.load(ROOT/'data/processed/symbols.npz'));extra=np.load(ROOT/'data/processed/hard-negatives.npz');contexts=np.concatenate([np.load(ROOT/'data/processed/contexts.npy'),np.load(ROOT/'data/processed/hard-contexts.npy')]);traces=np.concatenate([np.load(ROOT/'data/processed/trajectories.npy'),np.load(ROOT/'data/processed/hard-trajectories.npy')]);d={k:np.concatenate([d[k],extra[k]]) for k in d}
assert len(traces)==len(d['images'])==len(contexts)
x=torch.from_numpy(np.concatenate([d['images'],contexts],1).astype(np.float32)/255);g=torch.from_numpy(d['geometry']);t=torch.from_numpy(traces);y=torch.from_numpy(d['targets'].astype(np.int64));tr=np.where(d['splits']==0)[0];va=np.where(d['splits']==1)[0]
labels=json.loads((ROOT/'data/processed/labels.json').read_text());net=(DualSymbolNet if a.dual else TrajectorySymbolNet)(len(labels));old=torch.load(ROOT/('artifacts/v4.pt' if a.dual else 'artifacts/v2.pt'),map_location='cpu',weights_only=True)['state'];state=net.state_dict()
for k in state:
    if k not in old:continue
    if state[k].shape==old[k].shape:state[k]=old[k]
    elif k=='features.0.weight':state[k][:,:1]=old[k];state[k][:,1:]=0
    elif k=='head.0.weight':state[k][:,:old[k].shape[1]]=old[k];state[k][:,old[k].shape[1]:]*=.1
net.load_state_dict(state);net.to(device);opt=torch.optim.AdamW(net.parameters(),lr=.0007,weight_decay=.0001);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs,eta_min=.00006)
counts=np.bincount(y[tr].numpy(),minlength=len(labels));weights=torch.tensor(np.clip((np.median(counts[counts>0])/np.maximum(counts,1))**.2,.5,2),dtype=torch.float32,device=device)
print(json.dumps({'name':a.name,'parameters':sum(p.numel() for p in net.parameters()),'train':len(tr),'val':len(va),'device':device}),flush=True);logs=[];best=-1;begin=time.monotonic()
for epoch in range(a.epochs):
    tick=time.monotonic();net.train();np.random.shuffle(tr);losses=[]
    for off in range(0,len(tr),384):
        ix=tr[off:off+384];xb=x[ix].to(device);gb=g[ix].to(device);tb=t[ix].to(device);yb=y[ix].to(device)
        # 两幅线条通道同时做平移，轨迹保留原坐标；这是一种轻量模态扰动增强。
        if epoch%2==0:
            theta=torch.eye(2,3,device=device).unsqueeze(0).repeat(len(ix),1,1);theta[:,0,1]=torch.randn(len(ix),device=device)*.04;theta[:,:,2]=torch.randn(len(ix),2,device=device)*.03
            xb=F.grid_sample(xb,F.affine_grid(theta,xb.shape,align_corners=False),align_corners=False)
        opt.zero_grad(set_to_none=True);out=net(xb,gb,tb)
        if a.dual:
            classes,valid=out;positive=yb!=len(labels)-1
            loss=F.cross_entropy(classes[positive,:-1],yb[positive],weight=weights[:-1],label_smoothing=.01)+.8*F.binary_cross_entropy_with_logits(valid,positive.float())
        else:loss=F.cross_entropy(out,yb,weight=weights,label_smoothing=.01)
        loss.backward();opt.step();losses.append(loss.item())
    net.eval();correct=0;symcorrect=0;total=0
    with torch.inference_mode():
        for off in range(0,len(va),512):
            ix=va[off:off+512];pred=(net.probabilities if a.dual else net)(x[ix].to(device),g[ix].to(device),t[ix].to(device)).argmax(-1).cpu();truth=y[ix];mask=truth!=labels.index('<reject>');correct+=(pred==truth).sum().item();symcorrect+=((pred==truth)&mask).sum().item();total+=mask.sum().item()
    score=symcorrect/total;row={'epoch':epoch+1,'loss':float(np.mean(losses)),'val_symbol_accuracy':score,'val_accuracy':correct/len(va),'seconds':time.monotonic()-tick,'elapsed':time.monotonic()-begin};print(json.dumps(row),flush=True);logs.append(row)
    if score>best:
        best=score;torch.save({'state':{k:v.cpu() for k,v in net.state_dict().items()},'labels':labels,'epoch':epoch+1,'val_symbol_accuracy':score,'name':a.name,'architecture':'context-dual' if a.dual else 'context-trajectory','parameters':sum(p.numel() for p in net.parameters())},ROOT/f'artifacts/{a.name}.pt')
    (ROOT/f'reports/train-{a.name}.json').write_text(json.dumps(logs,indent=2));scheduler.step()

"""文件职责：在本地 MPS/CPU 训练轻量数学符号模型，按验证集保存最优权重与每轮记录。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,time,random
import numpy as np
import torch
from torch.nn import functional as F
from inkmath.model import SymbolNet
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=20);p.add_argument('--name',default='v1');p.add_argument('--resume');p.add_argument('--augment',action='store_true');p.add_argument('--batch',type=int,default=256);p.add_argument('--hard-negatives',action='store_true');a=p.parse_args()
torch.manual_seed(42);random.seed(42);np.random.seed(42);torch.set_num_threads(4)
device='mps' if torch.backends.mps.is_available() else 'cpu'
d=dict(np.load(ROOT/'data/processed/symbols.npz'))
if a.hard_negatives:
    extra=np.load(ROOT/'data/processed/hard-negatives.npz')
    d={k:np.concatenate([d[k],extra[k]]) for k in d}
labels=json.loads((ROOT/'data/processed/labels.json').read_text())
x=torch.from_numpy(d['images'].astype(np.float32)/255);g=torch.from_numpy(d['geometry']);y=torch.from_numpy(d['targets'].astype(np.int64));split=d['splits']
train=np.where(split==0)[0];val=np.where(split==1)[0]
net=SymbolNet(len(labels)).to(device)
if a.resume: net.load_state_dict(torch.load(a.resume,map_location='cpu',weights_only=True)['state'])
optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,a.epochs,eta_min=.00008)
# 轻度平衡长尾类别，保留真实字频，避免极端重加权损害常用符号。
counts=np.bincount(y[train].numpy(),minlength=len(labels));w=np.clip((np.median(counts[counts>0])/np.maximum(counts,1))**.25,.5,3)
weights=torch.tensor(w,dtype=torch.float32,device=device)
print(json.dumps({'device':device,'train':len(train),'val':len(val),'parameters':sum(p.numel() for p in net.parameters()),'labels':labels}),flush=True)
logs=[];best=-1;start=time.monotonic()
for epoch in range(a.epochs):
    tick=time.monotonic();net.train();np.random.shuffle(train);losses=[]
    for offset in range(0,len(train),a.batch):
        ix=train[offset:offset+a.batch];xb=x[ix].to(device);gb=g[ix].to(device);yb=y[ix].to(device)
        if a.augment:
            # 保持数学方向，允许轻微平移、缩放、倾斜；禁止翻转与大角度旋转。
            theta=torch.eye(2,3,device=device).unsqueeze(0).repeat(len(ix),1,1)
            theta[:,0,0]+=torch.randn(len(ix),device=device)*.06;theta[:,1,1]+=torch.randn(len(ix),device=device)*.06
            theta[:,0,1]=torch.randn(len(ix),device=device)*.08;theta[:,:,2]=torch.randn(len(ix),2,device=device)*.05
            xb=F.grid_sample(xb,F.affine_grid(theta,xb.shape,align_corners=False),align_corners=False)
        optimizer.zero_grad(set_to_none=True);logits=net(xb,gb);loss=F.cross_entropy(logits,yb,weight=weights,label_smoothing=.02);loss.backward();optimizer.step();losses.append(loss.item())
    net.eval();correct=0;symbol_correct=0;symbol_total=0
    with torch.inference_mode():
        for offset in range(0,len(val),a.batch):
            ix=val[offset:offset+a.batch];pred=net(x[ix].to(device),g[ix].to(device)).argmax(1).cpu();truth=y[ix];correct+=(pred==truth).sum().item();mask=truth!=labels.index('<reject>');symbol_correct+=((pred==truth)&mask).sum().item();symbol_total+=mask.sum().item()
    score=symbol_correct/max(symbol_total,1);row={'epoch':epoch+1,'loss':float(np.mean(losses)),'val_accuracy':correct/max(len(val),1),'val_symbol_accuracy':score,'seconds':time.monotonic()-tick,'elapsed':time.monotonic()-start};logs.append(row);print(json.dumps(row),flush=True)
    if score>best:
        best=score
        torch.save({'state':{k:v.cpu() for k,v in net.state_dict().items()},'labels':labels,'epoch':epoch+1,'val_symbol_accuracy':score,'name':a.name,'parameters':sum(p.numel() for p in net.parameters())},ROOT/f'artifacts/{a.name}.pt')
    (ROOT/f'reports/train-{a.name}.json').write_text(json.dumps(logs,indent=2));scheduler.step()

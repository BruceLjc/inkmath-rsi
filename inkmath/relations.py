"""文件职责：学习并解码数学符号间的二维关系，按图结构处理积分、求和和嵌套表达式。"""
import math
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch import nn
from .ink import XMLID
from .layout import Node,make,union,normalize_label
RELATIONS=['None','Right','Sup','Sub','Above','Below','Inside','Index']


def gold_edges(record):
    """从 MathML 和符号引用提取监督关系；此函数仅供离线训练和诊断使用。"""
    refs={s['ref']:i for i,s in enumerate(record['symbols']) if s.get('ref') is not None};edges=[]
    def visit(n):
        tag=n.tag.split('}')[-1];own=refs.get(n.get(XMLID));children=list(n)
        if tag in ['mi','mn','mo','mtext']:return own
        roots=[visit(c) for c in children]
        if tag in ['mfrac','msqrt','mroot']:
            if own is None:return next((r for r in roots if r is not None),None)
            roles=['Above','Below'] if tag=='mfrac' else ['Inside','Index'] if tag=='mroot' else ['Inside']*len(roots)
            for r,rel in zip(roots,roles):
                if r is not None:edges.append((own,r,RELATIONS.index(rel)))
            return own
        if tag in ['msup','msub','msubsup','mover','munder','munderover']:
            if not roots:return None
            base=roots[0];roles=['Sup'] if tag in ['msup','mover'] else ['Sub'] if tag in ['msub','munder'] else ['Sub','Sup']
            for r,rel in zip(roots[1:],roles):
                if base is not None and r is not None:edges.append((base,r,RELATIONS.index(rel)))
            return base
        roots=[r for r in roots if r is not None]
        for a,b in zip(roots,roots[1:]):edges.append((a,b,1))
        return roots[0] if roots else None
    root=visit(ET.fromstring(record['mathml']))
    return edges,root


def pair_features(a,b,median):
    """生成尺度无关的相对位置特征，原点和画板缩放不应改变关系判断。"""
    ax,ay,ar,ab=a;bx,by,br,bb=b;ha=max(ab-ay,1e-6);hb=max(bb-by,1e-6);wa=max(ar-ax,1e-6);wb=max(br-bx,1e-6);h=max(ha,hb,median*.2);m=max(median,1e-6)
    vals=[((bx+br)-(ax+ar))/2/h,((by+bb)-(ay+ab))/2/h,(bx-ax)/h,(br-ar)/h,(by-ay)/h,(bb-ab)/h,(bx-ar)/h,(by-ab)/h,math.log(ha/hb),math.log(wa/wb),min(ar,br)-max(ax,bx),min(ab,bb)-max(ay,by),wa/m,wb/m,ha/m,hb/m,(bx-ar)/m,(by-ab)/m]
    vals[10]/=max(min(wa,wb),m*.05);vals[11]/=max(min(ha,hb),m*.05)
    return np.clip(vals,-12,12).astype(np.float32)


class RelationNet(nn.Module):
    """小型关系分类器，把符号类别与相对空间特征共同用于二维布局。"""
    def __init__(self,classes):
        super().__init__();self.embedding=nn.Embedding(classes,12);self.layers=nn.Sequential(nn.Linear(42,96),nn.ReLU(),nn.Linear(96,64),nn.ReLU(),nn.Linear(64,len(RELATIONS)))
    def forward(self,features,labels):
        """预测有向符号对属于同行、上下标、分子分母等关系的分数。"""
        return self.layers(torch.cat([features,self.embedding(labels[:,0]),self.embedding(labels[:,1])],-1))


class GraphLayout:
    """以最大生成有向树消除关系环，再导出布局树；不读取任何人工答案。"""
    def __init__(self,path,mode='odds'):
        self.mode=mode
        c=torch.load(path,map_location='cpu',weights_only=True);self.labels=c['labels'];self.lookup={v:i for i,v in enumerate(self.labels)};self.net=RelationNet(len(self.labels));self.net.load_state_dict(c['state']);self.net.eval()

    def parse(self,symbols):
        """联合判断符号关系，允许积分上下限、分数及根式跨越书写顺序。"""
        import networkx as nx
        n=len(symbols)
        if not n:return Node('',[0,0,0,0],kind='row')
        if n==1:return make('row',symbols)
        med=float(np.median([max(s.box[3]-s.box[1],1e-6) for s in symbols]))
        pairs=[(i,j) for i in range(n) for j in range(n) if i!=j]
        features=np.stack([pair_features(symbols[i].box,symbols[j].box,med) for i,j in pairs]);labels=np.array([[self.lookup.get(symbols[i].label,0),self.lookup.get(symbols[j].label,0)] for i,j in pairs])
        with torch.inference_mode():probs=self.net(torch.from_numpy(features),torch.from_numpy(labels)).softmax(-1).numpy()
        graph=nx.DiGraph();graph.add_nodes_from(range(n+1));minx=min(s.box[0] for s in symbols)
        incoming={j:[] for j in range(n)}
        for (i,j),p in zip(pairs,probs):
            rel=int(p[1:].argmax())+1
            if self.mode=='calibrated':
                score=float(np.log(max(p[rel],1e-8))-4*p[0])
                # 语法结构有局部优先级，避免高置信同行边抢走分子、分母或根号内部。
                if symbols[i].label=='-' and rel in [4,5] and p[rel]>.65:score+=.6
                if symbols[i].label==r'\sqrt' and rel in [6,7] and p[rel]>.65:score+=.35
            else:score=float(np.log(max(p[rel],1e-8)/max(p[0],1e-8)))
            incoming[j].append((score,i,rel))
        for j in range(n):
            graph.add_edge(n,j,weight=-4.0-(symbols[j].box[0]-minx)/max(med,1e-6)*.1,rel=0)
            for score,i,rel in sorted(incoming[j],reverse=True)[:5]:graph.add_edge(i,j,weight=score,rel=rel)
        tree=nx.maximum_spanning_arborescence(graph,attr='weight',preserve_attrs=True)
        kids={i:{} for i in range(n+1)}
        for i,j,d in tree.edges(data=True):kids[i].setdefault(d['rel'],[]).append(j)
        def combine(indices):
            ordered=sorted(indices,key=lambda i:symbols[i].box[0]);return make('row',[build(i) for i in ordered])
        def build(i):
            base=symbols[i];c=kids[i]
            if base.label=='-' and c.get(4) and c.get(5):
                b=make('frac',[combine(c[4]),combine(c[5])]);b.indices+=base.indices;b.box=union([base,b]);base=b
            elif base.label==r'\sqrt' and c.get(6):
                b=make('root' if c.get(7) else 'sqrt',([combine(c[7])] if c.get(7) else [])+[combine(c[6])]);b.indices+=base.indices;b.box=union([base,b]);base=b
            if c.get(2) or c.get(3):base=make('subsup' if c.get(2) and c.get(3) else 'sub' if c.get(3) else 'sup',[base]+([combine(c[3])] if c.get(3) else [])+([combine(c[2])] if c.get(2) else []))
            # 未形成完整分式/根式的关系不能丢笔画，降为可见同行候选并暴露不确定结果。
            leftover=[j for rel,js in c.items() if rel not in [1,2,3] and not (rel in [4,5] and symbols[i].label=='-' and c.get(4) and c.get(5)) and not (rel in [6,7] and symbols[i].label==r'\sqrt' and c.get(6)) for j in js]
            rest=c.get(1,[])+leftover
            return make('row',[base,combine(rest)]) if rest else base
        return combine(kids[n].get(0,[]))

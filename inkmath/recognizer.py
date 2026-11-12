"""文件职责：执行候选笔画分组、轻量符号分类和二维结构解析，提供真实本地推理结果。"""
from pathlib import Path
import itertools, math, time
import numpy as np
import torch
from .ink import box,raster,validate_strokes,trajectory,context_raster
from .model import SymbolNet,TrajectorySymbolNet,DualSymbolNet
from .layout import Node,parse_layout,latex
ROOT=Path(__file__).resolve().parents[1]


class Recognizer:
    """常驻 CPU 数学笔迹识别器；不加载第三方识别模型，不调用云端。"""
    def __init__(self, checkpoint=None, beam=12, grouping_cost=None, max_candidates=900, layout='graph', edge_mode=None, refine_checkpoint=None):
        torch.set_num_threads(2)
        config={}
        if checkpoint is None and (ROOT/'artifacts/selected.json').exists():
            import json
            config=json.loads((ROOT/'artifacts/selected.json').read_text())
            checkpoint=ROOT/config['checkpoint']
            refine_checkpoint=refine_checkpoint or (ROOT/config['refine_checkpoint'] if config.get('refine_checkpoint') else None)
        grouping_cost=config.get('grouping_cost',0.0) if grouping_cost is None else grouping_cost
        edge_mode=config.get('edge_mode','odds') if edge_mode is None else edge_mode
        path=Path(checkpoint or ROOT/'artifacts/best.pt')
        if not path.exists():raise FileNotFoundError(f'缺少模型权重：{path}；请先训练并运行 scripts/rsi.py --promote')
        c=torch.load(path,map_location='cpu',weights_only=True)
        self.labels=c['labels'];self.context=c.get('architecture') in ['context-trajectory','context-dual'];self.dual=c.get('architecture')=='context-dual';self.model=(DualSymbolNet if self.dual else TrajectorySymbolNet if self.context else SymbolNet)(len(self.labels));self.model.load_state_dict(c['state']);self.model.eval()
        self.beam=beam;self.grouping_cost=grouping_cost;self.max_candidates=max_candidates;self.version=c['name'];self.parameters=c['parameters'];self.model_epoch=c['epoch']
        self.refiner=None
        if refine_checkpoint:
            ref=torch.load(refine_checkpoint,map_location='cpu',weights_only=True)
            if ref['labels']!=self.labels:raise ValueError('符号复核模型的标签表不匹配')
            self.refiner=SymbolNet(len(self.labels));self.refiner.load_state_dict(ref['state']);self.refiner.eval();self.version+='+'+ref['name']+'-labels';self.parameters+=ref['parameters']
        self.layout_mode=layout
        self.graph_layout=None
        if layout=='graph':
            from .relations import GraphLayout
            path=ROOT/'artifacts/relations.pt'
            if not path.exists():raise FileNotFoundError('缺少二维关系模型，请先运行 scripts/train_relations.py')
            self.graph_layout=GraphLayout(path,mode=edge_mode)
        # 预热放在构造阶段；CLI 和评测另报冷启动时间，不隐藏首次加载成本。
        with torch.inference_mode():
            if self.context:self.model(torch.zeros(1,2,32,32),torch.zeros(1,3),torch.zeros(1,5,64))
            else:self.model(torch.zeros(1,1,32,32),torch.zeros(1,3))

    def candidates(self, strokes):
        """生成有限的连续与近邻组合，兼容常见延迟补笔；不做指数规模的全子集搜索。"""
        n=len(strokes);boxes=[box([s]) for s in strokes];groups=set()
        for i in range(n):
            for size in range(1,min(6,n-i)+1):groups.add(tuple(range(i,i+size)))
        # 延迟补笔常出现在 x、等号、点和字母横画；空间近邻补充二笔、三笔候选。
        for i,b in enumerate(boxes):
            scale=max(b[2]-b[0],b[3]-b[1],1e-5)
            near=sorted((j for j in range(i+1,n)),key=lambda j:((boxes[j][0]+boxes[j][2]-b[0]-b[2])**2+(boxes[j][1]+boxes[j][3]-b[1]-b[3])**2))[:4]
            for j in near:
                bb=boxes[j];dx=max(0,max(b[0],bb[0])-min(b[2],bb[2]));dy=max(0,max(b[1],bb[1])-min(b[3],bb[3]))
                if max(dx,dy)<scale*.8:groups.add((i,j))
            for j,k in itertools.combinations(near[:3],2):groups.add(tuple(sorted((i,j,k))))
        # 单笔永不被裁掉，否则可能无法覆盖全部输入。其余候选确定性裁剪以控制尾延迟。
        singles=[(i,) for i in range(n)];other=sorted((g for g in groups if len(g)>1),key=lambda g:(max(g)-min(g),len(g),g))
        return singles+other[:max(0,self.max_candidates-n)]

    def classify(self, strokes, groups):
        """一次批量分类全部候选，避免逐符号调用神经网络的调度开销。"""
        ims=[];gs=[];boxes=[];traces=[]
        strokes=[np.asarray(s,dtype=np.float64)[:,:2] for s in strokes];all_boxes=np.array([box([s]) for s in strokes])
        for group in groups:
            ss=[strokes[i] for i in group];bb=all_boxes[list(group)];b=[float(bb[:,0].min()),float(bb[:,1].min()),float(bb[:,2].max()),float(bb[:,3].max())];im,g=raster(ss,bounds=b)
            if self.context:im=np.concatenate([im,context_raster(strokes,group,boxes=all_boxes,bounds=b)],0);traces.append(trajectory(ss,bounds=b))
            ims.append(im);gs.append(g);boxes.append(b)
        with torch.inference_mode():
            args=[torch.from_numpy(np.stack(ims).astype(np.float32)/255),torch.from_numpy(np.stack(gs))]
            if self.context:args.append(torch.from_numpy(np.stack(traces)))
            probs=self.model.probabilities(*args).numpy() if self.dual else self.model(*args).softmax(-1).numpy()
        return probs,boxes

    def refine(self,strokes,groups,probs):
        """分组模型决定笔画归属后，可用独立形状模型复核类别，减少拒识训练的类别偏移。"""
        if self.refiner is None:return probs
        ims=[];gs=[]
        for group in groups:
            im,g=raster([strokes[i] for i in group]);ims.append(im);gs.append(g)
        with torch.inference_mode():
            other=self.refiner(torch.from_numpy(np.stack(ims).astype(np.float32)/255),torch.from_numpy(np.stack(gs))).softmax(-1).numpy()
        return probs*.35+other*.65

    def parse_symbols(self,symbols):
        """根据显式配置使用学习图解析或规则基线，便于可复现对照。"""
        return self.graph_layout.parse(symbols) if self.graph_layout else parse_layout(symbols)

    def recognize(self, strokes, task_id='local', revision=0):
        """识别坐标并返回 LaTeX、结构与耗时；低分只标记不确定，不伪造成功。"""
        started=time.perf_counter();strokes=validate_strokes(strokes);groups=self.candidates(strokes);probs,boxes=self.classify(strokes,groups);classified=time.perf_counter()
        by_first={i:[] for i in range(len(strokes))};reject=self.labels.index('<reject>')
        for ci,group in enumerate(groups):
            # 首版保留 top-1 符号，分组用 beam；后续可将结构反馈加入候选重排。
            p=probs[ci].copy();p[reject]=0;li=int(p.argmax());confidence=float(p[li])
            score=math.log(max(confidence,1e-9))-self.grouping_cost
            mask=sum(1<<i for i in group)
            by_first[group[0]].append((mask,score,ci,li,confidence))
        for i in by_first:by_first[i].sort(key=lambda a:a[1],reverse=True)
        # 每个状态确保每笔恰好被消费一次；同一覆盖集合只保留较优路径。
        full=(1<<len(strokes))-1;buckets=[{} for _ in range(len(strokes)+1)];buckets[0][0]=(0.0,[])
        for consumed in range(len(strokes)):
            current=sorted(buckets[consumed].items(),key=lambda kv:kv[1][0],reverse=True)[:self.beam]
            for mask,(score,path) in current:
                i=next(j for j in range(len(strokes)) if not mask&(1<<j))
                for cm,cs,ci,li,conf in by_first[i][:30]:
                    if cm&mask:continue
                    nm=cm|mask;ns=score+cs;bucket=buckets[nm.bit_count()]
                    if nm not in bucket or ns>bucket[nm][0]:bucket[nm]=(ns,path+[(ci,li,conf)])
            # 分桶按已覆盖笔画数比较，避免只因路径符号数少就过早淘汰正确分组。
            for k in range(consumed+1,min(consumed+7,len(strokes))+1):
                if len(buckets[k])>self.beam*8:buckets[k]=dict(sorted(buckets[k].items(),key=lambda kv:kv[1][0],reverse=True)[:self.beam*4])
        complete=[v for k,v in buckets[-1].items() if k==full]
        if not complete:raise ValueError('候选搜索无法覆盖全部笔画')
        score,path=max(complete,key=lambda a:a[0])
        if self.refiner is not None:
            chosen=[ci for ci,li,conf in path];refined=self.refine(strokes,[groups[ci] for ci in chosen],probs[chosen]);refined[:,reject]=0;indices=refined.argmax(-1);path=[(ci,int(li),float(refined[k,li])) for k,(ci,li) in enumerate(zip(chosen,indices))]
        symbols=[Node(self.labels[li],boxes[ci],list(groups[ci]),conf) for ci,li,conf in path]
        tree=self.parse_symbols(symbols);result=latex(tree);elapsed=(time.perf_counter()-started)*1000
        uncertain=[{'stroke_ids':s.indices,'label':s.label,'score':round(s.confidence,4)} for s in symbols if s.confidence<.65]
        return {'task_id':task_id,'revision':revision,'latex':result,'status':'uncertain' if uncertain else 'candidate','uncertain_symbols':uncertain,'tree':tree.as_dict(),'symbols':[s.as_dict() for s in symbols],'timing_ms':{'total':round(elapsed,3),'candidate_and_model':round((classified-started)*1000,3)},'within_1s':elapsed<1000,'model':self.version,'candidate_count':len(groups),'warnings':['实验模型输出需核对；分组、布局和置信分数尚未校准。']}

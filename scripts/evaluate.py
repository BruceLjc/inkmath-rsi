"""文件职责：隔离真实端到端评测、已知分组诊断和已知符号布局诊断，记录错误与尾延迟。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,time,collections,hashlib
import numpy as np
from inkmath.recognizer import Recognizer
from inkmath.layout import Node,parse_layout,latex,canonical,mathml_latex
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='artifacts/v1.pt');p.add_argument('--split',choices=['val','test'],default='val');p.add_argument('--limit',type=int,default=120);p.add_argument('--name',default='v1-val');p.add_argument('--grouping-cost',type=float,default=0.0);p.add_argument('--layout',choices=['graph','rules'],default='graph');p.add_argument('--edge-mode',choices=['odds','calibrated'],default='odds');p.add_argument('--refine-checkpoint');a=p.parse_args()
records=[json.loads(s) for s in (ROOT/'data/processed/expressions.jsonl').read_text().splitlines()];records=[r for r in records if r['split']==a.split and len(r['strokes'])<=128 and sum(len(s) for s in r['strokes'])<=12000]
# 固定哈希顺序，后续修改只能使用验证集挑选，最终测试集不参与模型选择。
records.sort(key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest());records=records[:a.limit] if a.limit else records
load=time.perf_counter();rec=Recognizer(a.checkpoint,grouping_cost=a.grouping_cost,layout=a.layout,edge_mode=a.edge_mode,refine_checkpoint=a.refine_checkpoint);cold=(time.perf_counter()-load)*1000
rows=[]
for i,r in enumerate(records):
    target=mathml_latex(r['mathml']);gold_symbols=r['symbols'];groups=[s['indices'] for s in gold_symbols]
    try:
        pred=rec.recognize(r['strokes'],task_id='eval-'+str(i));correct=canonical(pred['latex'])==canonical(target)
        # 此两项诊断明确使用真值，不是可交付识别能力，绝不混入端到端指标。
        ps,bs=rec.classify(r['strokes'],groups);ps=rec.refine(r['strokes'],groups,ps);pl=ps.argmax(-1);symok=sum(rec.labels[li]==s['label'] for li,s in zip(pl,gold_symbols))
        oracle_layout=latex(rec.parse_symbols([Node(s['label'],s['box'],s['indices']) for s in gold_symbols]))
        oracle_group=latex(rec.parse_symbols([Node(rec.labels[int(li)],b,s['indices']) for li,b,s in zip(pl,bs,gold_symbols)]))
        predicted_groups={tuple(sorted(s['stroke_ids'])) for s in pred['symbols']};true_groups={tuple(sorted(s['indices'])) for s in gold_symbols}
        cats=[name for marker,name in [(r'\int','integral'),(r'\sum','sum'),(r'\lim','limit'),(r'\sqrt','root'),(r'\frac','fraction'),(r'\sin','function'),(r'\cos','function'),(r'\log','function'),('^{','superscript'),('_{','subscript')] if marker in target]
        rows.append({'id':r['id'],'categories':sorted(set(cats)) or ['basic'],'target':target,'prediction':pred['latex'],'correct':correct,'ms':pred['timing_ms']['total'],'symbol_correct':symok,'symbol_total':len(groups),'grouping_exact':predicted_groups==true_groups,'oracle_label_layout_correct':canonical(oracle_layout)==canonical(target),'oracle_group_correct':canonical(oracle_group)==canonical(target),'oracle_layout_prediction':oracle_layout,'uncertain':pred['status']=='uncertain','strokes':len(r['strokes'])})
    except Exception as e:rows.append({'id':r['id'],'correct':False,'error':str(e),'categories':['error']})
    if i%20==0:print('evaluated',i,flush=True)
ms=[r['ms'] for r in rows if 'ms' in r];n=len(rows)
summary={'split':a.split,'samples':n,'checkpoint':a.checkpoint,'grouping_cost':a.grouping_cost,'layout':a.layout,'edge_mode':a.edge_mode,'refine_checkpoint':a.refine_checkpoint,'model_epoch':rec.model_epoch,'model_parameters':rec.parameters,'model_load_and_warmup_ms':cold,'e2e_exact':sum(r['correct'] for r in rows)/max(n,1),'correct_within_1s':sum(r['correct'] and r.get('ms',1e10)<1000 for r in rows)/max(n,1),'latency_ms':{'p50':float(np.percentile(ms,50)),'p95':float(np.percentile(ms,95)),'p99':float(np.percentile(ms,99)),'max':max(ms)},'over_1s':sum(m>=1000 for m in ms),'symbol_accuracy_given_gold_groups':sum(r.get('symbol_correct',0) for r in rows)/max(sum(r.get('symbol_total',0) for r in rows),1),'grouping_exact':sum(r.get('grouping_exact',False) for r in rows)/max(n,1),'layout_exact_given_gold_labels_and_groups':sum(r.get('oracle_label_layout_correct',False) for r in rows)/max(n,1),'e2e_given_gold_groups':sum(r.get('oracle_group_correct',False) for r in rows)/max(n,1),'uncertain_rate':sum(r.get('uncertain',True) for r in rows)/max(n,1),'errors':sum('error'in r for r in rows)}
categories=collections.defaultdict(list)
for r in rows:
    for c in r['categories']:categories[c].append(r)
summary['categories']={c:{'n':len(rs),'correct':sum(r['correct'] for r in rs),'layout_oracle_correct':sum(r.get('oracle_label_layout_correct',False) for r in rs)} for c,rs in categories.items()}
(ROOT/f'reports/{a.name}.json').write_text(json.dumps({'summary':summary,'rows':rows},ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))

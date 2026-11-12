"""文件职责：运行有边界的本地训练/验证迭代，以整式准确率和一秒约束自动选择与晋级候选。"""
import argparse,hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    """记录模型与验证报告指纹，保证晋级结论可追溯到具体权重。"""
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    """只用验证集选择版本；独立测试集绝不被循环调参消费。"""
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=160);p.add_argument('--models',default='v4,v5');p.add_argument('--promote',action='store_true');p.add_argument('--train',action='store_true');p.add_argument('--epochs',type=int,default=8);p.add_argument('--rounds',type=int,default=1);a=p.parse_args()
    if not 1<=a.rounds<=3:raise ValueError('一次迭代最多 3 轮；请先检查上一轮真实误差')
    names=[n.strip() for n in a.models.split(',') if n.strip()]
    if any(not n.replace('-','').replace('_','').isalnum() for n in names):raise ValueError('模型名称只允许字母、数字、横线和下划线')
    if a.train:
        for rnd in range(a.rounds):
            name='rsi-'+time.strftime('%Y%m%d-%H%M%S')+'-'+str(rnd+1)
            subprocess.run([sys.executable,'scripts/train_context.py','--dual','--epochs',str(a.epochs),'--name',name],cwd=ROOT,check=True);names.append(name)
    if (ROOT/'artifacts/selected.json').exists() and 'best' not in names:names.append('best')
    candidates=[]
    for name in names:
        path=ROOT/f'artifacts/{name}.pt'
        if not path.exists():continue
        for edge in ['odds','calibrated']:
            report_name=f'rsi-{name}-{edge}-{a.limit}'
            command=[sys.executable,'scripts/evaluate.py','--checkpoint',str(path),'--split','val','--limit',str(a.limit),'--name',report_name,'--edge-mode',edge]
            with (ROOT/f'reports/{report_name}.log').open('w') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=log,check=True)
            summary=json.loads((ROOT/f'reports/{report_name}.json').read_text())['summary']
            entry={'name':name,'path':str(path.relative_to(ROOT)),'edge_mode':edge,'summary':summary,'checkpoint_sha256':digest(path),'report':'reports/'+report_name+'.json'};candidates.append(entry);print(json.dumps({'candidate':name,'edge':edge,'e2e':summary['e2e_exact'],'p99_ms':summary['latency_ms']['p99'],'max_ms':summary['latency_ms']['max']},ensure_ascii=False),flush=True)
    eligible=[c for c in candidates if c['summary']['errors']==0 and c['summary']['over_1s']==0]
    if not eligible:raise RuntimeError('没有同时通过输入稳定性和本轮一秒验证的候选，不晋级')
    best=max(eligible,key=lambda c:(c['summary']['correct_within_1s'],-c['summary']['latency_ms']['p95']))
    ledger={'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'selection_split':'val','test_used':False,'limit':a.limit,'candidates':candidates,'selected':best['name'],'selected_edge':best['edge_mode'],'promoted':a.promote}
    if a.promote:
        if (ROOT/best['path']).resolve()!=(ROOT/'artifacts/best.pt').resolve():shutil.copy2(ROOT/best['path'],ROOT/'artifacts/best.pt')
        config={'checkpoint':'artifacts/best.pt','source_checkpoint':best['path'],'checkpoint_sha256':best['checkpoint_sha256'],'relation_sha256':digest(ROOT/'artifacts/relations.pt'),'edge_mode':best['edge_mode'],'grouping_cost':0.0,'refine_checkpoint':None,'validation_report':best['report'],'selection_split':'val','input_limits':{'strokes':128,'points':12000,'candidates':900,'context_strokes':16},'research_only':True}
        (ROOT/'artifacts/selected.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    with (ROOT/'reports/rsi-ledger.jsonl').open('a') as f:f.write(json.dumps(ledger,ensure_ascii=False)+'\n')
    print('本轮选择',best['name'],best['edge_mode'],'晋级' if a.promote else '仅评测',flush=True)

if __name__=='__main__':main()

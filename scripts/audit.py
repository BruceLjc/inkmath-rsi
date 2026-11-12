"""文件职责：审计数据集隔离、难负例来源及真实推理笔画覆盖，防止评测泄漏和漏笔。"""
import sys,json,collections,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from inkmath.recognizer import Recognizer
from inkmath.layout import canonical
ROOT=Path(__file__).resolve().parents[1]

def main():
    """以完整划分和训练来源指纹验证隔离；演示样本检查输入每笔恰好被消费一次。"""
    records=[json.loads(s) for s in (ROOT/'data/processed/expressions.jsonl').read_text().splitlines()]
    fingerprints={};groups={};counts=collections.Counter()
    for r in records:
        assert r['fingerprint'] not in fingerprints,'发现重复笔迹'
        fingerprints[r['fingerprint']]=r['split'];counts[r['split']]+=1
        assert groups.setdefault(r['group'],r['split'])==r['split'],'同一作者/会话跨划分'
    provenance=json.loads((ROOT/'reports/hard-negative-provenance.json').read_text())
    assert provenance['split']=='train'
    assert all(fingerprints[f]=='train' for f in provenance['unique_source_fingerprints'])
    rec=Recognizer();checks=[]
    for path in sorted((ROOT/'examples').glob('*.json')):
        r=json.loads(path.read_text());prediction=rec.recognize(r['strokes']);used=[i for s in prediction['symbols'] for i in s['stroke_ids']]
        assert sorted(used)==list(range(len(r['strokes']))),'漏笔或重复消费'
        checks.append({'example':path.name,'coverage_exact':True,'example_exact':canonical(prediction['latex'])==canonical(r['expected_latex']),'candidate_count':prediction['candidate_count']})
    report={'split_counts':dict(counts),'unique_fingerprints':len(fingerprints),'disjoint_group_count':len(groups),'hard_negatives_train_only':True,'example_coverage':checks,'source_files_have_chinese_responsibility':all('文件职责' in p.read_text() for folder in ['inkmath','scripts','tests'] for p in (ROOT/folder).glob('*.py'))}
    assert report['source_files_have_chinese_responsibility']
    (ROOT/'reports/audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()

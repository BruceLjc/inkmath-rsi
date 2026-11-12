"""文件职责：测量真实笔迹及接近输入上限的压力样本，区分模型冷加载、热推理与进程启动。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,time,subprocess
import numpy as np
from inkmath.recognizer import Recognizer
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--checkpoint',default=None);p.add_argument('--name',default='benchmark');p.add_argument('--repeat',type=int,default=8);a=p.parse_args();begin=time.perf_counter();rec=Recognizer(a.checkpoint);load=(time.perf_counter()-begin)*1000
sample=json.loads((ROOT/'examples/integral.json').read_text())['strokes'];cases={'real_integral':sample}
# 压力样本不是准确率基准，只检验笔画/采样点上限的计算成本与稳定性。
for count in [32,64,128]:
    cases[f'stress_{count}_sparse']=[[[i*20+float(t),float(8*np.sin(t/3))] for t in np.linspace(0,12,(12000//count+(i<12000%count)) if count==128 else 80)] for i in range(count)]
    cases[f'stress_{count}_overlap']=[[[float(50+20*np.cos(t+i)),float(50+20*np.sin(t+i))] for t in np.linspace(0,2*np.pi,(12000//count+(i<12000%count)) if count==128 else 80)] for i in range(count)]
cases['stress_1_dense_sampling']=[[[float(t),float(np.sin(t)*20)] for t in np.linspace(0,100,12000)]]
rows=[]
for name,strokes in cases.items():
    times=[];errors=[]
    for i in range(a.repeat):
        try:r=rec.recognize(strokes,task_id=f'bench-{name}-{i}');times.append(r['timing_ms']['total'])
        except Exception as e:errors.append(str(e))
    row={'case':name,'strokes':len(strokes),'points':sum(map(len,strokes)),'runs':len(times),'p50_ms':float(np.percentile(times,50)) if times else None,'p99_ms':float(np.percentile(times,99)) if times else None,'max_ms':max(times) if times else None,'over_1s':sum(t>=1000 for t in times),'errors':errors};rows.append(row);print(json.dumps(row),flush=True)
# 完整进程时间包含 Python/PyTorch 导入，避免只报告 main() 之后的局部冷启动。
tick=time.perf_counter();p=subprocess.run([sys.executable,'-m','inkmath',str(ROOT/'examples/integral.json'),]+(['--checkpoint',a.checkpoint] if a.checkpoint else []),cwd=ROOT,capture_output=True,text=True);process_ms=(time.perf_counter()-tick)*1000
result={'checkpoint':a.checkpoint,'model':rec.version,'model_load_warmup_ms':load,'full_cli_process_ms':process_ms,'cli_exit_code':p.returncode,'stress_is_not_accuracy_test':True,'cases':rows}
(ROOT/f'reports/{a.name}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print('full_cli_process_ms',process_ms,flush=True)

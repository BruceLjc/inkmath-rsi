"""文件职责：启动临时本地服务验证真实 HTTP 识别与错误契约，结束后回收服务进程。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,socket,subprocess,time,urllib.request,urllib.error
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--checkpoint');a=p.parse_args()
sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close();base=f'http://127.0.0.1:{port}';logs=(ROOT/'reports/api-server.log').open('w')
proc=subprocess.Popen([sys.executable,'-m','inkmath.server','--port',str(port)]+(['--checkpoint',a.checkpoint] if a.checkpoint else []),cwd=ROOT,stdout=logs,stderr=logs)
def request(path,obj=None):
    data=json.dumps(obj).encode() if obj is not None else None
    req=urllib.request.Request(base+path,data=data,headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=10) as r:return r.status,json.load(r)
    except urllib.error.HTTPError as e:return e.code,json.load(e)
try:
    for _ in range(100):
        try:
            status,health=request('/health');break
        except (OSError,urllib.error.URLError):
            if proc.poll() is not None:raise RuntimeError('临时服务未正常启动')
            time.sleep(.05)
    else:raise RuntimeError('临时服务启动超时')
    sample=json.loads((ROOT/'examples/integral.json').read_text());status,result=request('/recognize',{'task_id':'smoke-integral','revision':7,'strokes':sample['strokes'],'latex':'MUST_NOT_BE_USED'})
    assert status==200 and result['task_id']=='smoke-integral' and result['revision']==7 and result['latex']!='MUST_NOT_BE_USED'
    checks=[]
    for payload in [[],{'strokes':[]},{'strokes':[[[True,1]]]},{'strokes':sample['strokes'],'revision':-1},{'strokes':[[[0,0]]]*129}]:
        status,error=request('/recognize',payload);assert status==400,(status,error);checks.append({'status':status,'error':error['error']})
    report={'health':health,'recognition':result,'expected_latex':sample['expected_latex'],'invalid_input_checks':checks,'process_reaped':False}
finally:
    proc.terminate()
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)
    logs.close()
report['process_reaped']=proc.poll() is not None
(ROOT/'reports/api-smoke.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({'model':result['model'],'latex':result['latex'],'server_total_ms':result['server_total_ms'],'invalid_checks':len(checks),'process_reaped':report['process_reaped']},ensure_ascii=False))

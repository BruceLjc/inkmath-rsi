"""文件职责：以真实 HTTP 检查体积/媒体类型限制和并发忙返回，验证常驻服务的时间边界。"""
import json,socket,subprocess,sys,time,threading,urllib.request,urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    """创建本机临时服务并回收；不连接外部地址，不将合成压力笔迹计入准确率。"""
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close();base=f'http://127.0.0.1:{port}'
    log=(ROOT/'reports/api-limits-server.log').open('w');proc=subprocess.Popen([sys.executable,'-m','inkmath.server','--port',str(port)],cwd=ROOT,stdout=log,stderr=log)
    def post(body,content='application/json'):
        started=time.perf_counter();req=urllib.request.Request(base+'/recognize',data=body,headers={'Content-Type':content})
        try:
            with urllib.request.urlopen(req,timeout=10) as response:return response.status,json.load(response),(time.perf_counter()-started)*1000
        except urllib.error.HTTPError as error:return error.code,json.load(error),(time.perf_counter()-started)*1000
    try:
        for _ in range(100):
            try:
                with urllib.request.urlopen(base+'/health',timeout=1) as response:health=json.load(response)
                break
            except OSError:
                if proc.poll() is not None:raise RuntimeError('临时服务启动失败')
                time.sleep(.05)
        else:raise RuntimeError('临时服务启动超时')
        checks=[]
        for name,body,media,expected in [('oversize',b' '*524289,'application/json',413),('wrong_media',b'{}','text/plain',415),('broken_json',b'{','application/json',400)]:
            status,result,ms=post(body,media);assert status==expected,(name,status,result);checks.append({'case':name,'status':status,'round_trip_ms':ms})
        sample=json.loads((ROOT/'examples/integral.json').read_text());barrier=threading.Barrier(4)
        def concurrent(index):
            barrier.wait();task='concurrent-'+str(index);body=json.dumps({'strokes':sample['strokes'],'task_id':task,'revision':index}).encode();status,result,ms=post(body);assert result['task_id']==task
            if status==200:assert result['revision']==index
            return {'task_id':task,'status':status,'round_trip_ms':ms}
        with ThreadPoolExecutor(max_workers=4) as pool:concurrency=list(pool.map(concurrent,range(4)))
        statuses=[r['status'] for r in concurrency];assert 200 in statuses and 429 in statuses and set(statuses)<={200,429},statuses
        report={'health':health,'rejection_checks':checks,'concurrency':concurrency,'all_concurrent_calls_returned':True}
    finally:
        proc.terminate()
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)
        log.close()
    report['process_reaped']=proc.poll() is not None;(ROOT/'reports/api-limits.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()

"""文件职责：提供仅绑定本机的常驻数学笔迹识别 HTTP 接口，模型预热后处理坐标请求。"""
import argparse,json,threading,time,uuid
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from .recognizer import Recognizer


def main():
    """启动本地实验服务；请求体和笔画规模有界，推理串行避免线程竞争。"""
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=18765);p.add_argument('--checkpoint');a=p.parse_args()
    recognizer=Recognizer(a.checkpoint);lock=threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        """处理健康检查和坐标识别，不暴露目录、文件内容或远程访问。"""
        def respond(self,status,obj):
            """统一返回 UTF-8 JSON，错误也保留 task_id。"""
            body=json.dumps(obj,ensure_ascii=False,allow_nan=False).encode();self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            """只开放模型状态，不伪造识别质量或业务成功。"""
            if self.path!='/health':return self.respond(404,{'error':'not_found'})
            self.respond(200,{'status':'ready','model':recognizer.version,'parameters':recognizer.parameters,'input':'stroke coordinates','max_strokes':128,'max_points':12000,'accuracy_guaranteed':False})
        def do_POST(self):
            """校验输入后串行识别；队列忙时立即拒绝，不把排队隐藏在一秒统计外。"""
            task_id=str(uuid.uuid4());start=time.perf_counter()
            if self.path!='/recognize':return self.respond(404,{'task_id':task_id,'error':'not_found'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if length<=0 or length>512*1024:return self.respond(413,{'task_id':task_id,'error':'请求体需在 1～524288 字节之间'})
                if 'application/json' not in self.headers.get('Content-Type',''):return self.respond(415,{'task_id':task_id,'error':'需要 application/json'})
                obj=json.loads(self.rfile.read(length))
                if not isinstance(obj,dict):raise ValueError('JSON 请求体必须为对象')
                task_id=str(obj.get('task_id',task_id))[:128];revision=obj.get('revision',0)
                if not isinstance(revision,int) or isinstance(revision,bool) or revision<0:raise ValueError('revision 必须是非负整数')
                if not lock.acquire(blocking=False):return self.respond(429,{'task_id':task_id,'error':'识别器忙，请稍后重试；本次未完成识别'})
                try:result=recognizer.recognize(obj['strokes'],task_id=task_id,revision=revision)
                finally:lock.release()
                result['server_total_ms']=round((time.perf_counter()-start)*1000,3);self.respond(200,result)
            except (ValueError,KeyError,TypeError,OverflowError) as e:self.respond(400,{'task_id':task_id,'status':'error','error':str(e)})
            except Exception:
                self.respond(500,{'task_id':task_id,'status':'error','error':'本地识别异常，请检查服务日志'});raise
        def log_message(self,format,*args):
            """只打印接口访问摘要，不将学生笔迹内容写入日志。"""
            print(format%args,flush=True)
    server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler);print(f'本地模型 {recognizer.version} 已预热：http://127.0.0.1:{a.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
if __name__=='__main__':main()

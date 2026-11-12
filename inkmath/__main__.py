"""文件职责：提供笔迹文件识别命令行，输出可追溯的 JSON 结果及冷启动耗时。"""
import argparse,json,time,sys
from .ink import load_input
from .recognizer import Recognizer

def main():
    """读取笔迹并真实执行识别，非法输入以非零退出码返回。"""
    p=argparse.ArgumentParser(description='本地手写数学坐标 → LaTeX');p.add_argument('input');p.add_argument('--checkpoint');p.add_argument('--task-id',default='cli');a=p.parse_args();start=time.perf_counter()
    try:
        strokes=load_input(a.input);r=Recognizer(a.checkpoint);result=r.recognize(strokes,task_id=a.task_id);result['cold_load_and_recognition_ms']=round((time.perf_counter()-start)*1000,3);print(json.dumps(result,ensure_ascii=False,indent=2))
    except (ValueError,KeyError,OSError) as e:
        print(json.dumps({'task_id':a.task_id,'status':'error','error':str(e)},ensure_ascii=False),file=sys.stderr);sys.exit(2)
if __name__=='__main__':main()

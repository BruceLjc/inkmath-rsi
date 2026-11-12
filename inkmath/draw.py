"""文件职责：提供原生 Matplotlib 笔迹实验窗口，采集分笔坐标、执行本地识别并保存人工标注。"""
import argparse,json,threading,time,uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from .ink import box
from .recognizer import Recognizer
ROOT=Path(__file__).resolve().parents[1]


class InkWindow:
    """研究用原生笔迹采集器；模型结果始终标为候选，不把它自动当训练真值。"""
    def __init__(self,recognizer):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button,TextBox
        from matplotlib import font_manager
        font=Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
        if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
        plt.rcParams['axes.unicode_minus']=False
        self.plt=plt;self.recognizer=recognizer;self.strokes=[];self.active=None;self.revision=0;self.future=None;self.last_result=None;self.executor=ThreadPoolExecutor(max_workers=1);self.saved=None
        self.fig=plt.figure(figsize=(12,7.5),facecolor='white');self.fig.canvas.manager.set_window_title('InkMath 本地手写实验')
        self.fig.text(.06,.95,'手写数学 → LaTeX',fontsize=21,weight='bold');self.fig.text(.06,.91,'仅在本机处理笔迹。函数、极限、积分、求和；输出是待核对候选。',fontsize=11,color='#53606f')
        self.ax=self.fig.add_axes([.06,.37,.88,.49]);self.ax.set_xlim(0,1000);self.ax.set_ylim(500,0);self.ax.set_xticks([]);self.ax.set_yticks([]);self.ax.set_facecolor('#fbfcfe')
        for spine in self.ax.spines.values():spine.set_color('#cad2dc')
        self.raw=self.fig.text(.06,.29,'LaTeX 候选：等待手写',fontsize=12,fontfamily=['DejaVu Sans Mono',plt.rcParams['font.family'][0]],wrap=True)
        self.rendered=self.fig.text(.06,.20,'',fontsize=21)
        self.status=self.fig.text(.06,.14,'按住鼠标或手写设备作画；点击“识别”提交当前笔迹。',fontsize=10,color='#53606f')
        self.buttons=[]
        for x,label,callback in [(.06,'识别',self.recognize),(.18,'撤销一笔',self.undo),(.30,'清空',self.clear),(.42,'载入例子',self.example),(.54,'保存样本',self.save)]:
            b=Button(self.fig.add_axes([x,.035,.105,.055]),label,color='#edf2f7',hovercolor='#dbeafe');b.on_clicked(callback);self.buttons.append(b)
        self.label=TextBox(self.fig.add_axes([.77,.035,.17,.055]),'人工标签 ',initial='');self.label.label.set_fontsize(10)
        self.fig.canvas.mpl_connect('button_press_event',self.press);self.fig.canvas.mpl_connect('motion_notify_event',self.move);self.fig.canvas.mpl_connect('button_release_event',self.release);self.fig.canvas.mpl_connect('close_event',self.close)
        self.timer=self.fig.canvas.new_timer(interval=50);self.timer.add_callback(self.poll);self.timer.start();self.sample_index=0

    def press(self,event):
        """一次按下开始一笔，保留用户原始采样而不拼接不同笔。"""
        if event.inaxes is self.ax and event.button==1:
            self.active=[[float(event.xdata),float(event.ydata)]];self.revision+=1;self.last_result=None;self.label.set_val('');self.raw.set_text('LaTeX 候选：笔迹已修改');self.rendered.set_text('');self.status.set_text('写完后点击识别。')
            self.current_line,=self.ax.plot([event.xdata],[event.ydata],color='#18283a',linewidth=2.1,solid_capstyle='round');self.fig.canvas.draw_idle()
    def move(self,event):
        """仅在画布内采样；轻微移动去重，防止重复事件膨胀输入。"""
        if self.active is None or event.inaxes is not self.ax:return
        p=[float(event.xdata),float(event.ydata)]
        if np.linalg.norm(np.array(p)-self.active[-1])<.8:return
        self.active.append(p);self.current_line.set_data(*np.array(self.active).T);self.fig.canvas.draw_idle()
    def release(self,event):
        """抬笔后提交一笔，识别由显式按钮触发，不把停笔等待隐藏在耗时里。"""
        if self.active is not None:self.strokes.append(self.active);self.active=None
    def redraw(self):
        """撤销或载入样本后重绘全部已有线条，保留笔画分隔。"""
        for line in list(self.ax.lines):line.remove()
        for s in self.strokes:self.ax.plot(*np.array(s).T,color='#18283a',linewidth=2.1,solid_capstyle='round')
        self.fig.canvas.draw_idle()
    def clear(self,event=None):
        """只清除当前窗口笔迹，不删除已经保存的用户数据。"""
        self.strokes=[];self.active=None;self.revision+=1;self.last_result=None;self.label.set_val('');self.raw.set_text('LaTeX 候选：等待手写');self.rendered.set_text('');self.status.set_text('画布已清空。');self.redraw()
    def undo(self,event=None):
        """撤销最后一笔并使已有识别结果失效。"""
        if self.strokes:self.strokes.pop();self.revision+=1;self.last_result=None;self.label.set_val('');self.raw.set_text('LaTeX 候选：笔迹已修改');self.rendered.set_text('');self.status.set_text('已撤销一笔，请重新识别。');self.redraw()
    def example(self,event=None):
        """载入明确标记为验证集演示的真实笔迹，标签只供人工核对。"""
        files=sorted((ROOT/'examples').glob('*.json'),key=lambda p:({'integral':0,'sum':1,'limit':2,'function':3,'fraction':4,'root':5}.get(p.stem,9),p.stem))
        if not files:return
        p=files[self.sample_index%len(files)];self.sample_index+=1;sample=json.loads(p.read_text());strokes=sample['strokes'];b=box(strokes);scale=min(900/max(b[2]-b[0],1e-6),400/max(b[3]-b[1],1e-6));self.strokes=[[[float((x-b[0])*scale+45),float((y-b[1])*scale+45)] for x,y in s] for s in strokes];self.revision+=1;self.last_result=None;self.raw.set_text('LaTeX 候选：等待识别');self.rendered.set_text('');self.label.set_val(sample.get('expected_latex',''));self.status.set_text('演示样本：'+p.stem+'（验证集筛选例子，非盲测）');self.redraw()
    def recognize(self,event=None):
        """后台识别当前版本，避免计算阻塞原生窗口；新笔迹会使旧结果失效。"""
        if self.future and not self.future.done():self.status.set_text('上一条正在处理，请稍后点击识别。');self.fig.canvas.draw_idle();return
        snapshot=json.loads(json.dumps(self.strokes));revision=self.revision;self.submitted=time.perf_counter();self.status.set_text('正在识别当前笔迹…');self.fig.canvas.draw_idle()
        self.future=self.executor.submit(self.recognizer.recognize,snapshot,'draw-'+str(uuid.uuid4()),revision)
    def poll(self):
        """只在主线程更新窗口，忽略被后续编辑覆盖的旧版本结果。"""
        if not self.future or not self.future.done():return
        future=self.future;self.future=None
        try:
            result=future.result()
            if result['revision']!=self.revision:self.status.set_text('笔迹已修改，旧结果已忽略，请重新识别。');return
            self.last_result=result;self.raw.set_text('LaTeX 候选：'+result['latex']);self.rendered.set_text('$'+result['latex']+'$')
            try:self.fig.canvas.draw()
            except (ValueError,RuntimeError):self.rendered.set_text('当前公式超出预览排版能力，请核对上方原始 LaTeX。')
            visible_ms=(time.perf_counter()-self.submitted)*1000;self.status.set_text(f"模型 {result['timing_ms']['total']:.0f} ms · 点击到显示 {visible_ms:.0f} ms · {result['status']} · 需人工核对")
        except Exception as e:self.status.set_text('未完成识别：'+str(e));self.rendered.set_text('')
        self.fig.canvas.draw_idle()
    def save(self,event=None):
        """将原始坐标和人工输入的标签另存为新文件，模型候选与人工标签分别保存。"""
        directory=ROOT/'captures';directory.mkdir(exist_ok=True);path=directory/(time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]+'.json');obj={'strokes':self.strokes,'human_label':self.label.text.strip(),'prediction':self.last_result,'label_verified':False};path.write_text(json.dumps(obj,ensure_ascii=False,indent=2));self.saved=path;self.status.set_text('已保存 '+str(path.name)+'；样本不会自动进入训练。');self.fig.canvas.draw_idle()
    def close(self,event=None):
        """回收实验窗口的计时器和后台线程，不终止其他项目进程。"""
        self.timer.stop();self.executor.shutdown(wait=False,cancel_futures=True)


def main():
    """启动手写实验窗口，或以无界面模式验证笔迹载入、推理和排版流程。"""
    p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.smoke:
        import matplotlib
        matplotlib.use('Agg')
    app=InkWindow(Recognizer(a.checkpoint))
    if a.smoke:
        app.example();app.recognize()
        while app.future and not app.future.done():time.sleep(.02)
        app.poll();app.fig.savefig(ROOT/'reports/draw-smoke.png',dpi=140);app.close();app.plt.close(app.fig)
        if not app.last_result:raise RuntimeError('画板识别冒烟测试失败')
        print(json.dumps({'latex':app.last_result['latex'],'timing_ms':app.last_result['timing_ms']},ensure_ascii=False))
    else:app.plt.show()
if __name__=='__main__':main()

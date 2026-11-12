"""文件职责：按固定类别顺序呈现预留测试笔迹及真实候选，不依据正确与否筛选展示。"""
import json,textwrap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    """每类取报告中首条记录，图像只是错误分析，不用于替代全量准确率。"""
    report=json.loads((ROOT/'reports/final-test.json').read_text());records={r['id']:r for r in map(json.loads,(ROOT/'data/processed/expressions.jsonl').read_text().splitlines())}
    font=Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    categories=['basic','function','limit','integral','sum','fraction','root','superscript'];chosen=[];used=set()
    for c in categories:
        row=next((r for r in report['rows'] if c in r['categories'] and r['id'] not in used),None)
        if row:chosen.append((c,row));used.add(row['id'])
    fig,axes=plt.subplots(4,2,figsize=(16,20));fig.suptitle('预留测试笔迹：每类首例，未按正确与否筛选',fontsize=18)
    for ax in axes.ravel():ax.axis('off')
    for ax,(category,r) in zip(axes.ravel(),chosen):
        for stroke in records[r['id']]['strokes']:
            p=np.array(stroke);ax.plot(p[:,0],p[:,1],color='#18283a',linewidth=1.7)
        ax.set_aspect('equal',adjustable='datalim');ax.invert_yaxis();ax.margins(.15)
        ax.set_title(category+' · '+('整式正确' if r['correct'] else '识别错误')+f" · {r.get('ms',0):.0f} ms",color='#166534' if r['correct'] else '#b91c1c',fontsize=12)
        lines=textwrap.wrap('真值: '+r['target'],width=68)+textwrap.wrap('候选: '+r.get('prediction',r.get('error','')),width=68)
        ax.text(0,-.04,'\n'.join(lines),transform=ax.transAxes,fontsize=8,va='top')
    fig.subplots_adjust(hspace=1.05,wspace=.25,top=.94,bottom=.1);fig.savefig(ROOT/'reports/test-cases.png',dpi=130);plt.close(fig)
    (ROOT/'reports/test-cases-selection.json').write_text(json.dumps({'method':'each category first row; no correctness filtering','ids':[r['id'] for c,r in chosen]},ensure_ascii=False,indent=2))

if __name__=='__main__':main()

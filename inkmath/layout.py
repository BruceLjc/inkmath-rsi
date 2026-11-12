"""文件职责：把数学符号及二维位置解析为可追溯结构树，再确定性导出 LaTeX。"""
from dataclasses import dataclass, field
import re
import xml.etree.ElementTree as ET
import numpy as np


@dataclass
class Node:
    """数学布局节点；笔画 ID 只用于追溯，不携带人工识别答案。"""
    label: str
    box: list
    indices: list = field(default_factory=list)
    confidence: float = 1.0
    kind: str = 'symbol'
    children: list = field(default_factory=list)

    def as_dict(self):
        """导出可审计结构，保留候选符号来源和位置。"""
        return {'kind':self.kind,'label':self.label,'box':self.box,'stroke_ids':self.indices,'score':round(self.confidence,4),'children':[n.as_dict() for n in self.children]}


def union(nodes):
    """计算结构节点的联合边界。"""
    return [min(n.box[0] for n in nodes),min(n.box[1] for n in nodes),max(n.box[2] for n in nodes),max(n.box[3] for n in nodes)]


def make(kind, children, label=''):
    """聚合子结构，不把输出符号当作后续训练的真值。"""
    return Node(label,union(children),sorted(set(i for n in children for i in n.indices)),min(n.confidence for n in children),kind,children)


def width(n): return max(n.box[2]-n.box[0],1e-6)
def height(n): return max(n.box[3]-n.box[1],1e-6)
def cx(n): return (n.box[0]+n.box[2])/2
def cy(n): return (n.box[1]+n.box[3])/2


def latex(n):
    """将布局树序列化为 LaTeX；保留学生表达式，不执行求值或数学纠错。"""
    c=[latex(v) for v in n.children]
    if n.kind=='row': return ' '.join(c)
    if n.kind=='frac': return r'\frac{'+c[0]+'}{'+c[1]+'}'
    if n.kind=='sqrt': return r'\sqrt{'+c[0]+'}'
    if n.kind=='root': return r'\sqrt['+c[0]+']{'+c[1]+'}'
    if n.kind=='sup': return c[0]+'^{'+c[1]+'}'
    if n.kind=='sub': return c[0]+'_{'+c[1]+'}'
    if n.kind=='subsup': return c[0]+'_{'+c[1]+'}^{'+c[2]+'}'
    return {'\\lt':'<','\\gt':'>','\\prime':"'"}.get(n.label,n.label)


def parse_layout(symbols, depth=0):
    """有界递归解析分式、根式、上下标及大运算符上下限；首版不解析矩阵和分段。"""
    nodes=list(symbols)
    if not nodes: return Node('',[0,0,0,0],kind='row')
    if depth>12: return make('row',sorted(nodes,key=lambda n:n.box[0]))
    hmed=float(np.median([height(n) for n in nodes if n.label not in ['-','=']]) if any(n.label not in ['-','='] for n in nodes) else 1)
    # 横线只有在上、下均有内容时才解释为分数，避免把减号一律转换为分式。
    for bar in sorted(nodes,key=width,reverse=True):
        if not any(bar is n for n in nodes) or bar.kind!='symbol' or bar.label!='-' or width(bar)<.6*hmed: continue
        pad=.12*hmed
        overlap=lambda n: min(n.box[2],bar.box[2]+pad)-max(n.box[0],bar.box[0]-pad)>.35*min(width(n),width(bar))
        above=[n for n in nodes if n is not bar and overlap(n) and n.box[3]<cy(bar)+.06*hmed and cy(bar)-n.box[3]<3*hmed]
        below=[n for n in nodes if n is not bar and overlap(n) and n.box[1]>cy(bar)-.06*hmed and n.box[1]-cy(bar)<3*hmed]
        if above and below:
            num=parse_layout(above,depth+1);den=parse_layout(below,depth+1);frac=make('frac',[num,den]);frac.box=union([frac,bar]);frac.indices+=bar.indices
            chosen={id(n) for n in above+below+[bar]};nodes=[n for n in nodes if id(n) not in chosen]+[frac]
    # 根号横线的包围盒含被开方区域；只取横线下且位于勾右侧的符号。
    for root in sorted(list(nodes),key=width,reverse=True):
        if root not in nodes or root.label!=r'\sqrt' or root.kind!='symbol': continue
        inside=[n for n in nodes if n is not root and cx(n)>root.box[0]+.22*width(root) and cx(n)<root.box[2]+.12*hmed and cy(n)>root.box[1]-.08*hmed and cy(n)<root.box[3]+.15*hmed]
        if inside:
            rad=make('sqrt',[parse_layout(inside,depth+1)]);rad.box=union([root,rad]);rad.indices+=root.indices
            chosen={id(n) for n in inside+[root]};nodes=[n for n in nodes if id(n) not in chosen]+[rad]
    nodes.sort(key=lambda n:(n.box[0],n.box[1]))
    # 大运算符下标/上标可横跨运算符左右，不能按普通字母的右上角规则处理。
    for op in list(nodes):
        if not any(op is n for n in nodes) or op.label not in [r'\sum',r'\prod',r'\int',r'\lim']: continue
        subs=[];sups=[]
        for n in nodes:
            if n is op:continue
            near=abs(cx(n)-cx(op))<max(width(op)*.85,hmed*.85)
            if not near:continue
            if op.label==r'\int':
                if height(n)<height(op)*.7 and cy(n)<op.box[1]+.2*height(op): sups.append(n)
                elif height(n)<height(op)*.7 and cy(n)>op.box[3]-.2*height(op): subs.append(n)
            else:
                if n.box[3]<op.box[1]+.2*height(op): sups.append(n)
                elif n.box[1]>op.box[3]-.15*height(op): subs.append(n)
        if subs or sups:
            kids=[op]+([parse_layout(subs,depth+1)] if subs else [])+([parse_layout(sups,depth+1)] if sups else [])
            new=make('subsup' if subs and sups else 'sub' if subs else 'sup',kids)
            chosen={id(n) for n in [op]+subs+sups};nodes=[n for n in nodes if id(n) not in chosen]+[new]
    nodes.sort(key=lambda n:(n.box[0],n.box[1]));output=[];i=0
    while i<len(nodes):
        base=nodes[i];i+=1;sup=[];sub=[]
        if base.label in ['+','-','=','<','>',r'\leq',r'\geq',',',r'\rightarrow']:
            output.append(base);continue
        while i<len(nodes):
            n=nodes[i];gap=n.box[0]-base.box[2]
            if gap>max(.7*hmed,.6*width(base)) or n.box[0]<base.box[0]+.3*width(base):break
            # 使用底边差和相对尺寸约束，区分 x² 与同行 x2；已形成的复合节点也可作脚标。
            small=height(n)<height(base)*.88 or n.kind!='symbol'
            if small and n.box[3]<base.box[3]-.32*height(base) and cy(n)<cy(base):sup.append(n);i+=1
            elif small and n.box[1]>base.box[1]+.48*height(base) and cy(n)>cy(base):sub.append(n);i+=1
            else:break
        if sup or sub:
            kids=[base]+([parse_layout(sub,depth+1)] if sub else [])+([parse_layout(sup,depth+1)] if sup else [])
            base=make('subsup' if sub and sup else 'sub' if sub else 'sup',kids)
        output.append(base)
    return make('row',output)


def mathml_latex(source):
    """将数据集 MathML 真值独立序列化，仅用于评测，绝不进入实际识别输入。"""
    def visit(n):
        tag=n.tag.split('}')[-1];c=[visit(x) for x in n]
        if tag in ['mi','mn','mo','mtext']: return normalize_label((n.text or '').strip())
        if tag=='mfrac':return r'\frac{'+c[0]+'}{'+c[1]+'}'
        if tag=='msqrt':return r'\sqrt{'+' '.join(c)+'}'
        if tag=='mroot':return r'\sqrt['+c[1]+']{'+c[0]+'}'
        if tag in ['msup','mover']:return c[0]+'^{'+c[1]+'}'
        if tag in ['msub','munder']:return c[0]+'_{'+c[1]+'}'
        if tag in ['msubsup','munderover']:return c[0]+'_{'+c[1]+'}^{'+c[2]+'}'
        return ' '.join(c)
    return visit(ET.fromstring(source))


def normalize_label(s):
    """统一研究数据中的函数命令写法，不进行代数等价化。"""
    return '\\'+s if s in ['sin','cos','tan','log','ln','lim','int','sum','prod','exp'] else s


def canonical(s):
    """消除排版空白和同义命令，保留分数、括号、上下标等结构差异。"""
    for a,b in [(r'\left',''),(r'\right',''),(r'\limits',''),(r'\dfrac',r'\frac'),(r'\lt','<'),(r'\gt','>'),(r'\prime',"'"),(r'\,',''),(r'\!','')]:s=s.replace(a,b)
    return re.sub(r'\s+','',s).strip('$')

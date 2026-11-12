"""文件职责：安全读取分笔坐标和 InkML，并将线条标准化为模型输入；不读取图片。"""
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
NS = {'i': 'http://www.w3.org/2003/InkML', 'm': 'http://www.w3.org/1998/Math/MathML'}
XMLID = '{http://www.w3.org/XML/1998/namespace}id'


def box(strokes):
    """计算一组笔画的包围盒，保留公式中的相对位置。"""
    a = np.concatenate([np.asarray(s, dtype=np.float32)[:, :2] for s in strokes])
    return [float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())]


def read_inkml(content):
    """解析公开研究数据的坐标和标注；生产识别只消费 strokes。"""
    if b'<!DOCTYPE' in content or b'<!ENTITY' in content:
        raise ValueError('不允许 XML 外部实体')
    root = ET.fromstring(content)
    annotations = {a.get('type'): (a.text or '').strip() for a in root.findall('i:annotation', NS)}
    ids, strokes = [], []
    for tr in root.findall('i:trace', NS):
        pts = [[float(v) for v in p.strip().split()[:2]] for p in (tr.text or '').strip().split(',') if p.strip()]
        if pts:
            ids.append(tr.get('id')); strokes.append(pts)
    idmap = {v: k for k, v in enumerate(ids)}
    symbols = []
    for group in root.findall('.//i:traceGroup', NS):
        refs = group.findall('i:traceView', NS)
        truth = group.find("i:annotation[@type='truth']", NS)
        if refs and truth is not None:
            indices = [idmap[t.get('traceDataRef')] for t in refs if t.get('traceDataRef') in idmap]
            if indices:
                ref = group.find('i:annotationXML', NS)
                symbols.append({'label': (truth.text or '').strip(), 'indices': indices, 'box': box([strokes[i] for i in indices]), 'ref': ref.get('href') if ref is not None else None})
    tree = root.find('.//m:math', NS)
    return {'strokes': strokes, 'symbols': symbols, 'latex': annotations.get('truth', '').strip('$ ').strip(), 'writer': annotations.get('writer'), 'mathml': ET.tostring(tree, encoding='unicode') if tree is not None else None}


def validate_strokes(strokes, max_strokes=128, max_points=12000):
    """限制计算规模，拒绝空笔画、非有限坐标及超范围输入，避免无界解析。"""
    if not isinstance(strokes, list) or not 1 <= len(strokes) <= max_strokes:
        raise ValueError(f'需要 1～{max_strokes} 笔')
    total = 0
    clean = []
    for s in strokes:
        if isinstance(s, dict):
            if len(s.get('x', [])) != len(s.get('y', [])):
                raise ValueError('x/y 数组长度必须一致')
            s = list(zip(s['x'], s['y']))
        if not isinstance(s, (list, tuple)) or not s:
            raise ValueError('不能包含空笔画')
        points = []
        for p in s:
            if not isinstance(p, (list, tuple)) or len(p) < 2 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and abs(v) < 1e9 for v in p[:2]):
                raise ValueError('坐标必须是有限数字')
            points.append([float(p[0]), float(p[1])])
        total += len(points)
        if total > max_points:
            raise ValueError(f'采样点上限为 {max_points}')
        clean.append(points)
    return clean


def load_input(path):
    """读取坐标 JSON 或 InkML；JSON 中的人工标签不会送给识别器。"""
    p = Path(path)
    if p.suffix.lower() == '.inkml':
        return validate_strokes(read_inkml(p.read_bytes())['strokes'])
    obj = json.loads(p.read_text())
    return validate_strokes(obj['strokes'] if isinstance(obj, dict) else obj)


def raster(strokes, size=32, bounds=None):
    """把已有线条绘成内部小型特征图，不涉及摄像头、图片 OCR 或手写笔硬件。"""
    b = bounds if bounds is not None else box(strokes); w, h = b[2]-b[0], b[3]-b[1]
    scale = (size-7) / max(w, h, 1e-6)
    ox, oy = (size-w*scale)/2, (size-h*scale)/2
    im = Image.new('L', (size, size)); d = ImageDraw.Draw(im)
    for s in strokes:
        pts = (np.asarray(s,dtype=np.float64)[:,:2]-np.array([b[0],b[1]]))*scale+np.array([ox,oy])
        if len(pts) == 1:
            x,y=pts[0]; d.ellipse((x-1,y-1,x+1,y+1), fill=255)
        else:
            d.line(pts.ravel().tolist(), fill=255, width=2, joint='curve')
    # 形状比例是区分竖线、短横、点等字符的必要信息，笔数辅助多笔组合。
    geom = np.array([w/max(w,h,1e-6), h/max(w,h,1e-6), min(len(strokes),8)/8], dtype=np.float32)
    return np.asarray(im, dtype=np.uint8)[None], geom


def trajectory(strokes, steps=64, bounds=None):
    """按弧长采样分笔轨迹，保留方向和抬笔标记；不依赖压力、摄像头或设备型号。"""
    b=bounds if bounds is not None else box(strokes);scale=max(b[2]-b[0],b[3]-b[1],1e-6);sequences=[]
    lengths=[float(np.linalg.norm(np.diff(np.asarray(s,dtype=np.float32)[:,:2],axis=0),axis=1).sum()) for s in strokes];total=max(sum(lengths),1e-6)
    for s,length in zip(strokes,lengths):
        pts=np.asarray(s,dtype=np.float32)[:,:2];count=max(2,round((steps-len(strokes))*length/total));distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(pts,axis=0),axis=1))]
        if distance[-1]>1e-8:
            q=np.linspace(0,distance[-1],count);res=np.stack([np.interp(q,distance,pts[:,k]) for k in range(2)],1)
        else:res=np.repeat(pts[:1],count,axis=0)
        res=(res-np.array([(b[0]+b[2])/2,(b[1]+b[3])/2]))/scale;delta=np.vstack([np.zeros((1,2)),np.diff(res,axis=0)]);pen=np.ones((len(res),1));pen[0]=0;sequences.append(np.concatenate([res,delta,pen],1))
    a=np.concatenate(sequences);ix=np.linspace(0,len(a)-1,steps).round().astype(int)
    return a[ix].T.astype(np.float32)


def context_raster(strokes, indices, size=32, boxes=None, bounds=None, max_context=16):
    """绘制候选周围未选中的笔迹，先按缓存包围盒筛选，避免重复扫描整页采样点。"""
    selected=set(indices);b=bounds if bounds is not None else box([strokes[i] for i in indices]);w,h=b[2]-b[0],b[3]-b[1];scale=(size-7)/max(w,h,1e-6);ox,oy=(size-w*scale)/2,(size-h*scale)/2
    im=Image.new('L',(size,size));draw=ImageDraw.Draw(im)
    boxes=np.asarray(boxes) if boxes is not None else np.array([box([s]) for s in strokes])
    # 加两像素容差以保留与候选接触的线条；只跳过确定完全不可见的笔画。
    xmin=b[0]-(ox+2)/scale;xmax=b[0]+(size-ox+2)/scale;ymin=b[1]-(oy+2)/scale;ymax=b[1]+(size-oy+2)/scale
    visible=np.where((boxes[:,2]>=xmin)&(boxes[:,0]<=xmax)&(boxes[:,3]>=ymin)&(boxes[:,1]<=ymax))[0]
    visible=np.array([i for i in visible if i not in selected],dtype=int)
    if len(visible)>max_context:
        centers=(boxes[visible,:2]+boxes[visible,2:])/2;center=np.array([(b[0]+b[2])/2,(b[1]+b[3])/2]);distance=((centers-center)**2).sum(1)
        visible=visible[np.argsort(distance,kind='stable')[:max_context]]
    for i in visible:
        a=np.asarray(strokes[i],dtype=np.float64)[:,:2];pts=(a-np.array([b[0],b[1]]))*scale+np.array([ox,oy]);pts=np.clip(pts,-1e6,1e6)
        if len(pts)>1:draw.line(pts.ravel().tolist(),fill=255,width=2,joint='curve')
        else:
            x,y=pts[0];draw.ellipse((x-1,y-1,x+1,y+1),fill=255)
    return np.asarray(im,dtype=np.uint8)[None]

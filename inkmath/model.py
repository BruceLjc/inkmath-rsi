"""文件职责：定义只处理笔迹特征图与几何特征的轻量符号分类网络。"""
import torch
from torch import nn


class SymbolNet(nn.Module):
    """将候选笔画组分类为数学符号或拒识；没有预训练 OCR 或外部 API。"""
    def __init__(self, classes):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(1,24,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(24,48,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(48,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten())
        self.head=nn.Sequential(nn.Linear(64*4*4+3,192),nn.ReLU(),nn.Dropout(.15),nn.Linear(192,classes))

    def forward(self, images, geometry):
        """融合候选形状与长宽比例，保留笔画之间的分组信息。"""
        return self.head(torch.cat([self.features(images),geometry],dim=1))


class TrajectorySymbolNet(nn.Module):
    """联合线条形状与笔顺方向的轻量网络，实验验证轨迹信息是否改善真实笔迹识别。"""
    def __init__(self,classes):
        super().__init__();base=SymbolNet(classes);self.features=base.features;self.features[0]=nn.Conv2d(2,24,3,padding=1)
        self.trace=nn.Sequential(nn.Conv1d(5,32,5,padding=2),nn.ReLU(),nn.Conv1d(32,48,5,padding=2),nn.ReLU(),nn.AdaptiveMaxPool1d(4),nn.Flatten())
        self.head=nn.Sequential(nn.Linear(64*4*4+3+192,192),nn.ReLU(),nn.Dropout(.15),nn.Linear(192,classes))
    def forward(self,images,geometry,trajectories):
        """融合形状、长宽比例、笔画方向和笔边界，输出数学符号分数。"""
        return self.head(torch.cat([self.features(images),geometry,self.trace(trajectories)],dim=1))


class DualSymbolNet(TrajectorySymbolNet):
    """分离数学类别和分组有效性，避免大量拒识样本直接挤压稀有符号类别。"""
    def __init__(self,classes):
        super().__init__(classes);self.valid=nn.Linear(192,1)
    def forward(self,images,geometry,trajectories):
        """类别只在真实符号组上监督，分组有效性则同时监督正例和错合并负例。"""
        features=torch.cat([self.features(images),geometry,self.trace(trajectories)],dim=1);hidden=self.head[:3](features)
        return self.head[3](hidden),self.valid(hidden).squeeze(-1)
    def probabilities(self,images,geometry,trajectories):
        """把独立有效性分数与类别概率合成统一候选分数；拒识类别始终位于最后。"""
        classes,valid=self.forward(images,geometry,trajectories);v=valid.sigmoid().unsqueeze(-1);return torch.cat([classes[:,:-1].softmax(-1)*v,1-v],-1)

import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init(self, S=7,B=2,C=20):
        # S:分割的网格数，B:每个网格预测的边界框数，C:类别数
        super(YoloLoss,self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5     #论文中计算loss函数的常数
        self.lambda_coord = 5       #同上
    

    
    def forward(self, predictions, target):
        
        # 原predictions是一维，通过reshape变成4维，[batch_size,7,7,30]  
        predictions = predictions.reshape(-1, self.S, self.S, self.C+self.B*5) 

        # ... 表示选择所有剩余的维度
        iou_b1 = intersection_over_union(predictions[...,21:25],target[...,21:25])  
        iou_b2 = intersection_over_union(predictions[...,26:30],target[...,21:25])
        ious = torch.cat([iou_b1.unsqueeze(0),iou_b2.unsqueeze(0)],dim=0) 
        iou_maxes, bestbox = torch.max(ious,dim=0)
        exists_box = target[...,20].unsqueeze(3) #Iobj_i #
  
        box_predictions = exists_box * (
            (
                bestbox * predictions[..., 26:30]
                + (1 - bestbox) * predictions[..., 21:25]
            )
        )
        box_targets = exists_box * target[..., 21:25]

        
        
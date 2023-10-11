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
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
    
    def forward(se;f, predictions, target):
        predictions = predictions.reshape(-1,self.S,self.S,self.C+self.B*5)

        iou_b1 = intersection_over_union(predictions[...,21:25],target[...,21:25])
        iou_b2 = intersection_over_union(predictions[...,26:30],target[...,21:25])
        ious = torch.cat([iou_b1.unsqueeze(0),iou_b2.unsqueeze(0)],dim=0)
        iou_maxes, bestbox = torch.max(ious,dim=0)
        exists_box = target[...,20].unsqueeze(3)
        
        
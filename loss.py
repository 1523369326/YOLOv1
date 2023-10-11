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
        # 得到的四维predictions[-1,7,7,30]，7*7表示二维图中49个网格，
        # 其中每个网格包含30个消息 21：25表示第一个框的x,y,w,h 和置信度 26：30表示第二个框的x,y,w,h 和置信度，前面20代表种类数
        #假如只能识别三种类 1，0，0代表识别第一种类， 0，1，0代表识别第二种类。。。
        # predictions[...,21:25]:x,y,w,h 和置信度，predictions[...,26:30]:x,y,w,h 和置信度

        # ... 表示选择所有剩余的维度，21：25表示第一个框信息
        iou_b1 = intersection_over_union(predictions[...,21:25],target[...,21:25])  #三维，形状为 [N, S, S]，其中 N 是批次大小，S 是网格的大小
        iou_b2 = intersection_over_union(predictions[...,26:30],target[...,21:25])
        # dim=0：指定连接维度，unsequeeze：在0维度增加一个维度 如形状为[N,S,S],unsequeeze之后变成 [1,N,S,S]
        ious = torch.cat([iou_b1.unsqueeze(0),iou_b2.unsqueeze(0)],dim=0) # [2,N,S,S]
        iou_maxes, bestbox = torch.max(ious,dim=0)
        exists_box = target[...,20].unsqueeze(3) #Iobj_i 原[N, S, S]变成[N, S, S, 1]，可以和predictions进行比较

        # ======================== #
        #   FOR BOX COORDINATES    #
        #   box坐标                #
        # ======================== #
        box_predictions = exists_box * (
            (
                bestbox * predictions[..., 26:30]
                + (1 - bestbox) * predictions[..., 21:25]
            )
        )

        box_targets = exists_box * target[..., 21:25]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2,4] + le-6)
        )
        # [N,S,S,25]
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # (N,S,S,4) -> (N*S*S,4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        #   目标损失            #
        # ==================== #
        pred_box = (
            bestbox * predictions[...,25:26] + (1 - bestbox) * predictions[...,20:21]
        )
        # [N,S,S,1] -> [N*S*S,1]
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[...,20:21]),
        )

         # ==================== #
         # FOR NO OBJECT LOSS   #
         # 无目标损失            #
         # ==================== #
        no_object_loss = self.mse(
           torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim = 1),
           torch.flatten((1 - exists_box) * target[...,20:21], start_dim=1)
         )
        
        no_object_loss += self.mse(
           torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim = 1),
           torch.flatten((1 - exists_box) * target[...,20:21], start_dim=1)
         )
        
        # ==================== #
        
        

        
        
        
import torch
import torch.nn as nn


architecture_config = [
    #Tuple: (kernel_size, num_filters, stride, padding) padding:填充，要手动计算
    (7, 64, 2, 3),
    "M", #MaxPool
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    # List: tuples and then last integer represents number of repeats
    [(1, 256, 1, 0), (3, 512, 1, 1), 4], #4:重复次数
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),

]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__() #调用父类的构造函数
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))
    
class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)  #创建卷积层
        self.fcs = self._create_fcs(**kwargs)   #创建全连接层

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x,start_dim=1))
    
    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                    in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                ]
                in_channels = x[1]
            
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            
            elif type(x) == list:
                conv1 = x[0]    #Tuple
                conv2 = x[1]    #Tuple
                num_repeats = x[2]  #Integer  重复次数

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                        in_channels, 
                        conv1[1], 
                        kernel_size=conv1[0], 
                        stride=conv1[2], 
                        padding=conv1[3],
                        )
                    ]

                    layers += [
                        CNNBlock(
                        conv1[1],   #conv1[1]是上一层的输出通道，这一层的输入通道 
                        conv2[1], 
                        kernel_size=conv2[0], 
                        stride=conv2[2], 
                        padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]
        return nn.Sequential(*layers)
    # 全连接层
    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),   #把模型展平
            nn.Linear(1024 * S * S, 496),  #Original paper this should be 4096  #把模型展平后，输入到全连接层 从1024*S*S到496
            nn.Dropout(0.0),    #Original paper uses 0.5 dropout  #防止过拟合
            nn.LeakyReLU(0.1), 
            nn.Linear(496, S * S * (C + B * 5)) #把模型展平后，输入到全连接层  （S，S，30） where C+B*5 = 30    30: 20个类别+5个bbox参数+1个置信度
        )
    
def test(S=7, B=2, C=20):
    model = Yolov1(split_size=S, num_boxes=B, num_classes=C)
    x = torch.randn((2, 3, 448, 448))   #创建一个形状为(2, 3, 448, 448)的张量x 2个样本的批次，每个样本有3个通道（例如RGB图像），每个通道的大小为448x448。
    print(model(x).shape)   #.shape属性返回模型输出的形状

test()
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange
from einops.layers.torch import Rearrange
from thop import profile
import time
from linformer import Linformer
from torchvision.ops import DeformConv2d


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        
        # 深度卷积 (depthwise convolution)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, groups=in_channels)
        
        # 点卷积 (pointwise convolution)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        # 首先进行深度卷积
        x = self.depthwise(x)
        # 然后进行点卷积
        x = self.pointwise(x)
        return x       
        

class ResDepWiseConv2Block(nn.Module):
    def __init__(self, input_channel=32, output_channel=32, stride=1, padding=1):
        super(ResDepWiseConv2Block, self).__init__()
        self.in_channel = input_channel
        self.out_channel = output_channel
        
        # 用深度可分离卷积替换原来的标准卷积
        self.conv_block = nn.Sequential(
            DepthwiseSeparableConv(input_channel, output_channel, kernel_size=3, stride=1, padding=padding),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            DepthwiseSeparableConv(output_channel, output_channel, kernel_size=3, stride=1, padding=padding),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
        # 跳跃连接也使用深度可分离卷积
        self.conv_skip = nn.Sequential(
            DepthwiseSeparableConv(input_channel, output_channel, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(output_channel),
        )
        
        # 初始化权重
        self.initialize_weights()
        
    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class MultiChannelAttention(nn.Module):
    def __init__(self, in_features, cha_num):
        super(MultiChannelAttention, self).__init__()
        self.in_features = in_features
        self.eeg_channels = cha_num
        
        self.avg_pool = nn.AdaptiveAvgPool2d((cha_num, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((cha_num, 1))

        self.fc = nn.Sequential(
            nn.Conv2d(in_features, in_features // 2, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_features // 2, in_features, kernel_size=1, bias=False) #########     cha_num     in_features
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, f, n, l = x.shape
        
        avg_pooled = self.avg_pool(x)
        max_pooled = self.max_pool(x)

        avg_out = self.fc(avg_pooled)
        max_out = self.fc(max_pooled)
        
        scale = self.sigmoid(avg_out + max_out)
        scale = scale.mean(dim=1, keepdim=True)
    
        attention_applied = x * scale
        
        return attention_applied


class ResDeformConv2Block(nn.Module):
    def __init__(self, input_channel=32, output_channel=32, stride=1, padding=1, kernel_size=3):
        super(ResDeformConv2Block, self).__init__()
        self.in_channel = input_channel
        self.out_channel = output_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 第一个可变形卷积的偏移量生成层
        self.offset_conv1 = nn.Conv2d(input_channel, 2 * kernel_size * kernel_size, kernel_size=3, stride=stride, padding=1, bias=True)
        # 第一个可变形卷积
        self.deform_conv1 = DeformConv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)


        self.offset_conv2 = nn.Conv2d(output_channel, 2 * kernel_size * kernel_size, kernel_size=3, stride=1, padding=1, bias=True)
        self.deform_conv2 = DeformConv2d(output_channel, output_channel, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # 跳跃连接（Skip Connection）
        if input_channel != output_channel or stride != 1:
            self.conv_skip = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(output_channel),
            )
        else:
            self.conv_skip = nn.Identity()

        self.initialize_weights()

    def forward(self, x):
        offset1 = self.offset_conv1(x)
        out = self.deform_conv1(x, offset1)
        out = self.bn1(out)
        out = self.relu1(out)

        offset2 = self.offset_conv2(out)
        out = self.deform_conv2(out, offset2)
        out = self.bn2(out)
        out = self.relu2(out)

        # 跳跃连接
        skip = self.conv_skip(x)

        return out + skip

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, DeformConv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class MCFE_Block(nn.Module):
    def __init__(self, input_channel=1, output_channel=32, cha_num=19):
        super(MCFE_Block, self).__init__()
        self.res1 = ResDeformConv2Block(input_channel, 8)
        self.res2 = ResDeformConv2Block(8, 16)
        self.res3 = ResDeformConv2Block(16, output_channel)
        
        self.multiatt1 = MultiChannelAttention(8, cha_num)
        self.multiatt2 = MultiChannelAttention(16, cha_num)
        self.multiatt3 = MultiChannelAttention(32, cha_num)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.initialize_weights()
        
    def forward(self, x):
        x = self.multiatt1(self.res1(x))
        x = self.lrelu(x)
        x = self.multiatt2(self.res2(x))
        x = self.lrelu(x)
        x = self.multiatt3(self.res3(x))
        x = self.lrelu(x)
        return x
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()



class MCFE_Block_DepWise(nn.Module):
    def __init__(self, input_channel=1, output_channel=32, cha_num=19):
        super(MCFE_Block_DepWise, self).__init__()
        self.res1 = ResDepWiseConv2Block(input_channel, 8)
        self.res2 = ResDepWiseConv2Block(8, 16)
        self.res3 = ResDepWiseConv2Block(16, output_channel)
        
        self.multiatt1 = MultiChannelAttention(8, cha_num)
        self.multiatt2 = MultiChannelAttention(16, cha_num)
        self.multiatt3 = MultiChannelAttention(32, cha_num)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.initialize_weights()
        
    def forward(self, x):
        x = self.multiatt1(self.res1(x))
        x = self.lrelu(x)
        x = self.multiatt2(self.res2(x))
        x = self.lrelu(x)
        x = self.multiatt3(self.res3(x))
        x = self.lrelu(x)
        return x
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class SCFE_Block(nn.Module):
    def __init__(self, eeg_length=1280):
        super(SCFE_Block, self).__init__()
        self.feature_extractors = TransformerEXt(eeg_length)
        
    def forward(self, x):
        feature_list = []
        for i in range(x.size(2)):  # 遍历第二维，即22个特征
            feature = x[:, :, i, :]  # 获取当前特征（batch size, 1, 1000）
            HD_feature = self.feature_extractors(feature)
            
            feature_list.append(HD_feature)
        SinChan_fea = torch.stack(feature_list, dim=2)
        return SinChan_fea
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class SCFE_Block_Linformer(nn.Module):
    def __init__(self, eeg_length=1280):
        super(SCFE_Block_Linformer, self).__init__()
        self.feature_extractors = LinformerEXt(eeg_length)
        
    def forward(self, x):
        feature_list = []
        for i in range(x.size(2)):  # 遍历第二维，即22个特征
            feature = x[:, :, i, :]  # 获取当前特征（batch size, 1, 1000）
            HD_feature = self.feature_extractors(feature)
            
            feature_list.append(HD_feature)
        SinChan_fea = torch.stack(feature_list, dim=2)
        return SinChan_fea
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class Swish(nn.Module):  # 自定义的激活函数
    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            Swish(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b t (h d) -> b h t d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b t (h d) -> b h d t", h=self.num_heads)
        values = rearrange(self.values(x), "b t (h d) -> b h t d", h=self.num_heads)
        energy = torch.matmul(queries, keys) # 张量乘法

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = torch.matmul(att, values)
        out = rearrange(out, "b h t d -> b t (h d)")
        out = self.projection(out)
        return out

class LinformerAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout, seq_len):
        super().__init__()
        self.attention = Linformer(
            dim=emb_size,
            seq_len=seq_len,
            depth=1,  # 单层Linformer，适合替换传统Transformer
            heads=num_heads,
            k=256  # 窗口大小
        )
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # 计算Linformer自注意力
        output = self.attention(x)
        x = self.projection(output)
        return x

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=8, drop_p=0.1, forward_expansion=1, forward_drop_p=0.1):
        super().__init__(  
            nn.Sequential(
                Rearrange('n (h) (w) -> n (w) (h)'),
            ),
                     
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p),
            )),
            nn.Sequential(
                Rearrange('n (w) (h) -> n (h) (w)'),
            )    
        )
      
class LinformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=4, drop_p=0.1, forward_expansion=1, forward_drop_p=0.1, seq_len=250):
        super().__init__(
            nn.Sequential(
                Rearrange('n (h) (w) -> n (w) (h)'),
            ),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                LinformerAttention(emb_size, num_heads, drop_p, seq_len),  # 替换为LinformerAttention
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p),
            )),
            nn.Sequential(
                Rearrange('n (w) (h) -> n (h) (w)'),
            )
        )  


class TransformerEXt(nn.Module):
    def __init__(self, data_num = 1280):
        super(TransformerEXt, self).__init__()
        self.data_num = data_num
        
        self.con1 = nn.Sequential(
            nn.Conv1d(1, 32, 3, 1, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
            )
        self.pool1 = nn.AvgPool1d(2,stride=2)
        self.att1 = TransformerEncoderBlock(emb_size=32)
        
        self.con2 = nn.Sequential(
            nn.Conv1d(32, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
            )
        self.pool2 = nn.AvgPool1d(2,stride=2)
        self.att2 = TransformerEncoderBlock(emb_size=64)
        
        self.con3 = nn.Sequential(
            nn.Conv1d(64, 32, 3, 1, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
            )
        self.linear = nn.Linear(32 * 250, data_num)
        self.con_up = nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=4, stride=4, padding=0, output_padding=0)


    def forward(self, x):
        x = self.pool1(self.con1(x))
        x = self.att1(x)
        x = self.pool2(self.con2(x))
        x = self.att2(x)
        x = self.con3(x)
        x = self.con_up(x)
        return x


class LinformerEXt(nn.Module):
    def __init__(self, data_num = 1280):
        super(LinformerEXt, self).__init__()
        self.data_num = data_num
        
        self.con1 = nn.Sequential(
            nn.Conv1d(1, 32, 3, 1, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
            )
        self.pool1 = nn.AvgPool1d(2,stride=2)
        self.att1 = LinformerEncoderBlock(emb_size=32, seq_len=data_num // 2)
        
        self.con2 = nn.Sequential(
            nn.Conv1d(32, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
            )
        self.pool2 = nn.AvgPool1d(2,stride=2)
        self.att2 = LinformerEncoderBlock(emb_size=64, seq_len=data_num // 4)
        
        self.con3 = nn.Sequential(
            nn.Conv1d(64, 32, 3, 1, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
            )
        self.linear = nn.Linear(32 * 250, data_num)
        self.con_up = nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=4, stride=4, padding=0, output_padding=0)


    def forward(self, x):
        x = self.pool1(self.con1(x))
        x = self.att1(x)
        x = self.pool2(self.con2(x))
        x = self.att2(x)
        x = self.con3(x)
        x = self.con_up(x)
        return x



# 新设计的重建模块
class Recon_Block(nn.Module):
    def __init__(self):
        super(Recon_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 5), padding=(0, 2), bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 5), padding=(0, 2), bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 5), padding=(0, 2), bias=False)
        self.bn3 = nn.BatchNorm2d(8)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # self.fc = nn.Linear(8 * channel_num * data_num, channel_num * data_num)
        self.final_conv = nn.Conv2d(8, 1, kernel_size=(1, 1), bias=False)
        
    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.lrelu(x)
        x = self.bn2(self.conv2(x))
        x = self.lrelu(x)
        x = self.bn3(self.conv3(x))

        x = self.final_conv(x)
        return x



# 默认处理数据为19通道、5s长、信号采样率为256Hz
class ASTI_Net_DepWise(nn.Module):
    def __init__(self, eeg_length=1280, cha_num=19, input_channel=1, fea_channel=32, kernel_size=3, padding=1):
        super(ASTI_Net_DepWise, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        
        self.mcfe = MCFE_Block_DepWise(input_channel, fea_channel, cha_num)
        self.scfe = SCFE_Block(eeg_length)
        self.re = Recon_Block()

    def forward(self, x):

        x1 = self.mcfe(x)
        x2 = self.scfe(x)
        x3 = torch.cat((x1, x2), dim=1)
        eeg_re = self.re(x3)
        
        return eeg_re

class ASTI_Net_Linformer(nn.Module):
    def __init__(self, eeg_length=1280, cha_num=19, input_channel=1, fea_channel=32, kernel_size=3, padding=1):
        super(ASTI_Net_Linformer, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        
        self.mcfe = MCFE_Block(input_channel, fea_channel, cha_num)
        self.scfe = SCFE_Block_Linformer(eeg_length)
        self.re = Recon_Block()

    def forward(self, x):

        x1 = self.mcfe(x)
        x2 = self.scfe(x)
        x3 = torch.cat((x1, x2), dim=1)
        eeg_re = self.re(x3)
        
        return eeg_re

class ASTI_Net_DepWise_Linformer(nn.Module):
    def __init__(self, eeg_length=1280, cha_num=19, input_channel=1, fea_channel=32, kernel_size=3, padding=1):
        super(ASTI_Net_DepWise_Linformer, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        
        self.mcfe = MCFE_Block_DepWise(input_channel, fea_channel, cha_num)
        self.scfe = SCFE_Block_Linformer(eeg_length)
        self.re = Recon_Block()

    def forward(self, x):

        x1 = self.mcfe(x)
        x2 = self.scfe(x)
        x3 = torch.cat((x1, x2), dim=1)
        eeg_re = self.re(x3)
        
        return eeg_re
    
    
class ASTI_Net(nn.Module):
    def __init__(self, eeg_length=1280, cha_num=19, input_channel=1, fea_channel=32, kernel_size=3, padding=1):
        super(ASTI_Net, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        
        self.mcfe = MCFE_Block(input_channel, fea_channel, cha_num)
        self.scfe = SCFE_Block(eeg_length)
        self.re = Recon_Block()

    def forward(self, x):

        x1 = self.mcfe(x)
        x2 = self.scfe(x)
        x3 = torch.cat((x1, x2), dim=1)
        eeg_re = self.re(x3)
        
        return eeg_re



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    batch_size = 32
    cha_num = 19
    input = torch.ones(32, 1, cha_num, 1280)
    # model = ASTI_Net(cha_num=cha_num)
    # model = ASTI_Net_DepWise(cha_num=cha_num)
    model = ASTI_Net_Linformer(cha_num=cha_num)
    # model = ASTI_Net_DepWise_Linformer(cha_num=cha_num)


    start_time = time.time()  # 记录开始时间
    out = model(input)
    end_time = time.time()  # 记录结束时间
    print("computation time ：", (end_time - start_time)/batch_size, "s")


    flops, params = profile(model, inputs=(input,))   # flops 将会是模型的浮点操作次数，params 将会是模型的参数数量
    print('input shape:', input.shape)
    print('parameters:', params)
    print('flops', flops)
    print('output shape', out.shape) 
    
    

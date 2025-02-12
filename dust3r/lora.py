import torch 
from torch import nn
import math 

LORA_ALPHA=1    # lora的a权重
LORA_R=8    # lora的秩
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # 训练设备

# Lora实现，封装linear，替换到父module里
class LoraLayer(nn.Module):
    def __init__(self,raw_linear,in_features,out_features,r,alpha):
        super().__init__()
        self.r = r 
        self.alpha = alpha
        self.lora_a = nn.Parameter(torch.empty((in_features, r)))
        self.lora_b = nn.Parameter(torch.zeros((r, out_features)))
    
        nn.init.kaiming_uniform_(self.lora_a,a=math.sqrt(5))

        self.raw_linear = raw_linear
    
    def forward(self,x):    # x:(batch_size,in_features)
        raw_output = self.raw_linear(x)   
        lora_output = x @ ((self.lora_a @ self.lora_b) * self.alpha / self.r)
        return raw_output + lora_output

def inject_lora(model, name, layer):
    name_cols = name.split('.')

    # 逐层下探到linear归属的module
    children = name_cols[:-1] # The last layer is the qkv layer
    cur_layer = model 
    for child in children:
        cur_layer = getattr(cur_layer,child)
    
    #print(layer==getattr(cur_layer,name_cols[-1]))
    lora_layer = LoraLayer(layer, layer.in_features, layer.out_features, LORA_R, LORA_ALPHA).to(DEVICE)
    setattr(cur_layer, name_cols[-1], lora_layer)
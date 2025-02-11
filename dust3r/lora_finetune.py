from unet import UNet
from dataset import train_dataset
from diffusion import forward_diffusion
from config import * 
import torch 
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os 

LORA_ALPHA = 1    # lora的a权重
LORA_R = 8        # lora的秩
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # 训练设备

# Lora实现，封装linear，替换到父module里
class LoraLayer(nn.Module):
    def __init__(self, raw_linear, in_features, out_features, r, alpha):
        super().__init__()
        self.r = r 
        self.alpha = alpha
        self.lora_a = nn.Parameter(torch.empty((in_features,r)))
        self.lora_b = nn.Parameter(torch.zeros((r,out_features)))
    
        nn.init.kaiming_uniform_(self.lora_a,a=math.sqrt(5))

        self.raw_linear = raw_linear
    
    def forward(self,x):    # x: (batch_size, in_features)
        raw_output = self.raw_linear(x)   
        lora_output = x @ ((self.lora_a @ self.lora_b) * self.alpha / self.r) # matmul(x,matmul(lora_a,lora_b)*alpha/r)
        return raw_output + lora_output

def inject_lora(model, name, layer):
    name_cols = name.split('.')

    # Iteratively get the last layer is the qkv layer
    children = name_cols[:-1]
    cur_layer = model 
    for child in children:
        cur_layer = getattr(cur_layer, child)
    
    #print(layer==getattr(cur_layer,name_cols[-1]))
    lora_layer = LoraLayer(layer, layer.in_features, layer.out_features, LORA_R, LORA_ALPHA).to(DEVICE)
    setattr(cur_layer, name_cols[-1], lora_layer)

if __name__ == '__main__':
    model = torch.load(args.model)

    ######################################### LoRA
    for name, layer in model.named_modules():
        print("name: ", name)
        name_cols = name.split('.')
        # Retrieve all linear layer in cross attention
        # For each linear layer, change y=WX to y=WX + WaWbX
        filter_names = ['qkv']
        if any(n in name_cols for n in filter_names) and isinstance(layer, nn.Linear):
            print(name)
            inject_lora(model, name, layer)

    # Load LoRA weight
    try:
        restore_lora_state = torch.load('lora.pt')
        model.load_state_dict(restore_lora_state, strict=False)
    except:
        pass

    model.to(args.device)

    for name, param in model.named_parameters():
        if name.split('.')[-1] not in ['lora_a','lora_b']:  # 非Lora部分不计算梯度
            param.requires_grad = False
        else:
            param.requires_grad = True
    #########################################

    ######################################### LoRA
    lora_state={}
    for name, param in model.named_parameters():
        name_cols = name.split('.')
        filter_names = ['lora_a','lora_b']
        if any(n == name_cols[-1] for n in filter_names):
            lora_state[name] = param
    torch.save(lora_state, 'lora.pt.tmp')
    os.replace('lora.pt.tmp', 'lora.pt')
    #########################################
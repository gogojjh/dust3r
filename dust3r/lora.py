import torch 
from torch import nn
import math 

LORA_ALPHA = 4    # LoRA's alpha scaling weight
LORA_R = 16       # LoRA rank (dimensionality of the low-rank matrices)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Training device (CUDA GPU if available, else CPU)

# LoRA layer implementation. Wraps a linear layer and replaces it in the parent module
class LoraLayer(nn.Module):
    def __init__(self, raw_linear, in_features, out_features, r, alpha):
        super().__init__()
        self.r = r 
        self.alpha = alpha
        self.lora_a = nn.Parameter(torch.empty((in_features, r)))
        self.lora_b = nn.Parameter(torch.zeros((r, out_features)))
    
        # Initialize weights using Kaiming uniform distribution
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

        self.raw_linear = raw_linear  # Original linear layer to enhance with LoRA
    
    def forward(self, x):    # x: (batch_size, in_features)
        # Combine original output with LoRA adaptation
        raw_output = self.raw_linear(x)   
        lora_output = x @ ((self.lora_a @ self.lora_b) * self.alpha / self.r)
        return raw_output + lora_output

def inject_lora(model, name, layer):
    name_cols = name.split('.')
    
    # Traverse the model hierarchy to reach the parent module of the target linear layer
    children = name_cols[:-1]  # Split the layer name into components (excluding the last part, which is the target layer name)
    cur_layer = model 
    for child in children:
        # Iterate through each component to reach the parent module
        cur_layer = getattr(cur_layer, child)
    
    # Create and inject the LoRA-enhanced layer
    lora_layer = LoraLayer(layer, layer.in_features, layer.out_features, LORA_R, LORA_ALPHA).to(DEVICE)
    # Replace the original linear layer with the LoRA-enhanced layer in the parent module
    setattr(cur_layer, name_cols[-1], lora_layer)

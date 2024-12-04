import torch 
import numpy 
import einops 
import torch.nn as nn 

class InterConv(nn.Module): 
    def __init__(self, in_channels, 
                kernel_size, 
                aggregate_type): 

        super(InterConv, self).__init__()

        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels *  2, in_channels, 
            kernel_size=kernel_size, padding=padding)
        
        self.aggregate_type = aggregate_type 
    
    def forward(self, x, set_size): 
        assert x.ndim == 4 
        x = einops.rearrange(x, '(b n) c h w -> b n c h w', n=set_size)
        if self.aggregate_type == "mean": 
            x_agg = x.mean(dim=1)
            x_agg = einops.repeat(x_agg, 'b c h w -> b n c h w', n=set_size)
        x = torch.cat([x, x_agg], dim=2)
        x = einops.rearrange(x, 'b n c h w -> (b n) c h w')
        x = self.conv(x)
        return x

    
layer = InterConv(3, 3, "mean")
x = torch.rand(24, 3, 32, 32)
x = layer(x, set_size=3)
print(x.shape)
        
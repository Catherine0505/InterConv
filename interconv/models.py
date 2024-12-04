import torch
import torch.nn as nn

import einops

import layers

class SimpleUNet(nn.Module): 
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 features=[64, 64, 64, 64],
                 kernel_size=3,
                 do_batchnorm=False):
        super(SimpleUNet, self).__init__() 
    
        self.kernel_size = kernel_size 
        self.padding =  (kernel_size - 1) // 2

        in_channels_down = [features[0]] + features[:-1]
        out_channels_down = features 

        out_channels_up = features[:-1][::-1] + [out_channels]
        in_channels_up = [features[-1] * 2] + [out_channels_up[i] + out_channels_down[-i-2] for i in range(len(out_channels_up) - 1)]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        # Initial Conv
        conv = nn.Conv2d(in_channels, features[0], kernel_size=self.kernel_size, padding=self.padding)
        lst = [conv]
        if do_batchnorm:
            lst.append(nn.BatchNorm2d(features[0]))
        lst.append(nn.PReLU())
        self.init_conv = nn.Sequential(*lst)
        print(type(self.init_conv))
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        for (in_channels, out_channels) in zip(in_channels_down, out_channels_down): 
            self.downs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding), 
                nn.PReLU()))
        
        for (in_channels, out_channels) in zip(in_channels_up[:-1], out_channels_up[:-1]): 
            self.ups.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding), 
                nn.PReLU()))
        
        self.bottleneck = nn.Sequential(nn.Conv2d(features[-1], features[-1], kernel_size=self.kernel_size, padding=self.padding), nn.PReLU())

        self.ups.append(nn.Conv2d(in_channels_up[-1], out_channels_up[-1], kernel_size=self.kernel_size, padding=self.padding))
    
    def forward(self, x): 
        skip_connections = []
        x = self.init_conv(x)

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for (i, up) in enumerate(self.ups): 
            x = self.upsample(x)
            x = torch.cat((skip_connections[i], x), dim=1)
            x = up(x)
        
        return x


class SimpleUNetInterConv(nn.Module): 
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 features=[64, 64, 64, 64],
                 kernel_size=3,
                 do_batchnorm=False, 
                 aggregate_type="mean"):
        
        super(SimpleUNetInterConv, self).__init__() 
    
        self.kernel_size = kernel_size 
        self.padding =  (kernel_size - 1) // 2

        in_channels_down = [features[0]] + features[:-1]
        out_channels_down = features 

        out_channels_up = features[:-1][::-1] + [out_channels]
        in_channels_up = [features[-1] * 2] + [out_channels_up[i] + out_channels_down[-i-2] for i in range(len(out_channels_up) - 1)]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        # Initial Conv
        conv = nn.Conv2d(in_channels, features[0], kernel_size=self.kernel_size, padding=self.padding)
        lst = [conv]
        if do_batchnorm:
            lst.append(nn.BatchNorm2d(features[0]))
        lst.append(nn.PReLU())
        self.init_conv = nn.Sequential(*lst)
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs_interconv = nn.ModuleList()
        self.ups_interconv = nn.ModuleList()

        for (in_channels, out_channels) in zip(in_channels_down, out_channels_down): 
            self.downs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding), 
                nn.PReLU()))
            self.downs_interconv.append(
                layers.InterConv(out_channels, kernel_size=self.kernel_size, aggregate_type=aggregate_type)
            )
        
        for (in_channels, out_channels) in zip(in_channels_up[:-1], out_channels_up[:-1]): 
            self.ups.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding), 
                nn.PReLU()))
            self.ups_interconv.append(
                layers.InterConv(out_channels, kernel_size=self.kernel_size, aggregate_type=aggregate_type)
            )
        
        self.bottleneck = nn.Sequential(nn.Conv2d(features[-1], features[-1], kernel_size=self.kernel_size, padding=self.padding), nn.PReLU())
        self.bottleneck_interconv = layers.InterConv(features[-1], kernel_size=self.kernel_size, aggregate_type=aggregate_type)

        self.ups.append(nn.Conv2d(in_channels_up[-1], out_channels_up[-1], kernel_size=self.kernel_size, padding=self.padding))
        self.ups_interconv.append(layers.InterConv(out_channels_up[-1], kernel_size=self.kernel_size, aggregate_type=aggregate_type))
    
    def forward(self, x, set_size): 
        skip_connections = []
        x = self.init_conv(x)

        for (i, down) in enumerate(self.downs):
            x = down(x)
            x = self.downs_interconv[i](x, set_size)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        x = self.bottleneck_interconv(x, set_size)

        skip_connections = skip_connections[::-1]

        for (i, up) in enumerate(self.ups): 
            x = self.upsample(x)
            x = torch.cat((skip_connections[i], x), dim=1)
            x = up(x)
            x = self.ups_interconv[i](x, set_size)
        
        return x


model = SimpleUNet(in_channels=1,
                 out_channels=1,
                 features=[64, 64, 64, 64],
                 kernel_size=3,
                 do_batchnorm=False)
model = SimpleUNetInterConv(in_channels=1,
                 out_channels=1,
                 features=[64, 64, 64, 64],
                 kernel_size=3,
                 do_batchnorm=False, 
                 aggregate_type="mean")
x = torch.rand(24, 1, 32, 32)
x = model(x, set_size=3)
print(x.shape)
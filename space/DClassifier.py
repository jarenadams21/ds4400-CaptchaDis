import torch.nn as nn

class CNN2DAudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = []
        
        def build_conv_layer(in_d, out_d, ks, stride, pad):
            conv1 = nn.Conv2d(in_d, out_d, 
                              kernel_size=(ks, ks), 
                              stride=(stride, stride), 
                              padding=(pad, pad))
            relu1 = nn.ReLU()
            return [conv1, relu1]
        
        self.conv_layers += build_conv_layer(1, 8, 5, 2, 2)
        self.conv_layers += build_conv_layer(8, 16, 3, 2, 1)
        self.conv_layers += build_conv_layer(16, 32, 3, 2, 1)
        self.conv_layers += build_conv_layer(32, 64, 3, 2, 1)
        
        self.conv = nn.Sequential(*self.conv_layers)
        self.lin = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        return self.lin(x)

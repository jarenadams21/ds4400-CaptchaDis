# Classifier Using 2D CNN
import torch.nn as nn
import torch

class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SimpleGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

"""
class RNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn_layers = []
        
        def build_rnn_layer(in_d, out_d, ks, stride, pad):
            rnn1 = nn.RNN(in_d, out_d, 
                              kernel_size=(ks, ks), 
                              stride=(stride, stride), 
                              padding=(pad, pad))
            relu1 = nn.ReLU()
            bn1 = nn.BatchNorm2d(out_d)
            nn.init.kaiming_normal_(rnn1.weight, a=0.1)
            rnn1.bias.data.zero_()
            return [rnn1, relu1, bn1]
        
        self.rnn_layers += build_rnn_layer(1, 8, 5, 2, 2)
        self.rnn_layers += build_rnn_layer(8, 16, 3, 2, 1)
        self.rnn_layers += build_rnn_layer(16, 32, 3, 2, 1)
        self.rnn_layers += build_rnn_layer(32, 64, 3, 2, 1)
        
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=10)
        
        self.conv = nn.Sequential(*self.rnn_layers)
        
    def forward(self, x):
        x = self.conv(x)
        
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        
        return self.lin(x)
"""
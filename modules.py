import torch
import torch.nn as nn
from torch.nn import LeakyReLU, ReLU, Tanh, Sigmoid
from torch.nn.init import kaiming_normal_, xavier_normal_
from torch.utils import data



class Layer(nn.Module):
    def __init__(self, nonlinearity='relu', activation_use=True, norm=None):
        super().__init__()
        self.nonlinearity = nonlinearity
        self.activation = self.find_activation(nonlinearity)
        self.activation_use = activation_use
        self.wi = norm
        
        
    def weight_normalization(self):
        if self.wi == 'xaviar':
            xavier_normal_(self.layer.weight, gain=nn.init.calculate_gain(self.nonlinearity))
        if self.wi == 'he':
            kaiming_normal_(self.layer.weight, nonlinearity=self.nonlinearity)
        
    def find_activation(self,nonlinearity):
        if nonlinearity == 'relu':
            return ReLU()
        if nonlinearity == 'leaky_relu':
            return LeakyReLU()
        if nonlinearity == 'tanh':
            return Tanh()
        if nonlinearity == 'sigmoid':
            return Sigmoid()
        raise 'inaccessible activation function!'


class LinearNorm(Layer):
    def __init__(self,d_in,d_out,bias=True, nonlinearity='relu', activation_use=True, norm=True):
        super().__init__(nonlinearity=nonlinearity,activation_use=activation_use, norm=norm)
        self.layer = nn.Linear(d_in,d_out,bias=bias)
        self.weight_normalization()
      
    def forward(self,x):
        if self.activation_use:
            return self.activation(self.layer(x))
        return self.layer(x)



class ConvNorm(Layer):
    def __init__(self, type_, in_channels, out_channels, kernel_size,batch_norm=False,dropout=0, stride=1, padding=0, dilation=1,
                                groups=1, bias=True, padding_mode='zeros', device=None, dtype=None,nonlinearity='relu',activation_use=True,norm=True):
        
		super().__init__(nonlinearity=nonlinearity,activation_use=activation_use,norm=norm)
        self.batch_norm_use = batch_norm
        self.dropout = nn.Dropout(p=dropout)
        
        if type_ == '1d':
            self.layer = torch.nn.Conv1d(in_channels, out_channels, kernel_size, batch_norm=batch_norm,dropout=dropout, stride=stride, padding=padding, dilation=dilation,
                                         groups=groups, bias=bias, padding_mode=padding_mode)
            self.batch_norm = nn.BatchNorm1d(out_channels)
            
        elif type_ == '2d':
            self.layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                                         groups=groups, bias=bias, padding_mode=padding_mode)
            self.batch_norm = nn.BatchNorm2d(out_channels)
        else:
            raise 'Not implemented yet!'
           
        self.weight_normalization()

        
    
    def forward(self,x):
        x = self.layer(x)
        if self.batch_norm_use:
            x = self.batch_norm(x)
        if self.activation_use:
            x = self.activation(x)
        x = self.dropout(x)
        return x
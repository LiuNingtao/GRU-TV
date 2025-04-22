
import torch
from torch.nn import LSTMCell, Linear
import math


class TLSTM(torch.nn.Module):  
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.cell = LSTMCell(input_size, hidden_size)
    self.c_gate = Linear(hidden_size, hidden_size)
   
    self.lin = torch.nn.Linear(hidden_size, output_size, bias=True)
    self.reset_parameters()
  
  def reset_parameters(self):
    for params in self.parameters():
      torch.nn.init.normal_(params, mean=0, std=0.1)

  def forward(self, X, Mask, Delta, dt):
    h = torch.autograd.Variable(torch.zeros(1, self.hidden_size)).cuda()
    c = torch.autograd.Variable(torch.zeros(1, self.hidden_size)).cuda()
    if len(X.shape) != 2:
      print(X.shape)
    for layer in range(X.shape[0]):
      x = X[layer, :]
      dt_value = dt[layer]
      x = torch.unsqueeze(x, dim=0)
      
      h, c = self.cell(x, (h, c))
    output = self.lin(h) 
    output = torch.squeeze(output)     
    output = torch.sigmoid(output)
    return output


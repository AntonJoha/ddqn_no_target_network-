import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class TGeLU(nn.Module):
    def __init__(self, tl, tr, device, inplace:bool = False):
        super().__init__()
        self.inplace = inplace
        self.device = device
        self.tr = torch.tensor(tr).to(self.device)
        self.tl = torch.tensor(tl).to(self.device)
         
        
    def forward(self, input):
        dist = Normal(torch.zeros(input.shape).to(self.device),torch.ones(input.shape).to(self.device))
        
        cond1 = (input>=self.tr)
        cond2 = (input >= 0)*(input<self.tr)
        cond3 = (self.tl<=input)*(input<0)
        cond4 = (input<self.tl)
        
        term1 = self.tr*dist.cdf(self.tr) + (input-self.tr)*(1-dist.cdf(input-self.tr))
        term2 = input*dist.cdf(input)
        term3 = input*(1-dist.cdf(input))
        term4 = self.tl*(1-dist.cdf(self.tl)) + (input-self.tl)*dist.cdf(input-self.tl)
        
        return cond1*term1 + cond2*term2 + cond3*term3 + cond4*term4
                
        
    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else " "
        return inplace_str

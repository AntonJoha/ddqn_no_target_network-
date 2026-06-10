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
        self.normal = Normal(
            torch.tensor(0.0, device=self.device),
            torch.tensor(1.0, device=self.device),
        )
        self.cdf_tr = self.normal.cdf(self.tr)
        self.cdf_tl = self.normal.cdf(self.tl)
         
        
    def forward(self, input):
        cond1 = (input>=self.tr)
        cond2 = (input >= 0)*(input<self.tr)
        cond3 = (self.tl<=input)*(input<0)
        cond4 = (input<self.tl)

        cdf_input = self.normal.cdf(input)
        cdf_input_tr = self.normal.cdf(input - self.tr)
        cdf_input_tl = self.normal.cdf(input - self.tl)
        term1 = self.tr * self.cdf_tr + (input - self.tr) * (1 - cdf_input_tr)
        term2 = input * cdf_input
        term3 = input * (1 - cdf_input)
        term4 = self.tl * (1 - self.cdf_tl) + (input - self.tl) * cdf_input_tl
        
        return cond1*term1 + cond2*term2 + cond3*term3 + cond4*term4
                
        
    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else " "
        return inplace_str

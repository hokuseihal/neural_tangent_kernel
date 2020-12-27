import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class Linear(nn.Linear):
    def __init__(self,infeature,outfeature):
        super(Linear, self).__init__(infeature,outfeature)
        self.infeature=infeature
        torch.nn.init.normal_(self.weight)
        torch.nn.init.normal_(self.bias)
    def forward(self,x):
        return F.linear(x,self.weight/np.sqrt(self.infeature),self.bias)


class NTK(nn.Module):
    def __init__(self,in_feature,inter_feature,out_feature=10):
        super(NTK, self).__init__()
        self.fc1=Linear(in_feature,inter_feature)
        self.fc2=Linear(inter_feature,inter_feature)
        self.fc3=Linear(inter_feature,out_feature)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

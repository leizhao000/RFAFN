from torchstat import stat
from model import LFIFN7
from model.CFEB import RFAFN_DRB,RFAFN_ERB,RFAFN_CFEBwoCAM,RFAFN_CB
from model.RL import RFAFN_1234,RFAFN_4,RFAFN_3,RFAFN_124,RFAFN_134
from model.AFFB import ESA,SK
from model.model1 import CARN,IDN
from torchscan import summary
from ptflops import get_model_complexity_info
from thop import profile
import torch
import torch.nn as nn
model=CARN.CARN()
#stat(model, (3, 320, 180))

#summary(model, (3, 320, 180))
def count_pixelshuffle(m, x, y):
    x = x[0]
    nelements = x.numel()
    total_ops = nelements
    m.total_ops = torch.Tensor([int(total_ops)])

from thop import profile
from thop import clever_format
input = torch.randn(1, 3, 160, 160)
macs, params = profile(model, inputs=(input, ))
macs, params = clever_format([macs, params], "%.3f")
print(macs)
print(params)



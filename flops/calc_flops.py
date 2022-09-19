from model.model1 import IDN
from model.model1 import RFDN,RFAFN_134
from  model.CFEB import RFAFN_DRB, RFAFN_CFEBwoCAM
from flops.profile import  profile
import torch


model=RFDN.RFDN()



flops, params = profile(model, input_size=(1,3,120,120))
print('IMDN_light: {} x {}, flops: {:.10f} GFLOPs, params: {}'.format(120,120,flops/(1e9),params))


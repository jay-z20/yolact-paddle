import torch
import paddle
from collections import OrderedDict

path = './yolact_base_54_800000.pth'

state_dict = torch.load(path,map_location=torch.device('cpu'))

#print(state_dict.keys())
#print(state_dict['backbone.layers.0.0.conv1.weight'])

print(state_dict['backbone.layers.0.0.conv1.weight'].shape)

pd_state = OrderedDict()


for k in state_dict.keys():
    if k.endswith("num_batches_tracked"):
        continue
    v = state_dict[k].cpu().numpy()
    if "running" in k: # "backbone.layers.0.0.bn1.running_mean"
        k = k.replace("running","")
        if k.endswith("_var"):
            k = k.replace("_var","_variance")
 
    pd_state[k] = paddle.to_tensor(v)

print("len:",len(pd_state)) ## 558
paddle.save(pd_state,"yolact_base_54_800000.pdparams")


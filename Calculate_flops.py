from pthflops import count_ops
import torchvision.models as models
import torch


device = "cuda:0"
model = models.resnet18(weights=None).to(device)
inp = torch.rand(1, 3, 224, 224).to(device)

all_ops, all_data = count_ops(model, inp)

flops, bops = 0, 0
for op_name, ops_count in all_data:
    if 'conv2' in op_name and 'onnx::' not in op_name:
        bops += ops_count
    else:
        flops += ops_count

print('Total number of FLOPs: {}', flops)
print('Total number of BOPs: {}', bops)

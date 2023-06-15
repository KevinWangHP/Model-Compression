import torch
import torchvision.models as models

from bnn import BConfig, prepare_binary_model
# Import a few examples of quantizers
from bnn.ops import BasicInputBinarizer, BasicScaleBinarizer, XNORWeightBinarizer

device = 'cuda:0'

# Create your desire model (note the default R18 may be suboptimal)
# additional binarization friendly models are available in bnn.models
model = models.resnet18(weights=None).to(device)

# Define the binarization configuration and assign it to the model
bconfig = BConfig(
    activation_pre_process=BasicInputBinarizer,
    activation_post_process=BasicScaleBinarizer,
    # optionally, one can pass certain custom variables
    weight_pre_process=XNORWeightBinarizer.with_args(center_weights=True)
)
# Convert the model appropiately, propagating the changes from parent node to leafs
# The custom_config_layers_name syntax will perform a match based on the layer name, setting a custom quantization function.
bmodel = prepare_binary_model(model, bconfig, custom_config_layers_name=[{'conv1': BConfig()}]).to(device)
model = models.resnet18(weights=None).to(device)
# You can also ignore certain layers using the ignore_layers_name.
# To pass regex expression, frame them between $ symbols, i.e.: $expression$.
print("end")



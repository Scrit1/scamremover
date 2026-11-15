import os
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"
import torch
from torch import nn
from torchvision import models
from torchviz import make_dot

# 1. Rebuild your exact model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.eval() # Put in eval mode for visualizer

# 2. Create a dummy input tensor that matches your image size
x = torch.randn(1, 3, 224, 224)

# 3. Pass the input through the model
y = model(x)

# 4. Generate the graph and save it as a PNG file in your current folder
make_dot(y, params=dict(model.named_parameters())).render("resnet18_graph", format="png")

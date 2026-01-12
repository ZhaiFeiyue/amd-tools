import torch
import torch._inductor.config as iconfig
iconfig.trace.enabled = True
iconfig.trace.graph_diagram = True
torch.set_float32_matmul_precision('high')

class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(100, 100)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.l(x))

m = ToyModel().to(device="cuda:0")

m = torch.compile(m)
input_tensor = torch.randn(100).to(device="cuda:0")
out = m(input_tensor)
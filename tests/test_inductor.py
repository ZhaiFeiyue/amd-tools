import os
import torch
import torch._inductor
import torch.nn as nn


class AddModule(nn.Module):
    def __init__(self):
        super(AddModule, self).__init__()

    def forward(self, x, y):
        """
        前向传播：将输入的两个张量相加
        :param x: 张量
        :param y: 张量
        :return: x + y
        """
        return x + y

model = AddModule()

model.eval()

with torch.inference_mode():
    inductor_configs = {}

    if torch.cuda.is_available():
        device = "cuda"
        inductor_configs["max_autotune"] = True
    else:
        device = "cpu"

    model = model.to(device=device)
    example_inputs = (torch.randn(2, 3, 224, 224, device=device), torch.randn(2, 3, 224, 224, device=device))

    exported_program = torch.export.export(
        model,
        example_inputs,
    )
    path = torch._inductor.aoti_compile_and_package(
        exported_program,
        package_path=os.path.join(os.getcwd(), "resnet18.pt2"),
        inductor_configs=inductor_configs
    )
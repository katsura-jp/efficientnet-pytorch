import torch
import torch.nn as nn
from torch._jit_internal import weak_module, weak_script_method


@weak_module
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    @weak_script_method
    def forward(self, input):
        return input * torch.sigmoid(input)


def test():
    x = torch.FloatTensor(16, 128, 16, 16)
    swish = Swish()
    print(swish(x).size())

if __name__ == '__main__':
    test()
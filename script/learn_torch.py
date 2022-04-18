import torch
from torch import nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(20, 30),
            nn.ReLU(),
            nn.Linear(30, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.Linear(10, 2)
        )

    def forward(self, x):
        y = self.block(x)
        return y


class Temp:

    def __call__(self, *args, **kwargs):
        x = args[0]
        print(x)

    def add(self, x, y, z):
        print(x + y + z)

    def add2(self, *args, **kwargs):

        print('args is: {}'.format(args))
        print('kwargs is: {}'.format(kwargs))

        sum = 0
        for item in args:
            sum += item
        return sum


if __name__ == '__main__':
    model = MyModel()
    x_1 = torch.randn(3, 224, 224) # [bs]
    x_2 = torch.randn(3, 224, 224)

    x_1 = torch.reshape(x_1, [1, 3, 224, 224])
    x_2 = torch.reshape(x_2, [1, 3, 224, 224])

    input = torch.cat([x_1, x_2], dim=0)

    print(input.shape)
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
    a = [[0.32521570841061204, 0.255700325732899, 0.5667574931880109, 0.47164948453608246, 0.6875, 0.4199655765920826],
         [0.3289250866865575, 0.26302931596091206, 0.5722070844686649, 0.422680412371134, 0.6875, 0.387263339070568],
         [0.32731231352310297, 0.2768729641693811, 0.6049046321525886, 0.44329896907216493, 0.6875, 0.3993115318416523],
         [0.3158616240625756, 0.25895765472312704, 0.5858310626702997, 0.46134020618556704, 0.6875, 0.3769363166953528],
         [0.3230384646399484, 0.254885993485342, 0.5504087193460491, 0.4484536082474227, 0.6875, 0.4165232358003442],
         [0.32118377550197563, 0.253257328990228, 0.553133514986376, 0.42783505154639173, 0.6875, 0.423407917383821]]

    print(a[1][4])

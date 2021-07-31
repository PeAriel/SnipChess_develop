from torch.nn import Conv2d, ReLU, MaxPool2d, Linear, Module
from torch import flatten

class ChessConvNet(Module):
    def __init__(self):
        super(ChessConvNet, self).__init__()
        self.in_channels = 3
        self.out_vector_length = 8 ** 2 + 7

        ################ just testing for now ##########################
        # remove this block afterwards
        self.covn1 = Conv2d(self.in_channels, 1, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.relu = ReLU()
        self.maxpool = MaxPool2d(kernel_size=2, stride=2)
        self.ll = Linear(in_features=160 ** 2, out_features=8 ** 2 + 7)
        #################################################################


    def forward(self, x):
        out = self.covn1(x)
        out = self.relu(out)
        out = self.maxpool(out)
        out = flatten(out, 1)
        out = self.ll(out)
        out = self.relu(out)

        return out

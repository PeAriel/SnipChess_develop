from torch.nn import Conv2d, ReLU, MaxPool2d, Linear, Module, Sequential
from torch import flatten

class ChessConvNet(Module):
    def __init__(self):
        super(ChessConvNet, self).__init__()
        self.in_channels = 3
        self.out_vector_length = 13

        self.features = Sequential(
            Conv2d(self.in_channels, 10, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            ReLU(inplace=True),
            Conv2d(10, 20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(inplace=True),
            Conv2d(20, 20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(inplace=True),
            Conv2d(20, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.classifier = Sequential(
            Linear(in_features=4000, out_features=500, bias=True),
            ReLU(inplace=True),
            Linear(in_features=500, out_features=500, bias=True),
            ReLU(inplace=True),
            Linear(in_features=500, out_features=13, bias=True),
        )



    def forward(self, x):
        out = self.features(x)
        out = flatten(out, 1)
        out = self.classifier(out)

        return out

class ChessBoardConvNet(Module):
    def __init__(self):
        super(ChessBoardConvNet, self).__init__()
        self.in_channels = 3
        self.out_vector_length = 8 ** 2 + 7

        # ################ just testing for now ##########################
        # # remove this block afterwards
        # self.covn1 = Conv2d(self.in_channels, 1, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        # self.relu = ReLU()
        # self.maxpool = MaxPool2d(kernel_size=2, stride=2)
        # self.ll = Linear(in_features=160 ** 2, out_features=8 ** 2 + 7)
        # #################################################################

        self.convs = Sequential(
            Conv2d(self.in_channels, 10, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            ReLU(inplace=True),
            Conv2d(10, 20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            Conv2d(20, 20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(inplace=True),
            Conv2d(20, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.lls = Sequential(
            Linear(in_features=(320 // 8) ** 2 * 10, out_features=2809 // 2),
            ReLU(inplace=True),
            Linear(in_features=2809 // 2, out_features=2809 // 2),
            ReLU(inplace=True),
            Linear(in_features=2809 // 2, out_features=2809 // 2),
            ReLU(inplace=True),
            Linear(in_features=2809 // 2, out_features=2809 // 6),
            ReLU(inplace=True),
            Linear(in_features=2809 // 6, out_features=2809 // 15),
            ReLU(inplace=True),
            Linear(in_features=2809 // 15, out_features=8 ** 2 + 7),
            ReLU(inplace=True),
        )



    def forward(self, x):
        out = self.convs(x)
        out = flatten(out, 1)
        out = self.lls(out)

        return out
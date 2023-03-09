from torch import nn


class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()

        self.model_lenet = nn.Sequential(
            # [3, 32, 32] => [6, 28, 28]
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
            # [6, 28, 28] => [6, 14, 14]
            nn.MaxPool2d(2),
            # [6, 14, 14] => [16, 10, 10]
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            # [16, 10, 10] =>[16, 5, 5]
            nn.MaxPool2d(2),
            # [16, 5, 5] => 展开
            nn.Flatten(),
            # 400 => 120
            nn.Linear(400, 120),
            nn.LeakyReLU(),
            nn.Linear(120, 84),
            nn.LeakyReLU(),
            nn.Linear(84, 10),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.model_lenet(x)
        return x





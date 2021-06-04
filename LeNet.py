from torch import nn
from Model_Class import Model


class LeNet(Model):
    def __init__(self):
        super(LeNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(4, 4)),
            nn.BatchNorm2d(120),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Conv2d(in_channels=120, out_channels=84, kernel_size=(1, 1)),
            nn.BatchNorm2d(84),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Conv2d(in_channels=84, out_channels=43, kernel_size=(1, 1))
        )

        super().set_optimizer()

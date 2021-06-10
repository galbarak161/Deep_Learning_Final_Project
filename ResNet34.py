from torch import nn
from Model_Class import Model
import torch.nn.functional as nn_functional


class ResNet34(Model):
    def __init__(self, use_spatial_transformer: bool):
        self.model_name = 'ResNet34'

        super(ResNet34, self).__init__(self.model_name, use_spatial_transformer)

        self.block_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.Dropout(self.dropout_p),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.Dropout(self.dropout_p),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.Dropout(self.dropout_p),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.Dropout(self.dropout_p),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(2048, 43)
        )

        super().set_optimizer()

    def forward(self, x):
        block1_out = self.block_1(x)

        block2_out = self.block_2(block1_out)

        block3_out = self.block_3(block2_out)

        block4_out = self.block_4(block3_out)

        block4_reshaped = block4_out.view(-1, 256 * 6 * 6)
        class_scores = self.classifier(block4_reshaped)
        probabilities = self.log_softMax(class_scores)

        return probabilities

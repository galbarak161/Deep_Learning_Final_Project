from torch import nn
from Model_Class import Model


class LeNet(Model):
    def __init__(self, use_spatial_transformer: bool, dataset_name: str, num_of_classes: int):
        self.model_name = 'LeNet' + '_' + dataset_name
        super().__init__(self.model_name, use_spatial_transformer)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5)),
            nn.BatchNorm2d(120),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Conv2d(in_channels=120, out_channels=84, kernel_size=(1, 1)),
            nn.BatchNorm2d(84),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Conv2d(in_channels=84, out_channels=num_of_classes, kernel_size=(1, 1))
        )

        super().set_optimizer()

    def model_forward(self, x):
        features = self.feature_extractor(x)

        class_scores = self.classifier(features)
        return class_scores

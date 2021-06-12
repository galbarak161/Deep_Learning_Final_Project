import os
import torch
import torch.nn.functional as nn_functional

from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from GTSRBDataset import TRAIN, VALID, TEST
from Transforms import getDefaultTransform
from plots.PlotsMeta import PATH_TO_PLOTS
from model.ModelMeta import PATH_TO_MODEL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):

    def __init__(self, model_name: str, use_spatial_transformer: bool, input_size: int):
        super().__init__()

        self.model_name = model_name
        if use_spatial_transformer:
            self.model_name += '_spatial_transformer'

        self.use_spatial_transformer = use_spatial_transformer
        self.input_size = input_size

        # Hyper parameters
        self.rate = 0.001
        self.weight_decay = 0.001
        self.dropout_p = 0.2

        self.loss_function = torch.nn.CrossEntropyLoss()
        self.log_softMax = nn.LogSoftmax(dim=1)

        self.optimizer = None
        self.scheduler = None

        self.localization = None
        self.fc_loc = None

        if use_spatial_transformer:
            self.init_spatial_transformer()

    def init_spatial_transformer(self):
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=(7, 7)),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=(5, 5)),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.8)

        self.to(DEVICE)

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = nn_functional.affine_grid(theta, x.size(), align_corners=True)
        x = nn_functional.grid_sample(x, grid, align_corners=True)
        return x

    def forward(self, x):
        if self.use_spatial_transformer:
            x = self.stn(x)

        features = self.feature_extractor(x)
        class_scores = self.classifier(features)
        probabilities = self.log_softMax(class_scores)
        return probabilities

    def get_predictions(self, path_to_image):
        img = Image.open(path_to_image).convert('RGB')
        img = getDefaultTransform(self.input_size)(img)
        img = img.unsqueeze(0).to(DEVICE)

        self.eval()
        with torch.no_grad():
            probabilities = self(img).squeeze().to(DEVICE)
            prediction = torch.argmax(probabilities).item()

            return prediction

    def calculate_accuracy_and_loss(self, data_set: DataLoader):
        self.eval()
        with torch.no_grad():
            n_correct = 0
            total = 0

            losses = []

            for data, labels in data_set:
                data, labels = data.to(DEVICE), labels.to(DEVICE)

                # calculate output
                predictions_probabilities = self(data).squeeze().to(DEVICE)

                loss = self.loss_function(predictions_probabilities, labels).to(DEVICE)
                losses.append(loss.detach())
                # get the prediction
                predictions = torch.argmax(predictions_probabilities, dim=1)
                n_correct += torch.sum(predictions == labels).item()
                total += data.shape[0]

            return n_correct / total, torch.mean(torch.stack(losses).cpu(), dim=0)

    def train_model(self, epochs: int, data_loaders: dict):
        # early stopping params
        best_validation_acc = 0
        patience_limit = 20
        patience_counter = 0
        need_to_stop = False
        best_model_epoch_number = 0
        model_path = os.path.join(PATH_TO_MODEL, f'{self.model_name}.pth')

        # results lists
        train_accuracies = []
        train_losses = []
        val_accuracies = []
        valid_losses = []

        # training loop
        epoch = 0
        while epoch < epochs and not need_to_stop:

            # train phase
            self.train()
            for data, labels in data_loaders[TRAIN]:
                data, labels = data.to(DEVICE), labels.to(DEVICE)
                self.optimizer.zero_grad()
                predictions = self(data).to(DEVICE)
                predictions = predictions.squeeze()

                loss = self.loss_function(predictions, labels).to(DEVICE)
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()

            # calc acc and losses
            if epoch % 10 == 0:
                print(f'Calculating train accuracy and loss for epoch {epoch}')
            train_acc, train_loss = self.calculate_accuracy_and_loss(data_loaders[TRAIN])
            train_accuracies.append(train_acc)
            train_losses.append(train_loss)
            if epoch % 10 == 0:
                print(f'Calculating validation accuracy and loss for epoch {epoch}')
            val_acc, val_loss = self.calculate_accuracy_and_loss(data_loaders[VALID])
            valid_losses.append(val_loss)
            val_accuracies.append(val_acc)

            # early stopping check
            if val_acc > best_validation_acc:
                best_validation_acc = val_acc
                torch.save(self.state_dict(), model_path)
                patience_counter = 0
                best_model_epoch_number = epoch
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    need_to_stop = True
                    print(f'early stopping after {epoch} / {epochs}')

            epoch += 1

        # accuracy plots
        fig = plt.figure()
        val_acc_plot_name = f'Accuracy per epoch - {self.model_name}'
        plt.title(val_acc_plot_name)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')

        plt.plot(best_model_epoch_number, val_accuracies[best_model_epoch_number], 'r*',
                 label='Best Validation Accuracy')
        plt.legend()
        fig.savefig(os.path.join(PATH_TO_PLOTS, val_acc_plot_name))
        plt.close(fig)
        plt.clf()

        # loss plots
        val_loss_plot_name = f'Loss per epoch - {self.model_name}'
        fig = plt.figure()
        plt.title(val_loss_plot_name)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(valid_losses, label='Validation Loss')

        plt.plot(best_model_epoch_number, valid_losses[best_model_epoch_number], 'r*',
                 label='Best Validation Accuracy')
        plt.legend()
        fig.savefig(os.path.join(PATH_TO_PLOTS, val_loss_plot_name))
        plt.close(fig)
        plt.clf()

        # final results
        self.load_state_dict(torch.load(model_path))
        print('Calculating test accuracy')
        test_acc, test_loss = self.calculate_accuracy_and_loss(data_loaders[TEST])
        print(f'Train: Accuracy = {(train_accuracies[best_model_epoch_number] * 100):.2f}%, '
              f'Avg Loss = {train_losses[best_model_epoch_number]:.2f}')
        print(f'Validation: Accuracy = {(val_accuracies[best_model_epoch_number] * 100):.2f}%, '
              f'Avg Loss = {valid_losses[best_model_epoch_number]:.2f}')
        print(f'Test: Accuracy = {(test_acc * 100):.2f}%, Avg Loss = {test_loss:.2f}')
        print()

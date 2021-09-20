import os
import time
import torch
from GTSRBDataset import TEST
from LeNet import LeNet
from VGG16 import VGG16
from ResNet34 import ResNet34
from main_Train import init_data_and_get_loaders, print_time
from model.ModelMeta import PATH_TO_MODEL
import warnings

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main_test():
    image_size = 32
    data = init_data_and_get_loaders(image_size)

    for filename in os.listdir(PATH_TO_MODEL):
        if filename.endswith(".pth"):
            start_time = time.time()
            model_path = os.path.join(PATH_TO_MODEL, filename)
            use_spatial_transformer = False
            if 'spatial_transformer' in filename:
                use_spatial_transformer = True

            if 'LeNet' in filename:
                model_class = LeNet
            elif 'VGG16' in filename:
                model_class = VGG16
            else:
                model_class = ResNet34

            test_model(data, model_class, model_path, use_spatial_transformer)

            end_time = time.time()
            print_time(end_time - start_time)


def test_model(data, model_class, model_path, use_spatial_transformer):
    model = model_class(use_spatial_transformer=use_spatial_transformer)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
    print()
    print('#########################################################################')
    print(f'Test model: {model_class} spatial_transformer: {use_spatial_transformer}')
    print('Calculating test accuracy...')
    test_acc, test_loss = model.calculate_accuracy_and_loss(data[TEST])
    print(f'Test: Accuracy = {(test_acc * 100):.2f}%, Avg Loss = {test_loss:.2f}')


if __name__ == '__main__':
    print(f'Using {DEVICE} as device')
    main_test()

"""
#########################################################################
Test model: <class 'LeNet.LeNet'> spatial_transformer: False
Calculating test accuracy...
Test: Accuracy = 95.69%, Avg Loss = 0.16
	Time taken: 00:00:05.87


#########################################################################
Test model: <class 'LeNet.LeNet'> spatial_transformer: True
Calculating test accuracy...
Test: Accuracy = 98.15%, Avg Loss = 0.08
	Time taken: 00:00:06.80


#########################################################################
Test model: <class 'ResNet34.ResNet34'> spatial_transformer: False
Calculating test accuracy...
Test: Accuracy = 98.83%, Avg Loss = 0.05
	Time taken: 00:01:29.09


#########################################################################
Test model: <class 'ResNet34.ResNet34'> spatial_transformer: True
Calculating test accuracy...
Test: Accuracy = 98.82%, Avg Loss = 0.05
	Time taken: 00:01:43.23


#########################################################################
Test model: <class 'VGG16.VGG16'> spatial_transformer: False
Calculating test accuracy...
Test: Accuracy = 97.76%, Avg Loss = 0.13
	Time taken: 00:10:39.27


#########################################################################
Test model: <class 'VGG16.VGG16'> spatial_transformer: True
Calculating test accuracy...
Test: Accuracy = 96.07%, Avg Loss = 0.20
	Time taken: 00:10:14.89
"""

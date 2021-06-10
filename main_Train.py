import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
import time

#from torchsummary import summary

from GTSRBDataset import GTSRBDataset, TRAIN, TEST, VALID
from LeNet import LeNet
from Transforms import getTransforms, getDefaultTransform
from VGG16 import VGG16
from Model_Class import DEVICE
from ResNet34 import ResNet34


def print_time(time_taken: float) -> None:
    """
    Utility function for time printing
    :param time_taken: the time we need to print
    """
    hours, rem = divmod(time_taken, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\tTime taken: {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds))


def init_data_and_get_loaders(imageSize):
    print('\nInitiating data...')
    # dataset initialization
    default_transform = getDefaultTransform(imageSize)
    def_dataset = GTSRBDataset(transform=default_transform)
    datasets = [def_dataset]

    special_transforms_ratio = 0.000005

    train_set_size = len(def_dataset)
    indices = list(range(train_set_size))
    for data_transform in getTransforms(imageSize).values():
        np.random.shuffle(indices)
        split = int(np.floor(special_transforms_ratio * train_set_size))
        transform_sample = indices[:split]

        special_dataset = GTSRBDataset(transform_sample, transform=data_transform)
        datasets.append(special_dataset)

    train_dataset = ConcatDataset(datasets)
    old_train_dataset = GTSRBDataset(transform=default_transform)
    test_dataset = GTSRBDataset(train=False, transform=default_transform)

    print(f'old train set size: {len(old_train_dataset)}, new train set size: {len(train_dataset)}\n\n')

    validationRatio = 0.2

    indices = list(range(len(train_dataset)))
    np.random.shuffle(indices)
    split = int(np.floor(validationRatio * len(train_dataset)))
    trainSample = SubsetRandomSampler(indices[:split])
    validSample = SubsetRandomSampler(indices[split:])

    batch_size = 256
    num_workers = 4

    dataLoaders = {
        TRAIN: DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=trainSample),
        VALID: DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=validSample),
        TEST: DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    }

    return dataLoaders

def time_print_train_model(use_spatial_transformer, input_size, epochs, data_loaders, model_class):
    start_time = time.time()
    model = model_class(use_spatial_transformer=use_spatial_transformer)
    print(model)
    #summary(model, input_size=(3, input_size, input_size))
    model.train_model(epochs, data_loaders)
    end_time = time.time()
    print_time(end_time - start_time)

def main():

    epochs = 1

    # LeNet
    leNet_input_size = 32
    leNet_data_loaders = init_data_and_get_loaders(leNet_input_size)
    time_print_train_model(False, leNet_input_size, epochs, leNet_data_loaders, LeNet)

    time_print_train_model(True, leNet_input_size, epochs, leNet_data_loaders, LeNet)

    # VGG16
    vgg16_input_size = 224
    vgg16_data_loaders = init_data_and_get_loaders(vgg16_input_size)
    time_print_train_model(False, vgg16_input_size, epochs, vgg16_data_loaders, VGG16)
    time_print_train_model(True, vgg16_input_size, epochs, vgg16_data_loaders, VGG16)

    # ResNet
    resNet_input_size = 48
    resNet_data_loaders = init_data_and_get_loaders(resNet_input_size)
    time_print_train_model(False, resNet_input_size, epochs, resNet_data_loaders, ResNet34)
    time_print_train_model(True, resNet_input_size, epochs, resNet_data_loaders, ResNet34)


if __name__ == '__main__':
    print(f'Using {DEVICE} as device')
    main()

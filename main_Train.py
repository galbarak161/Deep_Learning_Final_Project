import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
import time
from torchvision import datasets
from torchvision.transforms import transforms
# from torchsummary import summary

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


def init_data_and_get_loaders(image_size):
    print('\nInitiating data...')

    # dataset initialization
    default_transform = getDefaultTransform(image_size)
    def_dataset = GTSRBDataset(transform=default_transform)
    datasets_list = [def_dataset]

    special_transforms_ratio = 0.5

    train_set_size = len(def_dataset)
    indices = list(range(train_set_size))
    for data_transform in getTransforms(image_size).values():
        np.random.shuffle(indices)
        split = int(np.floor(special_transforms_ratio * train_set_size))
        transform_sample = indices[:split]

        special_dataset = GTSRBDataset(transform_sample, transform=data_transform)
        datasets_list.append(special_dataset)

    train_dataset = ConcatDataset(datasets_list)
    old_train_dataset = GTSRBDataset(transform=default_transform)
    test_dataset = GTSRBDataset(train=False, transform=default_transform)

    print(f'old train set size: {len(old_train_dataset)}, new train set size: {len(train_dataset)}\n\n')

    return split_data_and_get_loaders(train_dataset, test_dataset)


def init_cifar10_data_and_get_loaders(image_size):
    data_mean = (125.30691805, 122.95039414, 113.86538318)
    data_std = (62.99321928, 62.08870764, 66.70489964)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std)
    ])

    train_set = datasets.CIFAR10('./cifar10_data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10('./cifar10_data', download=True, train=False, transform=transform)

    return split_data_and_get_loaders(train_set, test_set)


def split_data_and_get_loaders(train_set, test_set, ):
    validation_rate = 0.2

    indices = list(range(len(train_set)))
    np.random.shuffle(indices)
    split = int(np.floor(validation_rate * len(train_set)))
    train_sample = SubsetRandomSampler(indices[:split])
    valid_sample = SubsetRandomSampler(indices[split:])

    batch_size = 256
    num_workers = 4

    dataLoaders = {
        TRAIN: DataLoader(train_set, sampler=train_sample, batch_size=batch_size, num_workers=num_workers),
        VALID: DataLoader(train_set, sampler=valid_sample, batch_size=batch_size, num_workers=num_workers),
        TEST: DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
    }

    return dataLoaders


def time_print_train_model(use_spatial_transformer, input_size, epochs, data_loaders, model_class,
                           dataset_name, num_of_classes):
    start_time = time.time()
    model = model_class(use_spatial_transformer=use_spatial_transformer, dataset_name=dataset_name,
                        num_of_classes=num_of_classes)
    print(model)
    """print(model_class.__name__)
    summary(model, input_size=(3, input_size, input_size))"""
    model.train_model(epochs, data_loaders)
    end_time = time.time()
    print_time(end_time - start_time)


def main():
    epochs = 2
    image_size = 32
    GTSRB_name = 'GTSRB'
    GTSRB_num_of_classes = 43
    data = init_data_and_get_loaders(image_size)

    # LeNet
    time_print_train_model(False, image_size, epochs, data, LeNet, GTSRB_name, GTSRB_num_of_classes)
    time_print_train_model(True, image_size, epochs, data, LeNet, GTSRB_name, GTSRB_num_of_classes)

    # VGG16
    time_print_train_model(False, image_size, epochs, data, VGG16, GTSRB_name, GTSRB_num_of_classes)
    time_print_train_model(True, image_size, epochs, data, VGG16, GTSRB_name, GTSRB_num_of_classes)

    # ResNet
    time_print_train_model(False, image_size, epochs, data, ResNet34, GTSRB_name, GTSRB_num_of_classes)
    time_print_train_model(True, image_size, epochs, data, ResNet34, GTSRB_name, GTSRB_num_of_classes)

    cifar10_name = 'cifar10'
    cifar10_num_of_classes = 10
    cifar10_data = init_cifar10_data_and_get_loaders(image_size)

    # LeNet
    time_print_train_model(False, image_size, epochs, cifar10_data, LeNet, cifar10_name, cifar10_num_of_classes)
    time_print_train_model(True, image_size, epochs, cifar10_data, LeNet, cifar10_name, cifar10_num_of_classes)

    # VGG16
    time_print_train_model(False, image_size, epochs, cifar10_data, VGG16, cifar10_name, cifar10_num_of_classes)
    time_print_train_model(True, image_size, epochs, cifar10_data, VGG16, cifar10_name, cifar10_num_of_classes)

    # # ResNet
    time_print_train_model(False, image_size, epochs, cifar10_data, ResNet34, cifar10_name, cifar10_num_of_classes)
    time_print_train_model(True, image_size, epochs, cifar10_data, ResNet34, cifar10_name, cifar10_num_of_classes)


if __name__ == '__main__':
    print(f'Using {DEVICE} as device')
    main()

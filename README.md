# Deep_Learning_Final_Project
Final project in Deep Learning course. The University of Haifa.


---
ADD DESCRIPTION

---

## Load The Data
we used the original Kaggle dataset from [here](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).

The dataset contains almost 40,000 images of different traffic signs. It is further classified into 43 different classes. 
Then we used several techniques of dataset augmentations to create more training and validation data.

```
    old data size: 39209, new data size: 274457
```

## Networks Visualization   

```
--------------------------- LeNet 5 ----------------------------
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 6, 28, 28]             456
              ReLU-2            [-1, 6, 28, 28]               0
         MaxPool2d-3            [-1, 6, 14, 14]               0
            Conv2d-4           [-1, 16, 10, 10]           2,416
              ReLU-5           [-1, 16, 10, 10]               0
         MaxPool2d-6             [-1, 16, 5, 5]               0
            Conv2d-7            [-1, 120, 1, 1]          48,120
       BatchNorm2d-8            [-1, 120, 1, 1]             240
              ReLU-9            [-1, 120, 1, 1]               0
          Dropout-10            [-1, 120, 1, 1]               0
           Conv2d-11             [-1, 84, 1, 1]          10,164
      BatchNorm2d-12             [-1, 84, 1, 1]             168
             ReLU-13             [-1, 84, 1, 1]               0
          Dropout-14             [-1, 84, 1, 1]               0
           Conv2d-15             [-1, 43, 1, 1]           3,655
       LogSoftmax-16             [-1, 43, 1, 1]               0
================================================================
Total params: 65,219
----------------------------------------------------------------

--------------------------- VGG 16 -----------------------------
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,792
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
            Conv2d-4           [-1, 64, 32, 32]          36,928
       BatchNorm2d-5           [-1, 64, 32, 32]             128
              ReLU-6           [-1, 64, 32, 32]               0
         MaxPool2d-7           [-1, 64, 16, 16]               0
            Conv2d-8          [-1, 128, 16, 16]          73,856
       BatchNorm2d-9          [-1, 128, 16, 16]             256
             ReLU-10          [-1, 128, 16, 16]               0
           Conv2d-11          [-1, 128, 16, 16]         147,584
      BatchNorm2d-12          [-1, 128, 16, 16]             256
             ReLU-13          [-1, 128, 16, 16]               0
        MaxPool2d-14            [-1, 128, 8, 8]               0
           Conv2d-15            [-1, 256, 8, 8]         295,168
      BatchNorm2d-16            [-1, 256, 8, 8]             512
             ReLU-17            [-1, 256, 8, 8]               0
           Conv2d-18            [-1, 256, 8, 8]         590,080
      BatchNorm2d-19            [-1, 256, 8, 8]             512
             ReLU-20            [-1, 256, 8, 8]               0
           Conv2d-21            [-1, 256, 8, 8]         590,080
      BatchNorm2d-22            [-1, 256, 8, 8]             512
             ReLU-23            [-1, 256, 8, 8]               0
        MaxPool2d-24            [-1, 256, 4, 4]               0
           Conv2d-25            [-1, 512, 4, 4]       1,180,160
      BatchNorm2d-26            [-1, 512, 4, 4]           1,024
             ReLU-27            [-1, 512, 4, 4]               0
           Conv2d-28            [-1, 512, 4, 4]       2,359,808
      BatchNorm2d-29            [-1, 512, 4, 4]           1,024
             ReLU-30            [-1, 512, 4, 4]               0
           Conv2d-31            [-1, 512, 4, 4]       2,359,808
      BatchNorm2d-32            [-1, 512, 4, 4]           1,024
             ReLU-33            [-1, 512, 4, 4]               0
        MaxPool2d-34            [-1, 512, 2, 2]               0
           Conv2d-35            [-1, 512, 2, 2]       2,359,808
      BatchNorm2d-36            [-1, 512, 2, 2]           1,024
             ReLU-37            [-1, 512, 2, 2]               0
           Conv2d-38            [-1, 512, 2, 2]       2,359,808
      BatchNorm2d-39            [-1, 512, 2, 2]           1,024
             ReLU-40            [-1, 512, 2, 2]               0
           Conv2d-41            [-1, 512, 2, 2]       2,359,808
      BatchNorm2d-42            [-1, 512, 2, 2]           1,024
             ReLU-43            [-1, 512, 2, 2]               0
        MaxPool2d-44            [-1, 512, 1, 1]               0
AdaptiveAvgPool2d-45            [-1, 512, 7, 7]               0
           Linear-46                 [-1, 4096]     102,764,544
             ReLU-47                 [-1, 4096]               0
          Dropout-48                 [-1, 4096]               0
           Linear-49                 [-1, 4096]      16,781,312
             ReLU-50                 [-1, 4096]               0
          Dropout-51                 [-1, 4096]               0
           Linear-52                   [-1, 43]         176,171
       LogSoftmax-53                   [-1, 43]               0
================================================================
Total params: 134,445,163
----------------------------------------------------------------

------------------------- ResNet 34 ----------------------------
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             896
       BatchNorm2d-2           [-1, 32, 32, 32]              64
              ReLU-3           [-1, 32, 32, 32]               0
            Conv2d-4           [-1, 32, 32, 32]           9,248
       BatchNorm2d-5           [-1, 32, 32, 32]              64
              ReLU-6           [-1, 32, 32, 32]               0
            Conv2d-7           [-1, 32, 32, 32]           9,248
           Dropout-8           [-1, 32, 32, 32]               0
         MaxPool2d-9           [-1, 32, 16, 16]               0
      BatchNorm2d-10           [-1, 32, 16, 16]              64
             ReLU-11           [-1, 32, 16, 16]               0
           Conv2d-12           [-1, 64, 16, 16]          18,496
      BatchNorm2d-13           [-1, 64, 16, 16]             128
             ReLU-14           [-1, 64, 16, 16]               0
           Conv2d-15           [-1, 64, 16, 16]          36,928
      BatchNorm2d-16           [-1, 64, 16, 16]             128
             ReLU-17           [-1, 64, 16, 16]               0
           Conv2d-18           [-1, 64, 16, 16]          36,928
          Dropout-19           [-1, 64, 16, 16]               0
        MaxPool2d-20             [-1, 64, 8, 8]               0
      BatchNorm2d-21             [-1, 64, 8, 8]             128
             ReLU-22             [-1, 64, 8, 8]               0
           Conv2d-23            [-1, 128, 8, 8]          73,856
      BatchNorm2d-24            [-1, 128, 8, 8]             256
             ReLU-25            [-1, 128, 8, 8]               0
           Conv2d-26            [-1, 128, 8, 8]         147,584
      BatchNorm2d-27            [-1, 128, 8, 8]             256
             ReLU-28            [-1, 128, 8, 8]               0
           Conv2d-29            [-1, 128, 8, 8]         147,584
          Dropout-30            [-1, 128, 8, 8]               0
        MaxPool2d-31            [-1, 128, 4, 4]               0
      BatchNorm2d-32            [-1, 128, 4, 4]             256
             ReLU-33            [-1, 128, 4, 4]               0
           Conv2d-34            [-1, 256, 4, 4]         295,168
      BatchNorm2d-35            [-1, 256, 4, 4]             512
             ReLU-36            [-1, 256, 4, 4]               0
           Conv2d-37            [-1, 256, 4, 4]         590,080
      BatchNorm2d-38            [-1, 256, 4, 4]             512
             ReLU-39            [-1, 256, 4, 4]               0
           Conv2d-40            [-1, 256, 4, 4]         590,080
          Dropout-41            [-1, 256, 4, 4]               0
      BatchNorm2d-42            [-1, 256, 4, 4]             512
             ReLU-43            [-1, 256, 4, 4]               0
           Linear-44                 [-1, 2048]       8,390,656
      BatchNorm1d-45                 [-1, 2048]           4,096
             ReLU-46                 [-1, 2048]               0
          Dropout-47                 [-1, 2048]               0
           Linear-48                   [-1, 43]          88,107
       LogSoftmax-49                   [-1, 43]               0
================================================================
Total params: 10,441,835
----------------------------------------------------------------
```

## Training Results
```
--------------------------- LeNet 5 ----------------------------
Train: Accuracy = 97.54%, Avg Loss = 0.10
Validation: Accuracy = 94.99%, Avg Loss = 0.18
Test: Accuracy = 95.69%, Avg Loss = 0.16

	Time taken: 01:00:40.54

--------------------------- VGG 16 -----------------------------
Train: Accuracy = 99.72%, Avg Loss = 0.01
Validation: Accuracy = 98.45%, Avg Loss = 0.07
Test: Accuracy = 97.76%, Avg Loss = 0.13

	Time taken: 02:52:46.96

------------------------- ResNet 34 ----------------------------
Train: Accuracy = 99.81%, Avg Loss = 0.01
Validation: Accuracy = 98.90%, Avg Loss = 0.05
Test: Accuracy = 98.83%, Avg Loss = 0.05

	Time taken: 01:36:23.83
```

## Spatial Transformer
```
  (localization): Sequential(
    (0): Conv2d(3, 8, kernel_size=(7, 7), stride=(1, 1))
    (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (2): ReLU(inplace=True)
    (3): Conv2d(8, 10, kernel_size=(5, 5), stride=(1, 1))
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): ReLU(inplace=True)
  )
  (fc_loc): Sequential(
    (0): Linear(in_features=160, out_features=32, bias=True)
    (1): ReLU(inplace=True)
    (2): Linear(in_features=32, out_features=6, bias=True)
  )
  
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]           1,184
         MaxPool2d-2            [-1, 8, 13, 13]               0
              ReLU-3            [-1, 8, 13, 13]               0
            Conv2d-4             [-1, 10, 9, 9]           2,010
         MaxPool2d-5             [-1, 10, 4, 4]               0
              ReLU-6             [-1, 10, 4, 4]               0
            Linear-7                   [-1, 32]           5,152
              ReLU-8                   [-1, 32]               0
            Linear-9                    [-1, 6]             198
================================================================
```

## Training Results with Spatial Transformer
```
--------------------------- LeNet 5 ----------------------------
Train: Accuracy = 98.86%, Avg Loss = 0.05
Validation: Accuracy = 97.60%, Avg Loss = 0.09
Test: Accuracy = 98.15%, Avg Loss = 0.08

	Time taken: 01:26:54.03

--------------------------- VGG 16 -----------------------------
Train: Accuracy = 99.38%, Avg Loss = 0.02
Validation: Accuracy = 97.25%, Avg Loss = 0.11
Test: Accuracy = 96.07%, Avg Loss = 0.20

	Time taken: 02:21:11.51

------------------------- ResNet 34 ----------------------------
Train: Accuracy = 99.72%, Avg Loss = 0.01
Validation: Accuracy = 98.88%, Avg Loss = 0.05
Test: Accuracy = 98.82%, Avg Loss = 0.05

	Time taken: 01:50:42.37
```
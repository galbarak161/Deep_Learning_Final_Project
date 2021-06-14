# Deep_Learning_Final_Project
Final project in Deep Learning course. The University of Haifa.




C:\Users\Spankmaster\AppData\Local\Microsoft\WindowsApps\python3.8.exe "C:/Users/Spankmaster/Documents/BS.c - מדעי המחשב/שנה שלישית/סמסטר ב/למידה עמוקה/תרגילי בית/Deep_Learning_Final_Project/main_Train.py"
Using cpu as device

Initiating data...
old train set size: 39209, new train set size: 39209


LeNet(
  (loss_function): CrossEntropyLoss()
  (log_softMax): LogSoftmax(dim=1)
  (feature_extractor): Sequential(
    (0): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
    (4): ReLU()
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Conv2d(16, 120, kernel_size=(5, 5), stride=(1, 1))
    (1): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Dropout(p=0.2, inplace=False)
    (4): Conv2d(120, 84, kernel_size=(1, 1), stride=(1, 1))
    (5): BatchNorm2d(84, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU()
    (7): Dropout(p=0.2, inplace=False)
    (8): Conv2d(84, 43, kernel_size=(1, 1), stride=(1, 1))
  )
)
LeNet
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
Trainable params: 65,219
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 0.12
Params size (MB): 0.25
Estimated Total Size (MB): 0.38
----------------------------------------------------------------
Calculating train accuracy and loss for epoch 0
Calculating validation accuracy and loss for epoch 0
Calculating test accuracy
Train: Accuracy = 39.00%, Avg Loss = 3.17
Validation: Accuracy = 38.35%, Avg Loss = 3.18
Test: Accuracy = 36.18%, Avg Loss = 3.19

	Time taken: 00:00:38.47

LeNet(
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
  (loss_function): CrossEntropyLoss()
  (log_softMax): LogSoftmax(dim=1)
  (feature_extractor): Sequential(
    (0): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
    (4): ReLU()
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Conv2d(16, 120, kernel_size=(5, 5), stride=(1, 1))
    (1): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Dropout(p=0.2, inplace=False)
    (4): Conv2d(120, 84, kernel_size=(1, 1), stride=(1, 1))
    (5): BatchNorm2d(84, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU()
    (7): Dropout(p=0.2, inplace=False)
    (8): Conv2d(84, 43, kernel_size=(1, 1), stride=(1, 1))
  )
)
LeNet
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
           Conv2d-10            [-1, 6, 28, 28]             456
             ReLU-11            [-1, 6, 28, 28]               0
        MaxPool2d-12            [-1, 6, 14, 14]               0
           Conv2d-13           [-1, 16, 10, 10]           2,416
             ReLU-14           [-1, 16, 10, 10]               0
        MaxPool2d-15             [-1, 16, 5, 5]               0
           Conv2d-16            [-1, 120, 1, 1]          48,120
      BatchNorm2d-17            [-1, 120, 1, 1]             240
             ReLU-18            [-1, 120, 1, 1]               0
          Dropout-19            [-1, 120, 1, 1]               0
           Conv2d-20             [-1, 84, 1, 1]          10,164
      BatchNorm2d-21             [-1, 84, 1, 1]             168
             ReLU-22             [-1, 84, 1, 1]               0
          Dropout-23             [-1, 84, 1, 1]               0
           Conv2d-24             [-1, 43, 1, 1]           3,655
       LogSoftmax-25             [-1, 43, 1, 1]               0
================================================================
Total params: 73,763
Trainable params: 73,763
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 0.19
Params size (MB): 0.28
Estimated Total Size (MB): 0.48
----------------------------------------------------------------
Calculating train accuracy and loss for epoch 0
Calculating validation accuracy and loss for epoch 0
Calculating test accuracy
Train: Accuracy = 37.46%, Avg Loss = 3.27
Validation: Accuracy = 37.14%, Avg Loss = 3.27
Test: Accuracy = 35.21%, Avg Loss = 3.29

	Time taken: 00:00:29.52

VGG16(
  (loss_function): CrossEntropyLoss()
  (log_softMax): LogSoftmax(dim=1)
  (feature_extractor): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (9): ReLU(inplace=True)
    (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (12): ReLU(inplace=True)
    (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (16): ReLU(inplace=True)
    (17): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (19): ReLU(inplace=True)
    (20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (21): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (26): ReLU(inplace=True)
    (27): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (28): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (29): ReLU(inplace=True)
    (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (31): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (32): ReLU(inplace=True)
    (33): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (35): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (36): ReLU(inplace=True)
    (37): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (38): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (39): ReLU(inplace=True)
    (40): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (41): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (42): ReLU(inplace=True)
    (43): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avg_pool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=43, bias=True)
  )
)
VGG16
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
Trainable params: 134,445,163
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 6.95
Params size (MB): 512.87
Estimated Total Size (MB): 519.83
----------------------------------------------------------------
Calculating train accuracy and loss for epoch 0
Calculating validation accuracy and loss for epoch 0
Calculating test accuracy
Train: Accuracy = 6.34%, Avg Loss = 3.73
Validation: Accuracy = 6.03%, Avg Loss = 3.83
Test: Accuracy = 7.28%, Avg Loss = 3.75

	Time taken: 00:06:23.93

VGG16(
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
  (loss_function): CrossEntropyLoss()
  (log_softMax): LogSoftmax(dim=1)
  (feature_extractor): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (9): ReLU(inplace=True)
    (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (12): ReLU(inplace=True)
    (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (16): ReLU(inplace=True)
    (17): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (19): ReLU(inplace=True)
    (20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (21): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (26): ReLU(inplace=True)
    (27): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (28): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (29): ReLU(inplace=True)
    (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (31): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (32): ReLU(inplace=True)
    (33): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (35): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (36): ReLU(inplace=True)
    (37): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (38): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (39): ReLU(inplace=True)
    (40): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (41): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (42): ReLU(inplace=True)
    (43): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avg_pool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=43, bias=True)
  )
)
VGG16
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
           Conv2d-10           [-1, 64, 32, 32]           1,792
      BatchNorm2d-11           [-1, 64, 32, 32]             128
             ReLU-12           [-1, 64, 32, 32]               0
           Conv2d-13           [-1, 64, 32, 32]          36,928
      BatchNorm2d-14           [-1, 64, 32, 32]             128
             ReLU-15           [-1, 64, 32, 32]               0
        MaxPool2d-16           [-1, 64, 16, 16]               0
           Conv2d-17          [-1, 128, 16, 16]          73,856
      BatchNorm2d-18          [-1, 128, 16, 16]             256
             ReLU-19          [-1, 128, 16, 16]               0
           Conv2d-20          [-1, 128, 16, 16]         147,584
      BatchNorm2d-21          [-1, 128, 16, 16]             256
             ReLU-22          [-1, 128, 16, 16]               0
        MaxPool2d-23            [-1, 128, 8, 8]               0
           Conv2d-24            [-1, 256, 8, 8]         295,168
      BatchNorm2d-25            [-1, 256, 8, 8]             512
             ReLU-26            [-1, 256, 8, 8]               0
           Conv2d-27            [-1, 256, 8, 8]         590,080
      BatchNorm2d-28            [-1, 256, 8, 8]             512
             ReLU-29            [-1, 256, 8, 8]               0
           Conv2d-30            [-1, 256, 8, 8]         590,080
      BatchNorm2d-31            [-1, 256, 8, 8]             512
             ReLU-32            [-1, 256, 8, 8]               0
        MaxPool2d-33            [-1, 256, 4, 4]               0
           Conv2d-34            [-1, 512, 4, 4]       1,180,160
      BatchNorm2d-35            [-1, 512, 4, 4]           1,024
             ReLU-36            [-1, 512, 4, 4]               0
           Conv2d-37            [-1, 512, 4, 4]       2,359,808
      BatchNorm2d-38            [-1, 512, 4, 4]           1,024
             ReLU-39            [-1, 512, 4, 4]               0
           Conv2d-40            [-1, 512, 4, 4]       2,359,808
      BatchNorm2d-41            [-1, 512, 4, 4]           1,024
             ReLU-42            [-1, 512, 4, 4]               0
        MaxPool2d-43            [-1, 512, 2, 2]               0
           Conv2d-44            [-1, 512, 2, 2]       2,359,808
      BatchNorm2d-45            [-1, 512, 2, 2]           1,024
             ReLU-46            [-1, 512, 2, 2]               0
           Conv2d-47            [-1, 512, 2, 2]       2,359,808
      BatchNorm2d-48            [-1, 512, 2, 2]           1,024
             ReLU-49            [-1, 512, 2, 2]               0
           Conv2d-50            [-1, 512, 2, 2]       2,359,808
      BatchNorm2d-51            [-1, 512, 2, 2]           1,024
             ReLU-52            [-1, 512, 2, 2]               0
        MaxPool2d-53            [-1, 512, 1, 1]               0
AdaptiveAvgPool2d-54            [-1, 512, 7, 7]               0
           Linear-55                 [-1, 4096]     102,764,544
             ReLU-56                 [-1, 4096]               0
          Dropout-57                 [-1, 4096]               0
           Linear-58                 [-1, 4096]      16,781,312
             ReLU-59                 [-1, 4096]               0
          Dropout-60                 [-1, 4096]               0
           Linear-61                   [-1, 43]         176,171
       LogSoftmax-62                   [-1, 43]               0
================================================================
Total params: 134,453,707
Trainable params: 134,453,707
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 7.02
Params size (MB): 512.90
Estimated Total Size (MB): 519.93
----------------------------------------------------------------
Calculating train accuracy and loss for epoch 0
Calculating validation accuracy and loss for epoch 0
Calculating test accuracy
Train: Accuracy = 5.78%, Avg Loss = 3.72
Validation: Accuracy = 5.62%, Avg Loss = 3.77
Test: Accuracy = 5.75%, Avg Loss = 3.77

	Time taken: 00:06:30.29

ResNet34(
  (loss_function): CrossEntropyLoss()
  (log_softMax): LogSoftmax(dim=1)
  (block_1): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
    (6): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): Dropout(p=0.2, inplace=False)
    (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (9): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): ReLU()
  )
  (block_2): Sequential(
    (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
    (6): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): Dropout(p=0.2, inplace=False)
    (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (9): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): ReLU()
  )
  (block_3): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
    (6): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): Dropout(p=0.2, inplace=False)
    (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (9): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): ReLU()
  )
  (block_4): Sequential(
    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
    (6): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): Dropout(p=0.2, inplace=False)
    (8): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (9): ReLU()
  )
  (classifier): Sequential(
    (0): Linear(in_features=4096, out_features=2048, bias=True)
    (1): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Dropout(p=0.2, inplace=False)
    (4): Linear(in_features=2048, out_features=43, bias=True)
  )
)
ResNet34
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
Trainable params: 10,441,835
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 4.20
Params size (MB): 39.83
Estimated Total Size (MB): 44.05
----------------------------------------------------------------
Calculating train accuracy and loss for epoch 0
Calculating validation accuracy and loss for epoch 0
Calculating test accuracy
Train: Accuracy = 13.28%, Avg Loss = 3.38
Validation: Accuracy = 12.80%, Avg Loss = 3.44
Test: Accuracy = 14.11%, Avg Loss = 3.43

	Time taken: 00:02:30.17

ResNet34(
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
  (loss_function): CrossEntropyLoss()
  (log_softMax): LogSoftmax(dim=1)
  (block_1): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
    (6): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): Dropout(p=0.2, inplace=False)
    (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (9): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): ReLU()
  )
  (block_2): Sequential(
    (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
    (6): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): Dropout(p=0.2, inplace=False)
    (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (9): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): ReLU()
  )
  (block_3): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
    (6): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): Dropout(p=0.2, inplace=False)
    (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (9): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): ReLU()
  )
  (block_4): Sequential(
    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
    (6): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): Dropout(p=0.2, inplace=False)
    (8): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (9): ReLU()
  )
  (classifier): Sequential(
    (0): Linear(in_features=4096, out_features=2048, bias=True)
    (1): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Dropout(p=0.2, inplace=False)
    (4): Linear(in_features=2048, out_features=43, bias=True)
  )
)
ResNet34
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
           Conv2d-10           [-1, 32, 32, 32]             896
      BatchNorm2d-11           [-1, 32, 32, 32]              64
             ReLU-12           [-1, 32, 32, 32]               0
           Conv2d-13           [-1, 32, 32, 32]           9,248
      BatchNorm2d-14           [-1, 32, 32, 32]              64
             ReLU-15           [-1, 32, 32, 32]               0
           Conv2d-16           [-1, 32, 32, 32]           9,248
          Dropout-17           [-1, 32, 32, 32]               0
        MaxPool2d-18           [-1, 32, 16, 16]               0
      BatchNorm2d-19           [-1, 32, 16, 16]              64
             ReLU-20           [-1, 32, 16, 16]               0
           Conv2d-21           [-1, 64, 16, 16]          18,496
      BatchNorm2d-22           [-1, 64, 16, 16]             128
             ReLU-23           [-1, 64, 16, 16]               0
           Conv2d-24           [-1, 64, 16, 16]          36,928
      BatchNorm2d-25           [-1, 64, 16, 16]             128
             ReLU-26           [-1, 64, 16, 16]               0
           Conv2d-27           [-1, 64, 16, 16]          36,928
          Dropout-28           [-1, 64, 16, 16]               0
        MaxPool2d-29             [-1, 64, 8, 8]               0
      BatchNorm2d-30             [-1, 64, 8, 8]             128
             ReLU-31             [-1, 64, 8, 8]               0
           Conv2d-32            [-1, 128, 8, 8]          73,856
      BatchNorm2d-33            [-1, 128, 8, 8]             256
             ReLU-34            [-1, 128, 8, 8]               0
           Conv2d-35            [-1, 128, 8, 8]         147,584
      BatchNorm2d-36            [-1, 128, 8, 8]             256
             ReLU-37            [-1, 128, 8, 8]               0
           Conv2d-38            [-1, 128, 8, 8]         147,584
          Dropout-39            [-1, 128, 8, 8]               0
        MaxPool2d-40            [-1, 128, 4, 4]               0
      BatchNorm2d-41            [-1, 128, 4, 4]             256
             ReLU-42            [-1, 128, 4, 4]               0
           Conv2d-43            [-1, 256, 4, 4]         295,168
      BatchNorm2d-44            [-1, 256, 4, 4]             512
             ReLU-45            [-1, 256, 4, 4]               0
           Conv2d-46            [-1, 256, 4, 4]         590,080
      BatchNorm2d-47            [-1, 256, 4, 4]             512
             ReLU-48            [-1, 256, 4, 4]               0
           Conv2d-49            [-1, 256, 4, 4]         590,080
          Dropout-50            [-1, 256, 4, 4]               0
      BatchNorm2d-51            [-1, 256, 4, 4]             512
             ReLU-52            [-1, 256, 4, 4]               0
           Linear-53                 [-1, 2048]       8,390,656
      BatchNorm1d-54                 [-1, 2048]           4,096
             ReLU-55                 [-1, 2048]               0
          Dropout-56                 [-1, 2048]               0
           Linear-57                   [-1, 43]          88,107
       LogSoftmax-58                   [-1, 43]               0
================================================================
Total params: 10,450,379
Trainable params: 10,450,379
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 4.27
Params size (MB): 39.87
Estimated Total Size (MB): 44.15
----------------------------------------------------------------
Calculating train accuracy and loss for epoch 0
Calculating validation accuracy and loss for epoch 0
Calculating test accuracy
Train: Accuracy = 15.87%, Avg Loss = 3.20
Validation: Accuracy = 16.12%, Avg Loss = 3.22
Test: Accuracy = 17.51%, Avg Loss = 3.20

	Time taken: 00:02:33.31


Process finished with exit code 0


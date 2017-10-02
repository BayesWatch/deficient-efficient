Mobilenets use the latest craze of "depthwise seperable convolutions" (dw-sep) whereby you replace a bog-standard convolution with two. The first of which is depthwise where a filter of the desired kernel size (always 3x3 these days) is learnt for each input channel and applied. The second is a regular convolution with a 1x1 kernel. This uses less parameters.

conv3x3 --> BN --> RELU

is replaced with 

conv3x3DW --> BN --> RELU --> conv1x1 --> BN --> RELU

A CIFAR10 version of mobilenet is provided in https://github.com/kuangliu/pytorch-cifar. It has the following structure:

64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024

where (X,Y) describes each layer. X is the number of filters and Y is the stride (most often 1, and ommited).

Let's do paramter comparison for this structure with dw-sep convs and regular.

Regular:
```conv1.weight has 864 params
bn1.weight has 32 params
bn1.bias has 32 params
bn1.running_mean has 32 params
bn1.running_var has 32 params
layers.0.conv1.weight has 18432 params
layers.0.bn1.weight has 64 params
layers.0.bn1.bias has 64 params
layers.0.bn1.running_mean has 64 params
layers.0.bn1.running_var has 64 params
layers.1.conv1.weight has 73728 params
layers.1.bn1.weight has 128 params
layers.1.bn1.bias has 128 params
layers.1.bn1.running_mean has 128 params
layers.1.bn1.running_var has 128 params
layers.2.conv1.weight has 147456 params
layers.2.bn1.weight has 128 params
layers.2.bn1.bias has 128 params
layers.2.bn1.running_mean has 128 params
layers.2.bn1.running_var has 128 params
layers.3.conv1.weight has 294912 params
layers.3.bn1.weight has 256 params
layers.3.bn1.bias has 256 params
layers.3.bn1.running_mean has 256 params
layers.3.bn1.running_var has 256 params
layers.4.conv1.weight has 589824 params
layers.4.bn1.weight has 256 params
layers.4.bn1.bias has 256 params
layers.4.bn1.running_mean has 256 params
layers.4.bn1.running_var has 256 params
layers.5.conv1.weight has 1179648 params
layers.5.bn1.weight has 512 params
layers.5.bn1.bias has 512 params
layers.5.bn1.running_mean has 512 params
layers.5.bn1.running_var has 512 params
layers.6.conv1.weight has 2359296 params
layers.6.bn1.weight has 512 params
layers.6.bn1.bias has 512 params
layers.6.bn1.running_mean has 512 params
layers.6.bn1.running_var has 512 params
layers.7.conv1.weight has 2359296 params
layers.7.bn1.weight has 512 params
layers.7.bn1.bias has 512 params
layers.7.bn1.running_mean has 512 params
layers.7.bn1.running_var has 512 params
layers.8.conv1.weight has 2359296 params
layers.8.bn1.weight has 512 params
layers.8.bn1.bias has 512 params
layers.8.bn1.running_mean has 512 params
layers.8.bn1.running_var has 512 params
layers.9.conv1.weight has 2359296 params
layers.9.bn1.weight has 512 params
layers.9.bn1.bias has 512 params
layers.9.bn1.running_mean has 512 params
layers.9.bn1.running_var has 512 params
layers.10.conv1.weight has 2359296 params
layers.10.bn1.weight has 512 params
layers.10.bn1.bias has 512 params
layers.10.bn1.running_mean has 512 params
layers.10.bn1.running_var has 512 params
layers.11.conv1.weight has 4718592 params
layers.11.bn1.weight has 1024 params
layers.11.bn1.bias has 1024 params
layers.11.bn1.running_mean has 1024 params
layers.11.bn1.running_var has 1024 params
layers.12.conv1.weight has 9437184 params
layers.12.bn1.weight has 1024 params
layers.12.bn1.bias has 1024 params
layers.12.bn1.running_mean has 1024 params
layers.12.bn1.running_var has 1024 params
linear.weight has 10240 params
linear.bias has 10 params
Net has 28291306 params in total
```

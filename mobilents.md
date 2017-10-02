Mobilenets use the latest craze of "depthwise seperable convolutions" (dw-sep) whereby you replace a bog-standard convolution with two. The first of which is depthwise where a filter of the desired kernel size (always 3x3 these days) is learnt for each input channel and applied. The second is a regular convolution with a 1x1 kernel. This uses less parameters.

conv3x3 --> BN --> RELU

is replaced with 

conv3x3DW --> BN --> RELU --> conv1x1 --> BN --> RELU

A CIFAR10 version of mobilenet is provided in https://github.com/kuangliu/pytorch-cifar. It has the following structure:

64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024

where (X,Y) describes each layer. X is the number of filters and Y is the stride (most often 1, and ommited).

Let's do parameter comparison for this structure with dw-sep convs and regular.

(Note that I have chosen not to print batch norm params as they are insignficant).

dw-sep:
```conv1.weight has 864 params
layers.0.conv1.weight has 288 params
layers.0.conv2.weight has 2048 params
layers.1.conv1.weight has 576 params
layers.1.conv2.weight has 8192 params
layers.2.conv1.weight has 1152 params
layers.2.conv2.weight has 16384 params
layers.3.conv1.weight has 1152 params
layers.3.conv2.weight has 32768 params
layers.4.conv1.weight has 2304 params
layers.4.conv2.weight has 65536 params
layers.5.conv1.weight has 2304 params
layers.5.conv2.weight has 131072 params
layers.6.conv1.weight has 4608 params
layers.6.conv2.weight has 262144 params
layers.7.conv1.weight has 4608 params
layers.7.conv2.weight has 262144 params
layers.8.conv1.weight has 4608 params
layers.8.conv2.weight has 262144 params
layers.9.conv1.weight has 4608 params
layers.9.conv2.weight has 262144 params
layers.10.conv1.weight has 4608 params
layers.10.conv2.weight has 262144 params
layers.11.conv1.weight has 4608 params
layers.11.conv2.weight has 524288 params
layers.12.conv1.weight has 9216 params
layers.12.conv2.weight has 1048576 params
linear.weight has 10240 params
linear.bias has 10 params
Net has 3239114 params in total
```
normal:
```conv1.weight has 864 params
layers.0.conv1.weight has 288 params
layers.0.conv2.weight has 2048 params
layers.1.conv1.weight has 576 params
layers.1.conv2.weight has 8192 params
layers.2.conv1.weight has 1152 params
layers.2.conv2.weight has 16384 params
layers.3.conv1.weight has 1152 params
layers.3.conv2.weight has 32768 params
layers.4.conv1.weight has 2304 params
layers.4.conv2.weight has 65536 params
layers.5.conv1.weight has 2304 params
layers.5.conv2.weight has 131072 params
layers.6.conv1.weight has 4608 params
layers.6.conv2.weight has 262144 params
layers.7.conv1.weight has 4608 params
layers.7.conv2.weight has 262144 params
layers.8.conv1.weight has 4608 params
layers.8.conv2.weight has 262144 params
layers.9.conv1.weight has 4608 params
layers.9.conv2.weight has 262144 params
layers.10.conv1.weight has 4608 params
layers.10.conv2.weight has 262144 params
layers.11.conv1.weight has 4608 params
layers.11.conv2.weight has 524288 params
layers.12.conv1.weight has 9216 params
layers.12.conv2.weight has 1048576 params
linear.weight has 10240 params
linear.bias has 10 params
Net has 3239114 params in total
```


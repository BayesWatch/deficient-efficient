Mobilenets use the latest craze of "depthwise seperable convolutions" whereby you replace a bog-standard convolution with two. The first of which is depthwise where a filter of the desired kernel size (always 3x3 these days) is learnt for each input channel and applied. The second is a regular convolution with a 1x1 kernel. This uses less parameters.

conv3x3 --> BN --> RELU

is replaced with 

conv3x3DW --> BN --> RELU --> conv1x1 --> BN --> RELU

A CIFAR10 version of mobilenet is provided in https://github.com/kuangliu/pytorch-cifar. It has the following structure:

64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024

where (X,Y) describes each layer. X is the number of filters and Y is the stride (most often 1, and ommited).


Mobilenets use the latest craze of "depthwise seperable convolutions" (dw-sep) whereby you replace a bog-standard convolution with two. The first of which is depthwise where a filter of the desired kernel size (always 3x3 these days) is learnt for each input channel and applied. The second is a regular convolution with a 1x1 kernel. This uses less parameters.

conv3x3 --> BN --> RELU

is replaced with 

conv3x3DW --> BN --> RELU --> conv1x1 --> BN --> RELU

A CIFAR10 version of mobilenet is provided in https://github.com/kuangliu/pytorch-cifar. It has the following structure:

64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024

where (X,Y) describes each layer. X is the number of filters and Y is the stride (most often 1, and ommited).

Let's do parameter comparison for this structure with dw-sep convs and regular.

(Note that I have chosen not to print batch norm params as they are insignificant).

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
layers.0.conv1.weight has 18432 params
layers.1.conv1.weight has 73728 params
layers.2.conv1.weight has 147456 params
layers.3.conv1.weight has 294912 params
layers.4.conv1.weight has 589824 params
layers.5.conv1.weight has 1179648 params
layers.6.conv1.weight has 2359296 params
layers.7.conv1.weight has 2359296 params
layers.8.conv1.weight has 2359296 params
layers.9.conv1.weight has 2359296 params
layers.10.conv1.weight has 2359296 params
layers.11.conv1.weight has 4718592 params
layers.12.conv1.weight has 9437184 params
linear.weight has 10240 params
linear.bias has 10 params
Net has 28291306 params in total
```
So the normal one has 8.7x more parameters. Ouch.

Let's do a quick performance comparison on CIFAR 10. We (dumbly) train both in the same manner. 200 epochs, a learning rate of 0.1 that is multiplied by 0.2 at epochs 60,120, and 160. 

Normal: 93.84%
dw-sep: 90.45%

An aside: in https://arxiv.org/abs/1610.02357 Chollet finds things improve when we remove the intermediate RELU in the depthwise sep module

conv3x3DW --> BN --> ***DEBATABLE RELU*** --> conv1x1 --> BN --> RELU

When we do this we actually get a lower acc of 89.86%, so this seems inconclusive.

I'm going to now call the two nets C and D (conv and depthwise-sep) because I'm tired of typing depthwise-sep.

Let's evaluate: C when it has D no. params
            and D when it has C no. params
            
We just introduce a multiplier w that is applied to each channel size

64*w, (128*w,2), etc.

For w =3 (massive channels!) C has 28519242 params (99% the number of D)

for w=.338 D has 3239114 params (100.2% the number of C)

The reduced D model actually does better than the depthwise seperable model, with an acc of 91.71%

Table time:

| Model      | Conv Type | No. Params (M) | Acc.     | 
|------------|-----------|----------------|----------|
| Mobilenet  | D         |   3.24         | 90.45%   |
| Mobilenet  | C         |  28.23         | 93.84%   |
| Mobilenet  | D         |  28.52         | 91.07%   |
| Mobilenet  | C         |   3.24         | 91.39%   |

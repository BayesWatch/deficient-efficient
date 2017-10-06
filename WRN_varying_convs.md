Mobilenets are stupid beyond their use of separable convolutions.

As such, I'm going to perform some similar experiments but on a better (smaller!) network.

Wideresnets (which are deceptively often thinner than normal resnets) look like:

| group | output size | filters in block  | no. blocks |
|-------|-------------|-------------------|------------| 
| conv1 | 32x32       | 16                |  1         |
| conv2 | 32x32       |[16k 16k]          |  N         |
| conv3 | 16x16       |[32k 32k]          |  N         |
| conv4 | 8x8         |[64k 64k]          |  N         |

so for a net denoted as WRN-depth-width

k = width

and 

N = (depth - 4) / 6

Let's train some networks for different types of convolutions.

| Model         | Conv Type | No. Params (M) | Acc.     | KD w(a) | AT w(a) |
|---------------|-----------|----------------|----------|---------|---------|
|(a) WRN-40-2   | 3x3       | 2248954        | 94.94%   | 95.32%  | 95.00%  |
|(b) WRN-40-2   | 2x2_d2    | 1012474        | 93.42%   | 93.94%  | 94.85%  |
|(c) WRN-40-2   | 3x3DW+1x1 | 304074         | 91.49%   | 92.00%  | 93.52%  |
|---------------|-----------|----------------|----------| KD w(d) | AT w(d) |
|(d) WRN-40-1   | 3x3       | 566650         | 93.52%   |         |         |
|(e) WRN-40-1   | 2x2_d2    | 256890         | 91.28%   |         | 92.78%  |
|(f) WRN-40-1   | 3x3DW+1x1 | 87882          | 89.25%   |         |         |
|---------------|-----------|----------------|----------| KD w(g) | AT w(g) |
|(g) WRN-16-2   | 3x3       | 693498         | 93.47%   |         |         |
|(h) WRN-16-2   | 2x2_d2    | 317178         | 92.29%   |         | 93.29%  |
|(i) WRN-16-2   | 3x3DW+1x1 | 101578         | 90.61%   |         |         |
|---------------|-----------|----------------|----------| KD w(j) | AT w(j) |
|(j) WRN-16-1   | 3x3       | 175994         | 91.19%   |         |         |
|(k) WRN-16-1   | 2x2_d2    | 81274          | 88.51%   |         | 88.70%  |
|(l) WRN-16-1   | 3x3DW+1x1 | 29642          | 86.78%   |         |         |



-3x3 is vanilla
-2x2_d2 refers to a 2x2 kernel with dilation 2 (i.e. a 3x3 kernel with only the corners non-zero)
-3x3DW+1x1 refers to the separable convolution used in mobilenet.

Now let's experiment with knowledge distillation and attention transfer where the teacher is (a), and the student is (a),(b) or (c) (See extra columns above).

Note that doing knowledge distillation "with itself" was put in for curiosity.

Observations:
- A model gets better if it is taught by literally the same architecture. This is effectively an ensemble.
- Attention transfer lets us get almost identical results using 2x2 kernels, so a drop in parameters of over a half for nothing!
- It is pretty good for depthwise convolutions as well.

--------------------

What we want to demonstrate is when we use attention transfer to reduce kernel size it is superior to reducing depth/width given a parameter budget.
I will redo and extend table 1 of https://arxiv.org/abs/1612.03928:


Sanity check table:


| Experiment | Student       | Teacher        | student | KD     | AT     | teacher  |
|------------|---------------|----------------|---------|--------|--------|----------|
| Original   | WRN-16-1 0.2M | WRN-16-2 0.7M  | 91.23   | 92.59  | 92.07  | 93.69    | 
| Ours       | WRN-16-1 0.2M | WRN-16-2 0.7M  | 91.19   | 92.21  | 92.28  | 93.47    | 
|            |               |                |         |        |        |          |     
| Original   | WRN-16-1 0.2M | WRN-40-1 0.6M  | 91.23   | 91.61  | 91.75  | 93.42    | 
| Ours       | WRN-16-1 0.2M | WRN-40-1 0.6M  | 91.19   | 91.90  | 91.81  | 93.52    | 
|            |               |                |         |        |        |          |     
| Original   | WRN-16-2 0.7M | WRN-40-2 2.2M  | 93.69   | 93.92  | 94.15  | 94.77    |
| Ours       | WRN-16-2 0.7M | WRN-40-2 2.2M  | 93.47   |        |        | 94.94    |





| Student          | Teacher         | student | KD     | AT     | teacher  |
|------------------|-----------------|---------|--------|--------|----------|
|WRN-40-2_sep 0.3M | WRN-40-2 2.2M   | 91.49   | 92.00  | 93.52  | 94.94    |
|WRN-40-2_2x2 1M   | WRN-40-2 2.2M   | 93.42   | 93.94  | 94.85  | 94.94    |
---------------------------------------------------------------------------- 
|WRN-40-1_sep 0.3M | WRN-40-2 2.2M   | 
|WRN-40-1_2x2 0.3M | WRN-40-2 2.2M   | 

    

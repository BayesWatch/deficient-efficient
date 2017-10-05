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
|(a) WRN-40-2   | 3x3       | 2248954        | 94.94%   | 
|(b) WRN-40-2   | 2x2_d2    | 1012474        | 93.42%   | 93.94%  | 94.69%  |
|(c) WRN-40-2   | 3x3DW+1x1 | 304074         | 91.49%   | 92.00%  | 93.52%  |

-3x3 is vanilla
-2x2_d2 refers to a 2x2 kernel with dilation 2 (i.e. a 3x3 kernel with only the corners non-zero)
-3x3DW+1x1 refers to the separable convolution used in mobilenet.

Now let's experiment with knowledge distillation and attention tranfer where the teacher is (a), and the student is (a),(b) or (c) (See extra columns above).

Note that doing knowledge distillation "with itself" was put in for curiosity.


    

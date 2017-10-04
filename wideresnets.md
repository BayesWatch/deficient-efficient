Mobilenets are stupid, and only seperable convolutions should be taken away from them.

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

| Model         | Conv Type | No. Params (M) | Acc.     | 
|---------------|-----------|----------------|----------|
|(a) WRN-40-2   | 3x3       | 2248954        | 94.94%   | 
|(b) WRN-40-2   | 2x2_d2    | 1012474        | 93.42%   | 
|(c) WRN-40-2   | 3x3DW+1x1 | 304074         | 91.49%   | 

When we use knowledge distillation with the teacher as (a), and (c) learnt from scratch we get an increased acc of 92.0%
    
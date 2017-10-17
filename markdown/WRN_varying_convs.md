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

| Model         | Conv Type | No. Params (M) | Acc.   | KD      | AT      | AT+KD   | AT 6   | AT + KD 6 |
|---------------|-----------|----------------|--------|---------|---------|---------|--------|-----------|
|(a) WRN-40-2   | 3x3       | 2248954        | 94.94% |**95.32%**  | 95.08%  | 95.00%  | 94.76% | 87.53%    |
|(b) WRN-40-2   | 2x2_d2    | 1012474        | 93.42% | 93.94%  | 94.64%  | **94.85%**  | 94.65% | 94.52%    |
|(c) WRN-40-2   | 3x3DW+1x1 | 304074         | 91.49% | 92.00%  | 93.33%  | 93.52%  | 93.61% | **93.84%**    |
|(d) WRN-40-1   | 3x3       | 566650         | 93.52% | 93.35%  | 93.46%  | **93.78%**  | 93.17% | 84.77%    |
|(e) WRN-40-1   | 2x2_d2    | 256890         | 91.28% | 92.12%  | **92.33%**  | 92.78%  | 92.20% | 92.30%    |
|(f) WRN-40-1   | 3x3DW+1x1 | 87882          | 89.25% | 90.12%  | 90.71%  | 90.86%  | 91.50% | **91.51%**    |
|(g) WRN-16-2   | 3x3       | 693498         | 93.47% | **94.02%**  | 93.36%  | 93.63%  | 93.59% | 89.61%    |
|(h) WRN-16-2   | 2x2_d2    | 317178         | 92.29% | **93.29%**  | 92.64%  | **93.29%**  | 92.15% | 92.61%    |
|(i) WRN-16-2   | 3x3DW+1x1 | 101578         | 90.61% | **92.55%**  | 91.80%  | 92.42%  | 91.99% | 91.99%    |
|(j) WRN-16-1   | 3x3       | 175994         | 91.19% | **92.08%**  | 90.96%  | 91.21%  | 90.85% | 88.30%    |
|(k) WRN-16-1   | 2x2_d2    | 81274          | 88.51% | **90.09%**  | 88.98%  | 88.70%  | 88.50% | 88.87%    |
|(l) WRN-16-1   | 3x3DW+1x1 | 29642          | 86.78% | **88.94%**  | 88.40%  | 88.65%  | 87.93% | 87.94%    |



-3x3 is vanilla
-2x2_d2 refers to a 2x2 kernel with dilation 2 (i.e. a 3x3 kernel with only the corners non-zero)
-3x3DW+1x1 refers to the separable convolution used in mobilenet.

Now let's experiment with knowledge distillation and attention transfer where the teacher is (a), and the student is (a),(b) or (c) (See extra columns above).

Note that doing knowledge distillation "with itself" was put in for curiosity.

Observations:
- A model gets better if it is taught by literally the same architecture. This is effectively an ensemble. *EDIT: probably not an ensemble*
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
| Ours       | WRN-16-2 0.7M | WRN-40-2 2.2M  | 93.47   | 94.25  | 94.37  | 94.94    |


Some more experiments where everything is taught by the best model (WRN-40-2-3x3)

| Student | type| KD    | AD    | AD6   | ADKD  | ADKD6 |
|---------|-----|-------|-------|-------|-------|-------|
|WRN_40_1 | 3x3 | 93.43 | 94.27 | **94.56** | 93.87 | 93.98 |
|WRN_40_1 | 2x2 | 91.74 | 92.20 | 92.51 | 92.23 | **92.67** |
|WRN_40_1 | sep | 89.76 | 91.26 | 91.40 | 90.90 | **91.89** |
|WRN_16_2 | 3x3 | 94.14 | 94.03 | **94.36** | 94.12 | **94.36** |
|WRN_16_2 | 2x2 | 92.47 | 92.87 | 92.96 | 92.33 | **92.97** |
|WRN_16_2 | sep | 90.57 | 92.00 | **92.43** | 91.25 | 91.40 |
|WRN_16_1 | 3x3 | 91.43 | 91.74 | **91.95** | 91.39 | 91.57 |
|WRN_16_1 | 2x2 | 88.84 | 88.70 | 88.72 | 88.71 | **88.95** |
|WRN_16_1 | sep | 87.20 | 87.80 | 87.65 | 87.69 | **87.92** |

| Student | type| KD    | AD    | AD6   | ADKD  | ADKD6 | AD(ii) | AD6 (ii) | AD9  | AD18  |
|---------|-----|-------|-------|-------|-------|-------|--------|----------|------|-------|
|WRN_40_1 | sep | 89.76 | 91.26 | 91.40 | 90.90 | 91.89 | 91.37  | 91.16    | 91.18| 91.34 |




Reminder: wideresnets look like this:


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

Let's see what happens when I mix and match different convolutional filters in different groups for WRN-16-2

| conv2 | conv3 | conv4 | No. Params | Acc.  |
|-------|-------|-------|------------|-------|
| 3x3   | 3x3   | 3x3   | 693498     | 93.87 |
| 3x3   | 3x3   | 2x2   | 406778     | 93.52 |
| 3x3   | 3x3   | sep   | 240570     | 92.20 |
| 3x3   | 2x2   | 3x3   | 621818     | 93.27 |
| 3x3   | 2x2   | 2x2   | 335098     | 92.47 |
| 3x3   | 2x2   | sep   | 168890     | 92.12 |
| 3x3   | sep   | 3x3   | 581722     | 93.06 |
| 3x3   | sep   | 2x2   | 295002     | 92.03 |
| 3x3   | sep   | sep   | 128794     | 91.19 |
| 2x2   | 3x3   | 3x3   | 675578     | 93.54 |
| 2x2   | 3x3   | 2x2   | 388858     | 93.09 |
| 2x2   | 3x3   | sep   | 222650     | 92.03 |
| 2x2   | 2x2   | 3x3   | 603898     | 92.78 |
| 2x2   | 2x2   | 2x2   | 317178     | 92.15 |
| 2x2   | 2x2   | sep   | 150970     | 91.24 |
| 2x2   | sep   | 3x3   | 563802     | 92.77 |
| 2x2   | sep   | 2x2   | 277082     | 91.75 |
| 2x2   | sep   | sep   | 110874     | 90.30 |
| sep   | 3x3   | 3x3   | 666282     | 93.93 |
| sep   | 3x3   | 2x2   | 379562     | 93.36 |
| sep   | 3x3   | sep   | 213354     | 91.90 |
| sep   | 2x2   | 3x3   | 594602     | 93.12 |
| sep   | 2x2   | 2x2   | 307882     | 91.93 |
| sep   | 2x2   | sep   | 141674     | 91.30 |
| sep   | sep   | 3x3   | 554506     | 92.92 |
| sep   | sep   | 2x2   | 267786     | 91.99 |
| sep   | sep   | sep   | 101578     | 90.37 |

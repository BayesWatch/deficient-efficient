We should Moonshine an Imagenet model and show it performs well with less parameters than a typical one.

For reference, if we take vanilla Resnet pre-trained models from Pytorch (irritatingly, not pre-activation) the performance looks like:

| Model    |  Top1   |  Top5   |
|----------|---------|---------|
| Resnet34 | 26.73   |  8.57   |    
| Resnet18 | 30.36   | 11.02   |     

Now let's train a Resnet18 using attention transfer from a Resnet34 to compare with the same experiment in the AT paper.

A Resnet contains 4 "blocks" in the same way a WRN has 3. The output of these being used for the AT losses.

In the paper, the authors claim to use the last 2 of the 4 block, and don't add the losses until epochs 60-100 (for 100 epochs of training). Unhelpfully, no hyperparameters are given and they say:

```we plan to update the paper with losses on all 4 groups of residual blocks```

Not sure why they did this, as it's easy enough to do it with all four.

In the code for the AT paper there is a script `https://github.com/szagoruyko/attention-transfer/blob/master/imagenet.py` that gives some indication of hyperparameters. Annoyingly, beta is not given.

I simply ran the experiment with the following:

- Same training hyperparameters as in that script (LR 0.1, drops to {.01,.001,.0001} at epochs {30,60,90}, weight-decay 1e-4, batch-size 256)
- all 4 block outputs used for AT, with beta as 3*(1000/4)
- AT used from epoch 1 rather than sticking it in later

Let's now compare the results to the paper:

| Model       | Top1 Elliot |  Top1 Original  |  Top5 Elliot |  Top5 Original  |
|-------------|-------------|-----------------|--------------|-----------------|
| Resnet34    | 26.73       |  26.1           | 8.57         | 8.3             |    
| Resnet18    | 30.36       |  30.4           | 11.02        | 10.8            |
| Resnet18 AT | 29.18       |  29.3           | 10.05        | 10.0            |

Note that, in the above the top1 and top5 Elliot for the pre-trained models (Resnet34 and 18) are just what I got when I downloaded the ones in the torchvision repo. The pre-trained models used in the AT paper appear to be better than that.

Despite this, the top1 and top5 Elliot for the AT paper (which i did train using the pre-trained Resnet34 as teacher, and a new student Resnet18 from scratch) is comparable to their paper.

Good enough!

--------------------------------------------------------------

Next experiment. Let's see what happens when we do the same experiment but with the student as a Resnet34 with depthwise separable convolutions.


| Model             | No. Params |  Top1   |  Top5   |
|-------------------|------------|---------|---------|
| Resnet34          | 21.8M      | 26.73   |  8.57   |    
| Resnet18 AT       | 11.7M      | 29.18   | 11.02   |
| Resnet34 Sep + AT | 3.16M      | 30.16   | 10.66   |
| Resnet34 Sep      | 3.16M      | 32.98   | 12.26   |
| Resnet18(w.5) + AT| 3.20M      | 37.20   |  15.02  |
| Resnet18(w.5)     |

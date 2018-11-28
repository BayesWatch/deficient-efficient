
21st November 2018
==================

This research follows on from research done on ACDC transforms and training
networks incorporated them into the convolutional layers, with the goal of
efficiency. That research log is found [here][acdc]

In this repository we're going to be comparing various other methods using
low-rank approximations to the full-rank random matrices typically used in
deep learning. There were some simple tricks that made the training of ACDC
convolutional layers possible:

1. Set an appropriate weight decay.
2. Use distillation.
3. Ensure initialisation maintains the properties of common deep learning
random initialisation strategies.

Currently, we have only added HashedNets to the implemented layers, and
they need to use separable layers (current implementation does not). The
implementation of ACDC layers is separable, so it is required for a level
playing field. Also, it's a standard choice to improve efficiency anyway.

After implementing the current full convolution HashedNet layers, tested
them with a WRN-28-10 and a teacher that achieves 3.3% top 1 error on
CIFAR-10. Strangely, the network trained without a teacher worked better,
achieving a final top-1 error of 6.48%. The student network converged to
7.47%.

The training loss converges to zero in the case of the network trained
without a teacher. Student networks typically don't converge to zero; the
attention transfer loss is hard to minimise.

![](images/train_loss_Nov21.png)

The overfitting here is not bad, similar to experiments with traditional
networks:

![](images/val_loss_Nov21.png)

The original network had the following number of parameters and mult-adds:

```
Mult-Adds: 5.24564E+09
Params: 3.64792E+07
```

And we are able to reduce that to be about a 10th, because these HashedNet
layers have been set to use 1/10 the parameters of the original layer.

```
Mult-Adds: 5.24564E+09
Params: 1.49884E+06
```

I'm not sure why the number of parameters even smaller than a 1/10. That
suggests there might be a problem in the script counting parameters.

To check, added a sanity check explicitly checking the number of parameters
in the whole network. Result was the same for both networks, so it could be
that the HashedNet layer's method of choosing how many parameters to budget
for is broken somehow.

Update: found the mistake. Number of original parameters was being
estimated using only one of the kernel dimensions, meaning it was out by,
on average, a factor of three. Fixed it and the network using
HashedDecimate is indeed about 10 times smaller:

```
Mult-Adds: 5.24564E+09
Params: 3.91190E+06
Sanity check, parameters: 3.91190E+06
```

Don't know what performance this network might get, though. Presumably a
little better.

Started an experiment with a SeparableHashedNet, including this fix. So
far, it is performing better, but that could easily be because of the 3x
larger parameter budget in each layer.

[acdc]: https://github.com/gngdb/pytorch-acdc/blob/master/research-log.md

Budget Parameterisation
-----------------------

I wanted to, regardless of the low-rank approximation method chosen, be
able to set a budget in parameters for a network then the code would just
set the hyperparameter controlling how many parameters are used by the
network to meet that. To do that, I need to be able to pass options to the
module implementing the `Conv` that is substituted into the WRN used in
this code.

Currently, the argument given on the command line is just crudely matched
to the name of a `Conv` in `blocks.py`. There's even a horrible `if` block
involved.

Seems like the easiest way to do this is going to be to have a generator
function in `blocks.py` that returns a `Conv` with a hyperparameter set to
whatever we like. Then, it's easy enough to search through a range of
settings when a budget is prescribed.

Wrote it to used scipy's scalar_minimize to set the hyperparameter to match
the budget specified.

DARTS
-----

Would like to run experiments on a state of the art network. DARTS is,
conventiently, one of the best CIFAR-10 networks published, and they
provided a stored model in PyTorch.

22nd November 2018
==================

### DARTS results

Trained a DARTS network with our default training settings overnight. The
final top 1 error was only 4.96%. The accuracy of the pre-trained network
supplied with the paper is below 3%, so their training strategy seems to
also be important. If we're going to use this network in these experiments,
we'll have to port their training script into ours and make sure everything
is the same. This could involve also using the extra augmentations they
use, which could slow down training.

### ACDC results

Also, overnight I ran another experiment with a WRN-28-10 and the original
ACDC parameterisation using a full permutation instead of a riffle shuffle.
The final top 1 error was 5.35%, which is slightly worse than the 5.23%
achieved with the riffle shuffle. Although, they are so similar it seems
likely that how the shuffle is done isn't very important.

### HashedNet results

Trained separable HashedNet WRN-28-10 with and without distillation using
the same teacher used in other experiments. Before, for reference, the
network has `3.64792E+07` parameters, and after it has `3.91190E+06`
parameters, so approximately 10x fewer. Trained without distillation, the
top-1 error is 5.95%, with distillation it decreases to 3.84%: similar to
the results obtained with grouping and bottlenecks described in the
[pytorch-acdc research log][acdc].

### Matching DARTS training protocol

Looking at the code provided for training DARTS models, and trying to make
sure we do exactly the same thing when we train one. Hyperparameters:

* `batch_size = 96`
* `learning_rate = 0.025`
* `momentum = 0.9`
* `weight_decay = 3e-4`
* `epochs = 600` !!!

Data tranforms:

* port `_data_transforms_cifar10` functions from `utils.py` and use it
* will also need to port `Cutout` (succint implementation of Cutout), also
in `utils.py`

Training specifics:

* Cosine annealing learning rate schedule, annealing over the entire
training schedule.
* Linearly schedule `drop_path_prob` in the model object from 0 to 0.2
over the entire training schedule.

Training will probably take about 24 hours, with these changes.

Did some hacky `if` statements to make these changes when running with a
`DARTS` network. Should work.

Started an experiment training the DARTS network with the proposed
settings. Unfortunately, my estimate of 24 hours looks to be optimistic.
The estimate given by the code itself is currently 42 hours. I think the
training code provided in the DARTS repo was a little faster than this.

Started parallel experiment running the original training code from scratch
on a separate GPU. Should be able to compare the learning curves later, if
required.

### Matching Teacher/Student WRN-28-10 Training

Currently, the experiments reported here with WRN-28-10 used a WRN-28-10
that had been trained using Cutout augmentation, but the student hadn't
used this augmentation. I thought it better to use a teacher network that
wasn't trained with Cutout augmentation in order that they match, and then
the results will match the performance of WRN-28-10 reported in the
literature.

I got a pretrained WRN-28-10 from someone else in our group, who tested it
and found it's test error was 3.95%, but after loading it into this code I
tested it at 4.2%. I'm not sure what the source of error might be.

23rd November 2018
==================

Seemed reasonable to use the best reported WRN architecture in our ImageNet
experiments, as that is a reasonable benchmark. Unfortunately, it turns out
the WRN-50-2 reported in the paper, and provided with an esoteric
functional implementation [here][func] is slightly different from the
models trained on CIFAR-10. It doesn't match the figures in the paper on
the structure of the network.

Maybe it's mentioned somewhere in the paper in passing but I didn't see it.
It turns out it's a ResNet-50, but with `expansion=2` and all channels
doubled.

So, I adapted the official ResNet50 implementation from torchvision and
loaded the supplied parameters: implemeneted in the script
`load_wrn_50_2.py`. Luckily, the ordering of parameters matched without too
much difficulty.

Testing this over the validation set:

```
Error@1 22.530 Error@5 6.406
```

Which is unfortunately 0.5% short of the expected top-1/top-5 of 22.0/6.05.
Not sure why that might be, if there had been a problem loading parameters
(if something hadn't matched properly) I would've expected it to fail
completely.

To double check, updated `load_wrn_50_2.py` to run a random input through
both networks and check the output is always the same. The max absolute
error never gets about 1e-3, so they're doing the same thing. The
difference in error may just be because this is an old implementation and
some small thing may have changed in the PyTorch code. The only way to know
for sure would be to run the original validation script and see if the
results still hold.

So, I did that, and the results matched the results got from my own
experiment on the validation set (top-1/top-5): `[77.47, 93.594]`. I can't
explain that 0.5%. Committing the version of the script I ran,
[here](https://gist.github.com/gngdb/c5855e10dea83c99a44b338acc76759f).

[func]: https://github.com/szagoruyko/functional-zoo/blob/master/wide-resnet-50-2-export.ipynb

### Testing ImageNet Training

Before we train a student network, we need to know that our training
routine works for this WRN-50-2 network. Looking at the original paper,
they report the learning rate, weight decay and momentum match what we
already set to do CIFAR-10 training with these WideResNets. Unfortunately,
they don't give more details on the ImageNet training, other than saying
they use `fb.resnet.torch`. That gives no clear single prescription for a
ResNet-50, beyond setting the minibatch size to 256 and using 4 GPUs in
parallel.

As I've trained ImageNet models in the past using PyTorch, I'm just going
to use those settings. Matching the [PyTorch ImageNet
example][imagenetexample]:

* 90 epochs
* learning rate decays every 30 epochs to 1/10 of prior
* batch size 256 (unlikely to fit on 1 GPU)

Only have 1 GPU free right now. Was not able to start an experiment with
batch size 256, or 64. Had to set it to 32. Unfortunately, by my estimate
it will take two weeks and may not even converge properly with the wrong
batch size. Hopefully, this is only because we're not using multi-gpu
training and not a problem with our training script.

**update**: killed this experiment, after a few days it did not appear to
be converging.

[imagenetexample]: https://github.com/pytorch/examples/blob/master/imagenet/main.py

26th November 2018
==================

Talking to Amos, seems like using cutout in all experiments is probably a
safer course of action. Don't want to arbitrarily limit the results.

Started experiment to test chosen AT taps, and distillation in general,
when using a DARTS network. Used Separabled-Hashed-Decimate substitution.

28th November 2018
==================

Tested running student WRN-28-10 with Cutout enabled, as the teacher was
also trained with Cutout. Used the Separable-Hashed-Decimate substitute
convolution layer, as we've already done an experiment with this same
network so we can compare. The results (found above) were previously 3.84%
top-1 error at the end of training. With Cutout, it actually performed
worse, with a top-1 error of 3.99%.


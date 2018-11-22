
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

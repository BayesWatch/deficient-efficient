
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


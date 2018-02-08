# moonshine

Code used to produce https://arxiv.org/abs/1711.02613

## Installation Instructions

Use Conda! Make a new environment then activate it.
```
conda create -n torch python=3
source activate torch
```
then

```
conda install pytorch torchvision cuda90 -c pytorch
pip install tqdm
pip install tensorboardX
pip install tensorflow
```

## Training a Teacher

In general, the following code trains a teacher network:

```
python main.py <DATASET> teacher --conv <CONV-TYPE> -t <TEACHER_CHECKPOINT> --wrn_depth <TEACHER_DEPTH> --wrn_width <TEACHER_WIDTH>
```

In the paper, results are typically reported using a standard 40-2 WRN,
which would be the following (on cifar-10):

```
python main.py cifar10 teacher --conv Conv -t wrn_40_2.ckpt --wrn_depth 40 --wrn_width 2
```

## Training a Student

To train a student using KD:

```
python main.py <DATASET> KD --conv <CONV-TYPE> -t <EXISTING TEACHER CHECKPOINT> -s <STUDENT CHECKPOINT> --wrn_depth <STUDENT_DEPTH> --wrn_width <STUDENT_WIDTH>
```
  
To train a student using AT:

```
python main.py <DATASET> AT --conv <CONV-TYPE> -t <EXISTING TEACHER CHECKPOINT> -s <STUDENT CHECKPOINT> --wrn_depth <STUDENT_DEPTH> --wrn_width <STUDENT_WIDTH>
```
  
Note: the AT method uses KD by default, so to turn it off, set alpha to 0

As an example, this would train a model with the same structure as the
teacher network, but using a bottleneck grouped + pointwise convolution as
a substitute for the full convolutions in the full network:

```
python main.py cifar10 AT --conv G8B2 -t wrn_40_2.ckpt -s wrn_40_2.g8b2.student.ckpt --wrn_depth 40 --wrn_width 2
```

## Acknowledgements

Code has been liberally borrowed from other repos.

A non-exhaustive list follows:

```
https://github.com/szagoruyko/attention-transfer
https://github.com/kuangliu/pytorch-cifar
https://github.com/xternalz/WideResNet-pytorch
```

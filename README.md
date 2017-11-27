# teach

Student-teacher toolbox for pytorch.

## Install

I installed requirements as follows by creating a conda environment with miniconda2. Make sure your bashrc points towards cudnn and CUDA

e.g.
```
export CUDA_HOME=/opt/cuda-8.0.44
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib64/:$CUDA_HOME/lib64:/disk/scratch/ecrowley/cudnn_v7/lib64/"
export CUDNN_INCLUDE_DIR="/disk/scratch/ecrowley/cudnn_v7/include/"
export CUDNN_LIB_DIR="/disk/scratch/ecrowley/cudnn_v7/lib64/"
```
Some of the above is likely redundant.

- conda create -n torch3 python=3
- source activate torch3
- export CMAKE_PREFIX_PATH="/disk/scratch/ecrowley/miniconda2/envs/torch3"
- conda install numpy pyyaml mkl setuptools cmake cffi
- conda install -c soumith magma-cuda80

Then go to some directory:
- git clone --recursive https://github.com/pytorch/pytorch
- cd pytorch
- python setup.py install

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

# Custom Blocks

Say you've come up with some alternative convolution or block structure,
and you want to plug it into this code and see how well it performs when
trained with attention transfer using a good teacher model. To do that, all
you have to do is write a python file with `nn.Module` child objects
named `Conv` and (optionally) `Block`. If `Block` is not defined, we will
default to whatever the `blocktype` option is.

To use this, you no longer need to specify `--conv`, but can just specify
the name of this module file:

```
python main.py <DATASET> AT --module <YOUR-FILE.py> -t <EXISTING TEACHER CHECKPOINT> -s <STUDENT CHECKPOINT> --wrn_depth <STUDENT_DEPTH> --wrn_width <STUDENT_WIDTH>
```

To see what interface your modules must present, look at `dummy_module.py`
for an example.

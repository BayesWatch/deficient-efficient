# teach

**Edit: now using the pytorch implementation for grouped convolutions as it was fixed,  install pytorch from source**

Student-teacher toolbox for pytorch.

## Install

I installed requirements as follows by creating a conda environment with miniconda2:

- conda create -n torch python=2
- source activate torch
- conda install pytorch torchvision cuda80 -c soumith
- pip install tqdm
- pip install git+https://github.com/szagoruyko/pyinn.git@master

Pyinn uses cupy which annoyingly writes to the home directory by default (which on AFS leads to errors). I found setting the cache dir using the recommended environmental variable didn't work.

A crude workaround is instead to modify `<CONDA_PATH>/envs/torch/lib/python2.7/site-packages/cupy/cuda/compiler.py` and change the line (103) to `_default_cache_dir = <SOMEWHERE ON SCRATCH>`

## Training a Teacher

In general, the following code trains a teacher network:

```
python main.py <DATASET> teacher <CONV-TYPE> -t <TEACHER_CHECKPOINT> --wrn_depth <TEACHER_DEPTH> --wrn_width <TEACHER_WIDTH>
```

In the paper, results are typically reported using a standard 40-2 WRN,
which would be the following (on cifar-10):

```
python main.py cifar10 teacher Conv -t wrn_40_2.ckpt --wrn_depth 40 --wrn_width 2
```

## Training a Student

To train a student using KD:

```
python main.py <DATASET> KD <CONV-TYPE> -t <EXISTING TEACHER CHECKPOINT> -s <STUDENT CHECKPOINT> --wrn_depth <STUDENT_DEPTH> --wrn_width <STUDENT_WIDTH>
```
  
To train a student using AT:

```
python main.py <DATASET> AT <CONV-TYPE> -t <EXISTING TEACHER CHECKPOINT> -s <STUDENT CHECKPOINT> --wrn_depth <STUDENT_DEPTH> --wrn_width <STUDENT_WIDTH>
```
  
Note: the AT method uses KD by default, so to turn it off, set alpha to 0

As an example, this would train a model with the same structure as the
teacher network, but using a bottleneck grouped + pointwise convolution as
a substitute for the full convolutions in the full network:

```
python main.py cifar10 AT G8B2 -t wrn_40_2.ckpt -s wrn_40_2.g8b2.student.ckpt --wrn_depth 40 --wrn_width 2
```


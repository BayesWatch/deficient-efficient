# teach

Student-teacher toolbox for pytorch.

I installed requirements as follows by creating a conda environment with miniconda2:

- conda create -n torch python=2
- source activate torch
- conda install pytorch torchvision cuda80 -c soumith
- pip install tqdm
- pip install git+https://github.com/szagoruyko/pyinn.git@master

Pyinn uses cupy which annoyingly writes to the home directory by default (which on AFS leads to errors). I found setting the cache dir using the recommended environmental variable didn't work.

A crude workaround is instead to modify `<CONDA_PATH>/envs/torch/lib/python2.7/site-packages/cupy/cuda/compiler.py` and change the line (103) to `_default_cache_dir = <SOMEWHERE ON SCRATCH>`

To train a teacher:

python main.py teacher <CONV-TYPE> -t <TEACHER_CHECKPOINT> --wrn_depth <TEACHER_DEPTH> --wrn_width <TEACHER_WIDTH>

To train a student using KD:

python main.py KD <CONV-TYPE> -t <EXISTING TEACHER CHECKPOINT> -s <STUDENT CHECKPOINT> --wrn_depth <STUDENT_DEPTH> --wrn_width <STUDENT_WIDTH>
  
To train a student using AT:

python main.py AT <CONV-TYPE> -t <EXISTING TEACHER CHECKPOINT> -s <STUDENT CHECKPOINT> --wrn_depth <STUDENT_DEPTH> --wrn_width <STUDENT_WIDTH>
  
the AT method uses KD by default, so to turn it off, set alpha to 0
  
    

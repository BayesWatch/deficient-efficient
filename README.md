# teach

Student-teacher toolbox for pytorch.

To use this you will have to install tqdm (e.g. pip install tqdm), torch, and pyinn (pip install git+https://github.com/szagoruyko/pyinn.git@master), then make a directory called "checkpoints"

train_teacher.py trains a teacher.

`python teacher_train.py <MODEL_TYPE> -t <PLACE TO SAVE TEACHER> --GPU  <WHICH GPU TO USE> --wrn_depth <Depth of your WRN> --wrn_width <Width of your WRN>`

Then to train a student model via knowledge distillation use train_student_KT.py

`python train_student_KD.py <MODEL_TYPE> -s <PLACE TO SAVE STUDENT> -t <TEACHER NETWORK TO LEARN WITH> --GPU  <WHICH GPU TO USE> --wrn_depth <Depth of your student WRN> --wrn_width <Width of your student WRN>`

Training with attention transfer is more painful as you need to be able to extract intermediate activations from the networks.



# teach

Student-teacher toolbox for pytorch.

To use this you will have to install tqdm (e.g. pip install tqdm), torch, and pyinn (pip install git+https://github.com/szagoruyko/pyinn.git@master), then make a directory called "checkpoints"

teach_teacher.py trains a teacher.

`python teacher_train.py <MODEL_TYPE> -t <PLACE TO SAVE TEACHER> --GPU  <WHICH GPU TO USE> --wrn_depth <Depth of your WRN> --wrn_width <Width of your WRN>`

Run train_teacher.py to train a teacher
then train_student_distillation.py to train a student using knowledge distillation

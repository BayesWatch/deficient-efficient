# teach

Student-teacher toolbox for pytorch.

To use this you will have to install tqdm (e.g. pip install tqdm), torch, and pyinn (pip install git+https://github.com/szagoruyko/pyinn.git@master), then make a directory called "checkpoints"

train_teacher.py trains a teacher.

`python teacher_train.py <MODEL_TYPE> -t <PLACE TO SAVE TEACHER> --GPU  <WHICH GPU TO USE> --wrn_depth <Depth of your WRN> --wrn_width <Width of your WRN>`

Then to train a student model via knowledge distillation use train_student_KT.py

`python train_student_KD.py <MODEL_TYPE> -s <PLACE TO SAVE STUDENT> -t <TEACHER NETWORK TO LEARN WITH> --GPU  <WHICH GPU TO USE> --wrn_depth <Depth of your student WRN> --wrn_width <Width of your student WRN>`

Training with attention transfer is more painful as you need to be able to extract intermediate activations from the networks.

I've added a class to in models/wide_resnet.py called WideResNetInt that explicitly returns the output and three intermediate activations.

To turn a standard model into this format, you'll have to load it, and copy its state_dict to one of these.

Then you can use train_student_AT.py in the same manner as above.

`python train_student_KD.py <MODEL_TYPE> -s <PLACE TO SAVE STUDENT> -t <TEACHER NETWORK OF CLASS WIDERESNETINT TO LEARN WITH> --GPU  <WHICH GPU TO USE> --wrn_depth <Depth of your student WRN> --wrn_width <Width of your student WRN>`

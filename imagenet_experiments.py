import json

settings = ['ACDC_%i'%n for n in [28, 128]] +\
           ['SepHashed_%.2f'%s for s in [0.08, 0.58]] +\
           ['Generic_%.2f'%s for s in [0.03, 0.21]] +\
           ['Tucker_%.2f'%s for s in [0.25, 0.73]] +\
           ['TensorTrain%.2f'%s for s in [0.27, 0.75]] +\
           ['Shuffle_%i'%n for n in [7, 1]]

experiments = []

import datetime
now = datetime.datetime.now()
monthday = now.strftime("%B")[:3]+"%i"%now.day

# use these settings to train WideResNets from scratch
for s in settings:
    experiment = ["python", "main.py", "imagenet", "teacher", "--conv", s,
                  "-t", "wrn_50_2.%s.%s"%(s.lower(), monthday), "--network", "WRN_50_2"]
    experiments.append(experiment)
# and to train WideResNets with a teacher
for s in settings:
    experiment = ["python", "main.py", "imagenet", "student", "--conv", s, "-t", "wrn_50_2.imagenet.modelzoo",
                  "-s", "wrn_50_2.%s.student.%s"%(s.lower(), monthday), "--network", "WRN_50_2",
                  "--alpha", "0.", "--beta", "1e3"]
    experiments.append(experiment)

with open("wrn_50_2_imagenet.json", "w") as f:
    f.write(json.dumps(experiments))

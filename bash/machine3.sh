#!/usr/bin/env bash
array=( DConv DConvG2 DConvG4  )
for i in "${array[@]}"
do
    tmux -f "/disk/scratch/ecrowley/.tmux.conf" \
      new-session  "python main.py cifar100  teacher $i -t 402_100_T_A2B$i               --GPU 0" \; \
      split-window "python main.py cifar100  AT      $i -s 402_100_AT_A4B$i --alpha 0    --GPU 1"\; \
      split-window "python main.py cifar100  KD      $i -s 402_100_KD_A8B$i --alpha 0.9  --GPU 2" \; \
      select-layout even-vertical
done

array=( G4B2 G8B2 G16B2 )
for i in "${array[@]}"
    do
    tmux -f "/disk/scratch/ecrowley/.tmux.conf" \
      new-session  "python main.py cifar100  teacher $i -t B402_100_T_A2B$i               --GPU 0 --block Bottle" \; \
      split-window "python main.py cifar100  AT      $i -s B402_100_AT_A4B$i --alpha 0    --GPU 1 --block Bottle"\; \
      split-window "python main.py cifar100  KD      $i -s B402_100_KD_A8B$i --alpha 0.9  --GPU 2 --block Bottle" \; \
      select-layout even-vertical
done


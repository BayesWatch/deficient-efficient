#!/usr/bin/env bash
array=( DConv DConvG2 DConvG4 )
for i in "${array[@]}"
do
    tmux \
      new-session  "python main.py cifar100  teacher $i -t 402_100_T_A2B$i               --GPU 0" \; \
      split-window "python main.py cifar100  AT      $i -s 402_100_AT_A4B$i --alpha 0    --GPU 1"\; \
      split-window "python main.py cifar100  KD      $i -s 402_100_KD_A8B$i --alpha 0.9  --GPU 2" \; \
      select-layout even-vertical
done

array=( DConvG8 DConvG16 DConvA16 DConvA8 DConvA4 DConvA2 )
for i in "${array[@]}"
    do
    tmux \
      new-session  "python main.py cifar100  teacher $i -t B402_100_T_A2B$i               --GPU 0 --block Bottle" \; \
      split-window "python main.py cifar100  AT      $i -s B402_100_AT_A4B$i --alpha 0    --GPU 1 --block Bottle"\; \
      split-window "python main.py cifar100  KD      $i -s B402_100_KD_A8B$i --alpha 0.9  --GPU 2 --block Bottle" \; \
      select-layout even-vertical
done





#!/usr/bin/env bash
tmux \
  new-session  "python main.py cifar10 KD Conv -s kd0_$1$2 -t wrn_40_2_T --GPU 0 --wrn_depth $1 --wrn_width $2; read" \; \
  split-window "python main.py cifar10 KD Conv -s kd1_$1$2 -t wrn_40_2_T --GPU 1 --wrn_depth $1 --wrn_width $2; read" \; \
  split-window "python main.py cifar10 KD Conv -s kd2_$1$2 -t wrn_40_2_T --GPU 2 --wrn_depth $1 --wrn_width $2; read" \; \
  split-window "python main.py cifar10 KD Conv -s kd3_$1$2 -t wrn_40_2_T --GPU 3 --wrn_depth $1 --wrn_width $2; read" \; \
  select-layout even-vertical


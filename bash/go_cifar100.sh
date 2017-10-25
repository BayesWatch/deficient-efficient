#!/usr/bin/env bash
tmux \
  new-session  "python main.py cifar100 teacher Conv -t cifar100_0_$1$2 --GPU 0 --wrn_depth $1 --wrn_width $2; read" \; \
  split-window "python main.py cifar100 teacher Conv -t cifar100_1_$1$2 --GPU 1 --wrn_depth $1 --wrn_width $2; read" \; \
  split-window "python main.py cifar100 teacher Conv -t cifar100_2_$1$2 --GPU 2 --wrn_depth $1 --wrn_width $2; read" \; \
  split-window "python main.py cifar100 teacher Conv -t cifar100_3_$1$2 --GPU 3 --wrn_depth $1 --wrn_width $2; read" \; \
  select-layout even-vertical


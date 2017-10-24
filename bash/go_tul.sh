#!/usr/bin/env bash
tmux \
  new-session  "python main.py AT DConvB2 -s 402_AT_DConvB2--alpha 0 --GPU 0 --block Bottle" \; \
  split-window "python main.py AT DConvB4 -s 402_AT_DConvB4 --alpha 0 --GPU 1 --block Bottle"\; \
  split-window "python main.py AT DConvB8 -s 402_AT_DConvB8 --alpha 0 --GPU 2 --block Bottle" \; \
  split-window "python main.py AT DConvB16 -s 402_AT_DConvB16 --alpha 0 --GPU 3 --block Bottle" \; \
  select-layout even-vertical

tmux \
  new-session  "python main.py KD DConvB2 -s 402_KD_DConvB2 --GPU 0 --block Bottle" \; \
  split-window "python main.py KD DConvB4 -s 402_KD_DConvB4 --GPU 1 --block Bottle"\; \
  split-window "python main.py KD DConvB8 -s 402_KD_DConvB8 --GPU 2 --block Bottle" \; \
  split-window "python main.py KD DConvB16 -s 402_KD_DConvB16--GPU 3 --block Bottle" \; \
  select-layout even-vertical

tmux \
  new-session  "python main.py teacher DConvB2 -t 402_T_DConvB2 --GPU 0 --block Bottle" \; \
  split-window "python main.py teacher DConvB4 -t 402_T_DConvB4 --GPU 1 --block Bottle"\; \
  split-window "python main.py teacher DConvB8 -t 402_T_DConvB8 --GPU 2 --block Bottle" \; \
  split-window "python main.py teacher DConvB16 -t 402_T_DConvB16 --GPU 3 --block Bottle" \; \
  select-layout even-vertical

tmux \
  new-session  "python main.py teacher DConv -t 402_T_DConv1 --GPU 0" \; \
  split-window "python main.py AT DConv -s 402_AT_DConv1 --GPU 1 --alpha 0"\; \
  split-window "python main.py KD DConv -s 402_KD_DConv1 --GPU 2" \; \
  select-layout even-vertical

new-session  "python main.py teacher Conv2x2 -t 402_T_22 --GPU 0" \; \
  split-window "python main.py AT Conv2x2 -s 402_AT_22 --GPU 1 --alpha 0"\; \
  split-window "python main.py KD Conv2x2 -s 402_KD_22 --GPU 2" \; \
  select-layout even-vertical
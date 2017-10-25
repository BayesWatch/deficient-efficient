#!/usr/bin/env bash
tmux \
  new-session  "python main.py AT Conv --wrn_depth 40 --wrn_width 1  -s basic_40_1_AT --alpha 0 --GPU 0" \; \
  split-window "python main.py KD Conv --wrn_depth 40 --wrn_width 1  -s basic_40_1_KD --GPU 1"\; \
#  split-window "python main.py AT Conv --wrn_depth 16 --wrn_width 1  -s basic_16_1_AT --alpha 0 --GPU 2" \; \
#  split-window "python main.py KD Conv --wrn_depth 16 --wrn_width 1  -s basic_16_1_KD --GPU 3" \; \
  select-layout even-vertical
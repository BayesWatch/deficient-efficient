#!/usr/bin/env bash
tmux \
  new-session  "python main.py AT $1 -s AT_$1_split2_40_2 --GPU 0 --wrn_depth 40 --wrn_width 2 --alpha 0 --AT_split 2" \; \
  split-window "python main.py AT $1 -s AT_$1_split3_40_2 --GPU 1 --wrn_depth 40 --wrn_width 2 --alpha 0 --AT_split 3"\; \
  split-window "python main.py AT $1 -s AT_$1_split6_40_2 --GPU 2 --wrn_depth 40 --wrn_width 2 --alpha 0 --AT_split 6" \; \
  select-layout even-vertical
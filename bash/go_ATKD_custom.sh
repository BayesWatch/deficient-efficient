#!/usr/bin/env bash
tmux \
  new-session  "python main.py KD custom --customconv "Conv_Conv_$1" -s KD_3$1_40_2 --GPU 0 --wrn_depth 40 --wrn_width 2" \; \
  split-window "python main.py KD custom --customconv "Conv_Conv_$2" -s KD_3$2_40_2 --GPU 1 --wrn_depth 40 --wrn_width 2" \; \
  split-window "python main.py KD custom --customconv "Conv_Conv_$3" -s KD_3$3_40_2 --GPU 2 --wrn_depth 40 --wrn_width 2" \; \
  split-window "python main.py KD custom --customconv "Conv_Conv_$4" -s KD_3$4_40_2 --GPU 3 --wrn_depth 40 --wrn_width 2" \; \
  select-layout even-vertical

tmux \
  new-session  "python main.py AT custom --customconv "Conv_Conv_$1" -s AT_3$1_40_2 --GPU 0 --wrn_depth 40 --wrn_width 2 --alpha 0" \; \
  split-window "python main.py AT custom --customconv "Conv_Conv_$2" -s AT_3$2_40_2 --GPU 1 --wrn_depth 40 --wrn_width 2 --alpha 0" \; \
  split-window "python main.py AT custom --customconv "Conv_Conv_$3" -s AT_3$3_40_2 --GPU 2 --wrn_depth 40 --wrn_width 2 --alpha 0" \; \
  split-window "python main.py AT custom --customconv "Conv_Conv_$4" -s AT_3$4_40_2 --GPU 3 --wrn_depth 40 --wrn_width 2 --alpha 0" \; \
  select-layout even-vertical
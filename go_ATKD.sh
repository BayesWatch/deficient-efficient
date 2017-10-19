#!/usr/bin/env bash
tmux \
  new-session  "python main.py KD $1 -s KD_$1_40_2 --GPU 0 --wrn_depth 40 --wrn_width 2" \; \
  split-window "python main.py KD $2 -s KD_$2_40_2 --GPU 1 --wrn_depth 40 --wrn_width 2" \; \
  split-window "python main.py KD $3 -s KD_$3_40_2 --GPU 2 --wrn_depth 40 --wrn_width 2" \; \
  split-window "python main.py KD $4 -s KD_$4_40_2 --GPU 3 --wrn_depth 40 --wrn_width 2" \; \
  select-layout even-vertical

tmux \
  new-session  "python main.py AT $1 -s AT_$1_40_2 --GPU 0 --wrn_depth 40 --wrn_width 2 --alpha 0" \; \
  split-window "python main.py AT $2 -s AT_$2_40_2 --GPU 1 --wrn_depth 40 --wrn_width 2 --alpha 0" \; \
  split-window "python main.py AT $3 -s AT_$3_40_2 --GPU 2 --wrn_depth 40 --wrn_width 2 --alpha 0" \; \
  split-window "python main.py AT $4 -s AT_$4_40_2 --GPU 3 --wrn_depth 40 --wrn_width 2 --alpha 0" \; \
  select-layout even-vertical
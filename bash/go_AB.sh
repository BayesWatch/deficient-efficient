#!/usr/bin/env bash
tmux \
  new-session  "python main.py AT A2B$1 -s 402_AT_A2B$1 --alpha 0 --GPU 0 --block Bottle" \; \
  split-window "python main.py AT A4B$1 -s 402_AT_A4B$1 --alpha 0 --GPU 1 --block Bottle"\; \
  split-window "python main.py AT A8B$1 -s 402_AT_A8B$1 --alpha 0 --GPU 2 --block Bottle" \; \
  split-window "python main.py AT A16B$1 -s 402_AT_A16B$1 --alpha 0 --GPU 3 --block Bottle" \; \
  select-layout even-vertical

tmux \
  new-session  "python main.py KD A2B$1 -s 402_KD_A2B$1 --GPU 0 --block Bottle" \; \
  split-window "python main.py KD A4B$1 -s 402_KD_A4B$1 --GPU 1 --block Bottle"\; \
  split-window "python main.py KD A8B$1 -s 402_KD_A8B$1 --GPU 2 --block Bottle" \; \
  split-window "python main.py KD A16B$1 -s 402_KD_A16B$1 --GPU 3 --block Bottle" \; \
  select-layout even-vertical

tmux \
  new-session  "python main.py teacher A2B$1 -t 402_teach_A2B$1 --GPU 0 --block Bottle" \; \
  split-window "python main.py teacher A4B$1 -t 402_teach_A4B$1 --GPU 1 --block Bottle"\; \
  split-window "python main.py teacher A8B$1 -t 402_teach_A8B$1 --GPU 2 --block Bottle" \; \
  split-window "python main.py teacher A16B$1 -t 402_teach_A16B$1 --GPU 3 --block Bottle" \; \
  select-layout even-vertical
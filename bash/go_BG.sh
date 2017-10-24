#!/usr/bin/env bash
tmux \
  new-session  "python main.py AT G2B$1 -s 402_AT_G2B$1 --alpha 0 --GPU 0" \; \
  split-window "python main.py AT G2B$1 -s 402_AT_G2B$1 --alpha 0 --GPU 1"\; \
  split-window "python main.py AT G2B$1 -s 402_AT_G2B$1 --alpha 0 --GPU 2" \; \
  split-window "python main.py AT G2B$1 -s 402_AT_G2B$1 --alpha 0 --GPU 3" \; \
  select-layout even-vertical

tmux \
  new-session  "python main.py KD G2B$1 -s 402_KD_G2B$1 --GPU 0" \; \
  split-window "python main.py KD G2B$1 -s 402_KD_G2B$1 --GPU 1"\; \
  split-window "python main.py KD G2B$1 -s 402_KD_G2B$1 --GPU 2" \; \
  split-window "python main.py KD G2B$1 -s 402_KD_G2B$1 --GPU 3" \; \
  select-layout even-vertical

tmux \
  new-session  "python main.py teacher G2B$1 -t 402_teach_G2B$1 --GPU 0" \; \
  split-window "python main.py teacher G2B$1 -t 402_teach_G2B$1 --GPU 1"\; \
  split-window "python main.py teacher G2B$1 -t 402_teach_G2B$1 --GPU 2" \; \
  split-window "python main.py teacher G2B$1 -t 402_teach_G2B$1 --GPU 3" \; \
  select-layout even-vertical
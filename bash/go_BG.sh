#!/usr/bin/env bash
tmux \
  new-session  "python main.py AT G2B$1 -s 402_AT_G2B$1 --alpha 0 --GPU 0 --block Bottle" \; \
  split-window "python main.py AT G4B$1 -s 402_AT_G4B$1 --alpha 0 --GPU 1 --block Bottle"\; \
  split-window "python main.py AT G8B$1 -s 402_AT_G8B$1 --alpha 0 --GPU 2 --block Bottle" \; \
  split-window "python main.py AT G16B$1 -s 402_AT_G16B$1 --alpha 0 --GPU 3 --block Bottle" \; \
  select-layout even-vertical

tmux \
  new-session  "python main.py KD G2B$1 -s 402_KD_G2B$1 --GPU 0 --block Bottle" \; \
  split-window "python main.py KD G4B$1 -s 402_KD_G4B$1 --GPU 1 --block Bottle"\; \
  split-window "python main.py KD G8B$1 -s 402_KD_G8B$1 --GPU 2 --block Bottle" \; \
  split-window "python main.py KD G16B$1 -s 402_KD_G16B$1 --GPU 3 --block Bottle" \; \
  select-layout even-vertical

tmux \
  new-session  "python main.py teacher G2B$1 -t 402_teach_G2B$1 --GPU 0 --block Bottle" \; \
  split-window "python main.py teacher G4B$1 -t 402_teach_G4B$1 --GPU 1 --block Bottle"\; \
  split-window "python main.py teacher G8B$1 -t 402_teach_G8B$1 --GPU 2 --block Bottle" \; \
  split-window "python main.py teacher G16B$1 -t 402_teach_G16B$1 --GPU 3 --block Bottle" \; \
  select-layout even-vertical
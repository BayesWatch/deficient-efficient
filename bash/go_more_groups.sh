#!/usr/bin/env bash
tmux \
  new-session  "python main.py AT DConvA2  -s A2_basic_402_AT --alpha 0 --GPU 0" \; \
  split-window "python main.py AT DConvA4  -s A4_basic_402_AT --alpha 0 --GPU 1"\; \
  split-window "python main.py AT DConvA8  -s A8_basic_402_AT --alpha 0 --GPU 2" \; \
  split-window "python main.py AT DConvA16  -s A16_basic_402_AT --alpha 0 --GPU 3" \; \
  select-layout even-vertical


#!/usr/bin/env bash
tmux \
  new-session  "python main.py KD DConvA2  -s A2_basic_402_KD --GPU 0" \; \
  split-window "python main.py KD DConvA4  -s A4_basic_402_KD --GPU 1"\; \
  split-window "python main.py KD DConvA8  -s A8_basic_402_KD --GPU 2" \; \
  split-window "python main.py KD DConvA16  -s A_16_basic_402_KD --GPU 3" \; \
  select-layout even-vertical
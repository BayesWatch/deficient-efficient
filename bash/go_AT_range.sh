#!/bin/bash
# declare an array called array and define 3 vales
array=( DConvG4 DConvG8 DConvG16 DConv DConvG2 )
for i in "${array[@]}"
do
    tmux -f /disk/scratch/ecrowley/.tmux.conf \
      new-session  "python main.py KD $i -s mondayKD_40_2_$i --GPU 0 --wrn_depth 40 --wrn_width 2" \; \
      split-window "python main.py AT $i -s mondayAT_split1_40_2_$i --GPU 1 --wrn_depth 40 --wrn_width 2 --alpha 0 --AT_split 1"\; \
      split-window "python main.py AT $i -s mondayAT_split2_40_2_$i --GPU 2 --wrn_depth 40 --wrn_width 2 --alpha 0 --AT_split 2"\; \
      split-window "python main.py AT $i -s mondayAT_split3_40_2_$i --GPU 3 --wrn_depth 40 --wrn_width 2 --alpha 0 --AT_split 3" \; \
      select-layout even-vertical
done

for i in "${array[@]}"
do
    tmux -f /disk/scratch/ecrowley/.tmux.conf \
      new-session  "python main.py KD $i -s m2ondayKD_40_2_$i --GPU 0 --wrn_depth 40 --wrn_width 2" \; \
      split-window "python main.py AT $i -s m2ondayAT_split1_40_2_$i --GPU 1 --wrn_depth 40 --wrn_width 2 --alpha 0 --AT_split 1"\; \
      split-window "python main.py AT $i -s m2ondayAT_split2_40_2_$i --GPU 2 --wrn_depth 40 --wrn_width 2 --alpha 0 --AT_split 2"\; \
      split-window "python main.py AT $i -s m2ondayAT_split3_40_2_$i --GPU 3 --wrn_depth 40 --wrn_width 2 --alpha 0 --AT_split 3" \; \
      select-layout even-vertical
done












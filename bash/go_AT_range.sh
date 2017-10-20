#!/bin/bash
# declare an array called array and define 3 vales
array=( DConv DConvG2 DConvG4 DConvG8 DConvG16 )
for i in "${array[@]}"
do
    tmux \
      new-session  "python main.py AT $i -s wkAT_$i_split2_40_2 --GPU 0 --wrn_depth 40 --wrn_width 2 --alpha 0 --AT_split 1" \; \
      split-window "python main.py AT $i -s wkAT_$i_split3_40_2 --GPU 1 --wrn_depth 40 --wrn_width 2 --alpha 0 --AT_split 2"\; \
      split-window "python main.py AT $i -s wkAT_$i_split3_40_2 --GPU 2 --wrn_depth 40 --wrn_width 2 --alpha 0 --AT_split 3"\; \
      split-window "python main.py AT $i -s wkAT_$i_split6_40_2 --GPU 3 --wrn_depth 40 --wrn_width 2 --alpha 0 --AT_split 6" \; \
      select-layout even-vertical
done

for i in "${array[@]}"
do
    tmux \
      new-session  "python main.py AT $i -s wkAT_$i_Bsplit2_40_2 --GPU 0 --wrn_depth 40 --wrn_width 2 --alpha 0 --AT_split 1 --block Bottle" \; \
      split-window "python main.py AT $i -s wkAT_$i_Bsplit3_40_2 --GPU 1 --wrn_depth 40 --wrn_width 2 --alpha 0 --AT_split 2 --block Bottle"\; \
      split-window "python main.py AT $i -s wkAT_$i_Bsplit3_40_2 --GPU 2 --wrn_depth 40 --wrn_width 2 --alpha 0 --AT_split 3 --block Bottle"\; \
      split-window "python main.py AT $i -s wkAT_$i_Bsplit6_40_2 --GPU 3 --wrn_depth 40 --wrn_width 2 --alpha 0 --AT_split 6 --block Bottle" \; \
      select-layout even-vertical
done



#!/usr/bin/env bash
python train_student_KD.py WRN 3    -t wrn_40_2 -s wrn_$1_$2_3x3_KD_w_40_2 --wrn_depth $1 --wrn_width $2 --GPU $3
python train_student_KD.py WRN2x2 3 -t wrn_40_2 -s wrn_$1_$2_2x2_KD_w_40_2 --wrn_depth $1 --wrn_width $2 --GPU $3
python train_student_KD.py WRNsep 3 -t wrn_40_2 -s wrn_$1_$2_sep_KD_w_40_2 --wrn_depth $1 --wrn_width $2 --GPU $3

python train_student_AT.py WRN 3    -t wrn_40_2 -s wrn_$1_$2_3x3_AD_w_40_2 --wrn_depth $1 --wrn_width $2 --GPU $3
python train_student_AT.py WRN2x2 3 -t wrn_40_2 -s wrn_$1_$2_2x2_AD_w_40_2 --wrn_depth $1 --wrn_width $2 --GPU $3
python train_student_AT.py WRNsep 3 -t wrn_40_2 -s wrn_$1_$2_sep_AD_w_40_2 --wrn_depth $1 --wrn_width $2 --GPU $3

python train_student_AT.py WRN 3    -t wrn_40_2 -s wrn_$1_$2_3x3_ADKD_w_40_2 --wrn_depth $1 --wrn_width $2 --GPU $3 --alpha 0.9
python train_student_AT.py WRN2x2 3 -t wrn_40_2 -s wrn_$1_$2_2x2_ADKD_w_40_2 --wrn_depth $1 --wrn_width $2 --GPU $3 --alpha 0.9
python train_student_AT.py WRNsep 3 -t wrn_40_2 -s wrn_$1_$2_sep_ADKD_w_40_2 --wrn_depth $1 --wrn_width $2 --GPU $3 --alpha 0.9
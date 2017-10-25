#!/usr/bin/env bash

python main.py cifar100  teacher Conv -t 162_100_T_               --GPU 3 --wrn_width 2 --wrn_depth 16
python main.py cifar100  AT      Conv -s 162_100_AT_ --alpha 0    --GPU 3 --wrn_width 2 --wrn_depth 16
python main.py cifar100  KD      Conv -s 162_100_KD_ --alpha 0.9  --GPU 3 --wrn_width 2 --wrn_depth 16


python main.py cifar100  teacher Conv -t 161_100_T_               --GPU 3 --wrn_width 1 --wrn_depth 16
python main.py cifar100  AT      Conv -s 161_100_AT_ --alpha 0    --GPU 3 --wrn_width 1 --wrn_depth 16
python main.py cifar100  KD      Conv -s 161_100_KD_ --alpha 0.9  --GPU 3 --wrn_width 1 --wrn_depth 16


python main.py cifar100  teacher Conv -t 401_100_T_               --GPU 3 --wrn_width 1 --wrn_depth 40
python main.py cifar100  AT      Conv -s 401_100_AT_ --alpha 0    --GPU 3 --wrn_width 1 --wrn_depth 40
python main.py cifar100  KD      Conv -s 401_100_KD_ --alpha 0.9  --GPU 3 --wrn_width 1 --wrn_depth 40
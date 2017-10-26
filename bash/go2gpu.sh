#!/usr/bin/env bash

python main.py cifar100  teacher ConvB2 -t 402_100_T_bot2               --GPU 3 --Block Bottle
python main.py cifar100  AT      ConvB2 -s 402_100_AT_bot2 --alpha 0    --GPU 3 --Block Bottle
python main.py cifar100  KD      ConvB2 -s 402_100_KD_bot2 --alpha 0.9  --GPU 3 --Block Bottle

python main.py cifar100  teacher ConvB4 -t 402_100_T_bot4               --GPU 3 --Block Bottle
python main.py cifar100  AT      ConvB4 -s 402_100_AT_bot4 --alpha 0    --GPU 3 --Block Bottle
python main.py cifar100  KD      ConvB4 -s 402_100_KD_bot4 --alpha 0.9  --GPU 3 --Block Bottle

python main.py cifar100  teacher Conv2x2 -t 402_100_T_22               --GPU 3
python main.py cifar100  AT      Conv2x2 -s 402_100_AT_22 --alpha 0    --GPU 3
python main.py cifar100  KD      Conv2x2 -s 402_100_KD_22 --alpha 0.9  --GPU 3


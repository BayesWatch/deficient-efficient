#!/usr/bin/env bash
for arg; do
      python main.py cifar100  teacher ${arg} -t 402_100_T_${arg}              --GPU 3
      python main.py cifar100  AT      ${arg} -s 402_100_AT_${arg} --alpha 0    --GPU 3
      python main.py cifar100  KD      ${arg} -s 402_100_KD_${arg} --alpha 0.9  --GPU 3
done

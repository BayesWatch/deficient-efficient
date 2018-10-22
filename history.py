# opens checkpoints and prints the commands used to run each
import torch
import os

if __name__ == '__main__':
    ckpt_paths = os.listdir("checkpoints")
    for p in ckpt_paths:
        try:
            ckpt = torch.load("checkpoints/"+p)
            if 'args' in ckpt.keys():
                print(p)
                print("  " + " ".join(ckpt['args']))
        except:
            pass

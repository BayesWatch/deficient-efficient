We should Moonshine an Imagenet model and show it performs well with less parameters than a typical one.

For reference, if we take vanilla Resnet pre-trained models from Pytorch (irritatingly, not pre-activation) the performance looks like:

| Model    |  Top1   |  Top5   |
|----------|---------|---------|
| Resnet34 | 26.73   |  8.57   |    
| Resnet18 | 30.36   | 11.02   |     

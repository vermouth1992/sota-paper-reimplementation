# Image Classification

## Overview

Training large scale image classification models on personal laptops/desktops is impossible. This repo serves for
educational purpose

### Cifar10

```bash
# Training
python train_cifar10.py --model ResNet18 --total_epoch 200 --enable_amp --seed 4465;
# Testing
python train_cifar10.py --model ResNet50 --eval
```

| Model    | Individual Model Accuracy | Ensemble Accuracy |
|----------|-------                    |-------------------|
| ResNet18 | 95.61±0.06 | 96.23             |
| ResNet34 | 95.65±0.13 | 96.40             |
| ResNet50 |  95.51±0.18         |      96.21             |
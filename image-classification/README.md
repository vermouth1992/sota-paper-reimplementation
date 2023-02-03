# Image Classification

## Overview

Training large scale image classification models on personal laptops/desktops is impossible. This repo serves for
educational purpose

### Cifar10

```bash
# To see a list of implemented models
python train_cifar10.py --help
# Training
python train_cifar10.py --model {model} --total_epoch 200 --enable_amp --seed 4465;
# Testing
python train_cifar10.py --model {model} --eval
```

Ensemble accuracy: take the average of the logits produced by the network (without log_softmax or softmax), then compute the predicted class according to the argmax.

| Model    | Individual Model Accuracy | Ensemble Accuracy |
|----------|-------                    |-------------------|
| ResNet18 | 95.61±0.06 | 96.23             |
| ResNet34 | 95.65±0.13 | 96.40             |
| ResNet50 |  95.51±0.18         |      96.21             |
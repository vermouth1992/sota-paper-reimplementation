"""
Code modified from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
"""

import argparse
import os

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from tqdm.auto import tqdm

from models import create_model, get_available_models
from logger import setup_logger_kwargs, Logger

TASK = 'cifar10'


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--model', '-m', type=str, choices=get_available_models(TASK))
    parser.add_argument('--total_epoch', type=int, help='Total number of epochs')
    parser.add_argument('--enable_amp', action='store_true', help='enable amp')
    parser.add_argument('--batch_size', type=int, help='Batch size', default=128)
    parser.add_argument('--seed', type=int, default=4463)
    parser.add_argument('--logger_dir', type=str, default='checkpoint')
    args = vars(parser.parse_args())

    model = args['model']
    resume = args['resume']
    lr = args['lr']
    enable_amp = args['enable_amp']
    batch_size = args['batch_size']
    seed = args['seed']
    logger_dir = args['logger_dir']

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    logger = Logger(**setup_logger_kwargs(exp_name=f'cifar10_{model}', seed=seed,
                                          data_dir=logger_dir, datestamp=False))
    logger.save_config(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = create_model(task=TASK, name=model)
    net.to(device)

    checkpoint_path = os.path.join(logger.output_dir, 'ckpt.pth')

    if resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        total_epoch = checkpoint['total_epoch']
        if 'num_epochs' in args:
            total_epoch = args['total_epoch']
    else:
        best_acc = 0  # best test accuracy
        start_epoch = 0  # start from epoch 0 or last checkpoint epoch
        total_epoch = args['total_epoch']

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch - start_epoch)

    # Training
    def train(epoch):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        t = tqdm(trainloader)
        for batch_idx, (inputs, targets) in enumerate(t):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type=device, dtype=torch.float16, enabled=enable_amp):
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            loss = train_loss / (batch_idx + 1)
            acc = 100. * correct / total

            message = f'Epoch {epoch}/{total_epoch} | Training Loss: {loss:.3f} | ' \
                      f'Training Acc: {acc:.2f} ({correct}/{total})'

            t.set_description(message)

        logger.log_tabular('train_loss', loss)
        logger.log_tabular('train_acc', acc)

    def test(epoch):
        nonlocal best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            t = tqdm(testloader)
            for batch_idx, (inputs, targets) in enumerate(t):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                loss = test_loss / (batch_idx + 1)
                acc = 100. * correct / total

                message = f'Epoch {epoch}/{total_epoch} | Val Loss: {loss:.3f} | Val Acc: {acc:.2f} ({correct}/{total})'

                t.set_description(message)

            logger.log_tabular('val_loss', loss)
            logger.log_tabular('val_acc', acc)

        # Save checkpoint.
        acc = 100. * correct / total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'total_epoch': total_epoch
            }

            torch.save(state, checkpoint_path)
            best_acc = acc

    for epoch in range(start_epoch, total_epoch):
        logger.log_tabular('Epoch', epoch + 1)
        train(epoch)
        test(epoch)
        scheduler.step()
        logger.dump_tabular()


if __name__ == '__main__':
    main()

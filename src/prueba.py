import argparse
import git
import os
import numpy as np
from utils import *

parser = argparse.ArgumentParser(description='Main Vehicle Model Recognition training program')
parser.add_argument('-arch', default='VGG16A', help='Network architecture [VGG16A, VGG19]')
parser.add_argument('-batch_size', default='16', help='Batch size')
parser.add_argument('-data_cfg', default=None, help='Data labels')
parser.add_argument('-epochs', default='1000', help='Number of training epochs')
parser.add_argument('-lr', default='1e-4', help='Learning rate')
parser.add_argument('-model', default=None, help='Model path')
parser.add_argument('-multi', action='store_true', help='Use all available GPUs for training')
parser.add_argument('-n_classes', default='196', help='Number of different classes')
parser.add_argument('-output', default='classification', help='Network output [classification, siamese]')
parser.add_argument('-resume', action='store_true', help='Resume previous training')
parser.add_argument('-train_cfg', default=None, help='Load training configuration')


args = parser.parse_args()


ROOT_DIR = get_git_root(os.getcwd())
print(ROOT_DIR)


if args.resume:
    if not args.data_cfg:
        data_cfg_files = os.listdir(os.path.join(ROOT_DIR, 'cfg'))
        data_cfg_files.sort()
        args.data_cfg = os.path.join('cfg', data_cfg_files[-1])
    if not args.train_cfg:
        train_cfg_files = os.listdir(os.path.join(ROOT_DIR, 'cfg'))
        train_cfg_files.sort()
        args.train_cfg = os.path.join('cfg', train_cfg_files[-1])
    if not args.model:
        # Load last trained model
        pass

    # Load data_cfg
    files = os.listdir(os.path.join(ROOT_DIR, args.data_cfg))
    files = np.sort(files)
    if len(files) == 2:
        if files[0] == 'data_cfg.txt' and files[1] == 'train_cfg.txt':
            labels = load_data_cfg(os.path.join(ROOT_DIR, args.data_cfg))
        else:
            raise Error(1)
    else:
        raise Error(1)

    # Load train_cfg
    files = os.listdir(os.path.join(ROOT_DIR, args.train_cfg))
    files = np.sort(files)
    if len(files) == 2:
        if files[0] == 'data_cfg.txt' and files[1] == 'train_cfg.txt':
            train_cfg = load_train_cfg(os.path.join(ROOT_DIR, args.train_cfg))
            args.arch = train_cfg[0]
            args.batch_size = train_cfg[1]
            args.lr = train_cfg[2]
            args.output = train_cfg[3]
        else:
            raise Error(2)
    else:
        raise Error(2)

else:
    if args.data_cfg:
        labels = load_data_cfg(os.path.join(ROOT_DIR, args.data_cfg))
    else:
        





print(labels)
print(args.arch)






print('Everything OK')

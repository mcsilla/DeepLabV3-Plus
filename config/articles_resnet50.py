#!/usr/bin/env python

"""Module for training deeplabv3plus on camvid dataset."""

from glob import glob

import tensorflow as tf

images = sorted(glob('/articles/MagyarNemzet/**/*.jpg', recursive=True))
annotations = sorted(glob('/articles/MagyarNemzet/**/*.png', recursive=True))

N = int(len(images) * 0.9)

# Sample Configuration
CONFIG = {
    # We mandate specifying project_name and experiment_name in every config
    # file. They are used for wandb runs if wandb api key is specified.
    'project_name': 'deeplabv3-plus',
    'experiment_name': 'articles-segmentation-resnet-50-backbone',
    'train_dataset_config': {
        'images': images[:N],
        'labels': annotations[:N],
        'height': 512, 'width': 512, 'batch_size': 2
    },
    'val_dataset_config': {
        'images': images[N:],
        'labels': annotations[N:],
        'height': 512, 'width': 512, 'batch_size': 2
    },
    'strategy': tf.distribute.OneDeviceStrategy(device="/gpu:0"),
    'num_classes': 16,
    'backbone': 'resnet50',
    'learning_rate': 1e-5,
    'checkpoint_dir': "/checkpoints",
    'checkpoint_file_prefix': "deeplabv3plus_with_resnet50_",
    'epochs': 1
}

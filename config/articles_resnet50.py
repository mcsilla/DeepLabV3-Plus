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
        'tf_records': glob('/deeplab/dataset/test_records/articles_deeplab_0000.tfrecord'),
        'images': images[:N],
        'labels': annotations[:N],
        'height': 768, 'width': 768, 'batch_size': 1
    },
    'val_dataset_config': {
        'tf_records': glob('/deeplab/dataset/test_records/articles_deeplab_0000.tfrecord'),
        'images': images[N:],
        'labels': annotations[N:],
        'height': 768, 'width': 768, 'batch_size': 1
    },
    'strategy': tf.distribute.OneDeviceStrategy(device="/gpu:0"),
    'num_classes': 18,
    'backbone': 'resnet50',
    'learning_rate': 1e-4,
    'checkpoint_dir': "/checkpoints",
    'checkpoint_file_prefix': "deeplabv3plus_with_resnet50_",
    'epochs': 2,
    'steps_per_epoch': 2000,
    'validation_steps': 200,
}

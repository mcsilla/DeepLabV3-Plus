#!/usr/bin/env python

"""Module for training deeplabv3plus on articles."""

from glob import glob
from pathlib import Path
import tensorflow as tf

tfrec_prefix = Path('/mnt/noah/dev/csilla/deeplab/tfrecords/')

CONFIG = {
    # We mandate specifying project_name and experiment_name in every config
    # file. They are used for wandb runs if wandb api key is specified.
    'project_name': 'deeplabv3-plus',
    'experiment_name': 'articles-segmentation-resnet-50-backbone',
    'train_dataset_config': {
        # 'tf_records': tf.io.gfile.glob(str(tfrec_prefix / '*train*')),
        'tf_records': tf.io.gfile.glob('/tfrecords/*train*'),
        'height': 768, 'width': 768, 'batch_size': 1
    },
    'val_dataset_config': {
        # 'tf_records': tf.io.gfile.glob(str(tfrec_prefix / '*val*')),
        'tf_records': tf.io.gfile.glob('/tfrecords/*val*'),
        'height': 768, 'width': 768, 'batch_size': 1
    },
    'strategy': 'onedevice',
    'num_classes': 18,
    'backbone': 'resnet50',
    'learning_rate': 1e-4,
    'checkpoint_dir': "/checkpoints",
    'checkpoint_file_prefix': "deeplabv3plus_articles_",
    'epochs': 2,
    'steps_per_epoch': 2000,
    'validation_steps': 200,
}

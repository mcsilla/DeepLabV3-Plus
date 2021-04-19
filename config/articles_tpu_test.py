#!/usr/bin/env python

"""Module for training deeplabv3plus on articles."""

from glob import glob
from pathlib import Path
import tensorflow as tf

tfrec_train_pattern = 'gs://arcanum-ml/cv/articles/deeplab/test_records/*train*'
tfrec_val_pattern = 'gs://arcanum-ml/cv/articles/deeplab/test_records/*val*'
model_dir = 'gs://arcanum-ml/cv/articles/deeplab/model'
log_dir = 'gs://arcanum-ml/cv/articles/deeplab/model/logs'

CONFIG = {
    # We mandate specifying project_name and experiment_name in every config
    # file. They are used for wandb runs if wandb api key is specified.
    'project_name': 'deeplabv3-plus',
    'experiment_name': 'articles-segmentation-resnet-50-backbone',
    'train_dataset_config': {
        'tf_records': tf.io.gfile.glob(tfrec_train_pattern),
        'height': 768, 'width': 768, 'batch_size': 1
    },
    'val_dataset_config': {
        'tf_records': tf.io.gfile.glob(tfrec_val_pattern),
        'height': 768, 'width': 768, 'batch_size': 1
    },
    'strategy': 'tpu',
    'tpu_name': 'deeplab-articles',
    'num_classes': 18,
    'backbone': 'resnet50',
    'learning_rate': 1e-4,
    'checkpoint_dir': model_dir,
    'checkpoint_file_prefix': "ckpt_",
    'log_dir': log_dir,
    'epochs': 1,
    'steps_per_epoch': 40,
    'validation_steps': 5,
}
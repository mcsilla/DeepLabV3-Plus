#!/usr/bin/env python

"""Module for training deeplabv3plus on articles."""

import tensorflow as tf

tfrec_train_pattern = 'gs://arcanum-ml/cv/articles/deeplab/tfrecords-all-with-random-validation/train*'
tfrec_val_pattern = 'gs://arcanum-ml/cv/articles/deeplab/tfrecords-all-with-random-validation/val*'
model_dir = 'gs://arcanum-ml/cv/articles/deeplab/model-all-with-random-validation/article-segmentation'
log_dir = 'gs://arcanum-ml/cv/articles/deeplab/model-all-with-random-validation/article-segmentation/logs'

CONFIG = {
    # We mandate specifying project_name and experiment_name in every config
    # file. They are used for wandb runs if wandb api key is specified.
    'project_name': 'deeplabv3-plus',
    'experiment_name': 'articles-segmentation-resnet-50-backbone',
    'train_dataset_config': {
        'tf_records': tf.io.gfile.glob(tfrec_train_pattern),
        'height': 1024, 'width': 1024, 'batch_size': 32
    },
    'val_dataset_config': {
        'tf_records': tf.io.gfile.glob(tfrec_val_pattern),
        'height': 1024, 'width': 1024, 'batch_size': 32
    },
    'strategy': 'tpu',
    'mode': 'gcp',
    'tpu_name': 'article-segmentation',
    'num_classes': 2,
    'backbone': 'resnet50',
    'initial_learning_rate': 5e-4,
    'end_learning_rate': 1e-5,
    'checkpoint_dir': model_dir,
    'checkpoint_file_prefix': "ckpt_",
    'log_dir': log_dir,
    'epochs': 50,
    'power': 0.9
}

num_of_train_examples = 143128
num_of_val_examples = 7005
steps_per_epoch = num_of_train_example // CONFIG['train_dataset_config']['batch_size']
CONFIG['decay_steps'] = steps_per_epoch * 18

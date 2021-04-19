#!/usr/bin/env python

"""Module for training deeplabv3plus on camvid dataset."""

# !pylint:disable=wrong-import-position

import argparse
from argparse import RawTextHelpFormatter

print("[-] Importing tensorflow...")
import tensorflow as tf  # noqa: E402
print(f"[+] Done! Tensorflow version: {tf.version.VERSION}")

print("[-] Importing Deeplabv3plus Trainer class...")
from deeplabv3plus.train import Trainer  # noqa: E402

print("[-] Importing config files...")
from config import CONFIG_MAP  # noqa: E402


if __name__ == "__main__":
    REGISTERED_CONFIG_KEYS = "".join(map(lambda s: f"  {s}\n", CONFIG_MAP.keys()))

    PARSER = argparse.ArgumentParser(
        description=f"""
Runs DeeplabV3+ trainer with the given config setting.

Registered config_key values:
{REGISTERED_CONFIG_KEYS}""",
        formatter_class=RawTextHelpFormatter
    )
    PARSER.add_argument('config_key', help="Key to use while looking up "
                        "configuration from the CONFIG_MAP dictionary.")
    PARSER.add_argument("--wandb_api_key",
                        help="""Wandb API Key for logging run on Wandb.
If provided, checkpoint_dir is set to wandb://
(Model checkpoints are saved to wandb.)""",
                        default=None)
    ARGS = PARSER.parse_args()

    CONFIG = CONFIG_MAP[ARGS.config_key]
    if ARGS.wandb_api_key is not None:
        CONFIG['wandb_api_key'] = ARGS.wandb_api_key
        CONFIG['checkpoint_dir'] = "wandb://"

    tf.config.set_visible_devices(tf.config.list_physical_devices("GPU")[0:7], "GPU")
    print('GPU Devices: {}'.format([device.name for device in tf.config.list_physical_devices("GPU")]))
    print('Phisical Devices: {}'.format([device.name for device in tf.config.list_physical_devices()]))
    print('Logical Devices: {}'.format([device.name for device in tf.config.list_logical_devices()]))
    strategy = None
    if CONFIG['strategy'] == "onedevice":
        CONFIG['strategy'] = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    if CONFIG['strategy'] == "mirrored":
        CONFIG['strategy'] = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(CONFIG['strategy'].num_replicas_in_sync))
    if CONFIG['strategy'] == "tpu":
        if CONFIG['mode'] == 'colab':
            # Get a handle to the attached TPU. On GCP it will be the CloudTPU itself
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=’grpc: // ’ + os.environ[‘COLAB_TPU_ADDR’])
            # Connect to the TPU handle and initialise it
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            CONFIG['strategy'] = tf.distribute.experimental.TPUStrategy(resolver)
        if CONFIG['mode'] == 'gcp':
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=CONFIG['tpu_name'])
            print(f"Connecting to tpu {CONFIG['tpu_name']}...")
            tf.config.experimental_connect_to_cluster(resolver)
            print(f"Initializing tpu {CONFIG['tpu_name']}...")
            tf.tpu.experimental.initialize_tpu_system(resolver)
            print("All TPU devices: ", tf.config.list_logical_devices('TPU'))
            CONFIG['strategy'] = tf.distribute.experimental.TPUStrategy(resolver)

    TRAINER = Trainer(CONFIG)
    HISTORY = TRAINER.train()

import tensorflow as tf
import PIL.Image as Image
import numpy as np
from .tf_example_decoder import TfExampleDecoder
from .transformations import make_random_transformation, create_input
class GenericDataLoader:

    def __init__(self, configs):
        self.configs = configs
        self._parser_fn = TfExampleDecoder().parse_example
        self._transform_fn = make_random_transformation
        self._create_input_fn = lambda image, label: create_input(image, label, height=1024, width=1024)

    def get_dataset(self, dataset_type):
        record_names_dataset = tf.data.Dataset.from_tensor_slices(self.configs['tf_records'])
        record_names_dataset.shuffle(len(self.configs['tf_records']), reshuffle_each_iteration=True)
        dataset = record_names_dataset.interleave(
            map_func=tf.data.TFRecordDataset,
            cycle_length=8, # the number of input elements that will be processed concurrently
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.map(self._parser_fn)
        if dataset_type == 'train':
            dataset = dataset.map(self._transform_fn)
        dataset = dataset.map(self._create_input_fn)
        dataset = dataset.shuffle(5000, reshuffle_each_iteration=True)
        dataset = dataset.batch(self.configs['batch_size'], drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

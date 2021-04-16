import tensorflow as tf
import PIL.Image as Image
import numpy as np
from .tf_example_decoder import TfExampleDecoder
class GenericDataLoader:

    def __init__(self, configs):
        self.configs = configs
        self._parser_fn = TfExampleDecoder().create_input
        # self.assert_dataset()

    # def assert_dataset(self):
    #     assert 'images' in self.configs and 'labels' in self.configs
    #     assert len(self.configs['images']) == len(self.configs['labels'])
    #     print('Train Images are good to go')
    #
    # def __len__(self):
    #     return len(self.configs['images'])
    #
    # def read_img_2(self, image_path, mask=False):
    #     if mask:
    #         image = tf.convert_to_tensor(np.array(Image.open(image_path)))
    #         image = tf.expand_dims(image, -1)
    #         # image.set_shape([None, None, 1])
    #         image = (tf.image.resize_with_pad(
    #             image=image,
    #             target_height=self.configs['height'],
    #             target_width=self.configs['width'],
    #             method="nearest"
    #         ))
    #         image = tf.cast(image, tf.float32)
    #     else:
    #         image = tf.io.read_file(image_path)
    #         image = tf.io.decode_image(image, channels=3)
    #         image.set_shape([None, None, 3])
    #         image = (tf.image.resize(
    #             images=image, size=[
    #                 self.configs['height'],
    #                 self.configs['width']
    #             ]
    #         ))
    #         image = tf.cast(image, tf.float32) / 127.5 - 1
    #     return image
    #
    # def _map_function_2(self, image_list, mask_list):
    #     for img_path, mask_path in zip(image_list, mask_list):
    #         image = self.read_img_2(img_path)
    #         mask = self.read_img_2(mask_path, mask=True)
    #         yield image, mask
    #
    # def get_dataset_2(self):
    #     dataset = tf.data.Dataset.from_generator(lambda: self._map_function_2(
    #         self.configs['images'], self.configs['labels']
    #     ),
    #         output_types=(tf.float32, tf.float32),
    #         output_shapes=(
    #             tf.TensorShape([self.configs['height'], self.configs['width'], 3]),
    #             tf.TensorShape([self.configs['height'], self.configs['width'], 1])
    #         )
    #     )
    #     dataset = dataset.batch(self.configs['batch_size'], drop_remainder=True)
    #     dataset = dataset.repeat()
    #     dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    #     return dataset

    def get_dataset(self):
        record_names_dataset = tf.data.Dataset.from_tensor_slices(self.configs['tf_records'])
        dataset = record_names_dataset.interleave(
            map_func=tf.data.TFRecordDataset,
            cycle_length=1, # the number of input elements that will be processed concurrently
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # def f(x):
        #     print("AAAA", x)
        #     return x
        dataset = dataset.map(self._parser_fn)
        dataset = dataset.shuffle(1000)
        dataset = dataset.batch(self.configs['batch_size'], drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

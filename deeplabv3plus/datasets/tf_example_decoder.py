import tensorflow as tf
import cProfile

class TfExampleDecoder:
    """Tensorflow Example proto decoder."""

    def __init__(self):
        self._keys_to_features = {
            'image/encoded':
                tf.io.FixedLenFeature((), tf.string),
            'image/filename':
                tf.io.FixedLenFeature((), tf.string),
            'image/height':
                tf.io.FixedLenFeature((), tf.int64),
            'image/width':
                tf.io.FixedLenFeature((), tf.int64),
            'label/encoded':
                tf.io.FixedLenFeature((), tf.string),
            'label/filename':
                tf.io.FixedLenFeature((), tf.string)
        }
    def _decode_image(self, content, channels):
      return tf.cond(
          tf.image.is_jpeg(content),
          lambda: tf.image.decode_jpeg(content, channels),
          lambda: tf.image.decode_png(content, channels))

    def decode(self, serialized_example):
        parsed_tensors = tf.io.parse_single_example(
            serialized=serialized_example, features=self._keys_to_features)
        image = self._decode_image(parsed_tensors['image/encoded'], 3)
        label = self._decode_image(parsed_tensors['label/encoded'], 1)
        width = parsed_tensors['image/width']
        height = parsed_tensors['image/height']
        return image, label, width, height

    def parse_example(self, example):
        image, label, width, height = self.decode(example)
        return image, label
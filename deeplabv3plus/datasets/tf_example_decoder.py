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

    # def _decode_image(self, parsed_tensors):
    #     """Decodes the image and set its static shape."""
    #     image = tf.io.decode_image(parsed_tensors['image/encoded'], channels=3)
    #     image = tf.ensure_shape(image, [None, None, 3])
    #     return image
    #
    # def _decode_label(self, parsed_tensors):
    #     """Decodes the label and set its static shape."""
    #     label = tf.io.decode_png(parsed_tensors['label/encoded'], channels=1)
    #     label = tf.ensure_shape(label, [None, None, 1])
    #     return label
    def decode(self, serialized_example):
        parsed_tensors = tf.io.parse_single_example(
            serialized=serialized_example, features=self._keys_to_features)
        image = self._decode_image(parsed_tensors['image/encoded'], 3)
        label = self._decode_image(parsed_tensors['label/encoded'], 1)
        width = parsed_tensors['image/width']
        height = parsed_tensors['image/height']
        return image, label, width, height

    def create_input(self, example):
        image, label, width, height = self.decode(example)
        image = tf.cast(image, tf.float32) / 127.5 - 1
        label = tf.cast(label, tf.float32)
        tf.print('SHAPES: ', image.shape, label.shape)
        image = tf.ensure_shape(image, [768, 768, 3])
        label = tf.ensure_shape(label, [768, 768, 1])
        tf.print('Final SHAPES: ', image.shape, label.shape)
        # label = tf.reshape(label, tf.stack([height, width, tf.constant(1, dtype=tf.int64)], 0))
        # print(tf.executing_eagerly())
        # print('SHAPES_NEW: ', tf.shape(image), tf.shape(label), height, width)
        return image, label
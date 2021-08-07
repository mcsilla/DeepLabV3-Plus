import tensorflow as tf
import cProfile

class TfExampleDecoder:
    """Tensorflow Example proto decoder."""

    def __init__(self):
        self._keys_to_features = {
            'image/encoded':
                tf.io.FixedLenFeature((), tf.string),
            'image_black/encoded':
                tf.io.FixedLenFeature((), tf.string),
            'image/filename':
                tf.io.FixedLenFeature((), tf.string),
            'image/height':
                tf.io.FixedLenFeature((), tf.int64),
            'image/width':
                tf.io.FixedLenFeature((), tf.int64),
            'label_article_segmentation/encoded':
                tf.io.FixedLenFeature((), tf.string),
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
        label = self._decode_image(parsed_tensors['label_article_segmentation/encoded'], 1)
        label = tf.math.divide(label, tf.constant(255, dtype=tf.uint8))
        image_black = self._decode_image(parsed_tensors['image_black/encoded'], 1)
        image_black = tf.image.grayscale_to_rgb(image_black)
        width = parsed_tensors['image/width']
        height = parsed_tensors['image/height']
        return image, image_black, label, width, height

    def make_gray(self, image, label):
        gray_image = tf.image.rgb_to_grayscale(image)
        return tf.image.grayscale_to_rgb(gray_image), label

    def parse_example(self, example):
        image, image_black, label, width, height = self.decode(example)
        rand = tf.random.uniform(
            [], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=None, name=None
        )
        gray = tf.constant(0.15)
        black = tf.constant(0.25)
        gray_fn = lambda: self.make_gray(image, label)
        black_fn = lambda: (image_black, label)
        id_fn = lambda: (image, label)
        image, label = tf.case([(tf.less(rand, gray), gray_fn),
                                (tf.less(rand, black), black_fn)], exclusive=False, default=id_fn)
        return image, label
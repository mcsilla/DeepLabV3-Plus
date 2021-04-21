import tensorflow as tf
import numpy as np


def random_brightness(image, label):
    seed = tf.random.uniform(
        [2], minval=0, maxval=10, dtype=tf.dtypes.int32, seed=None, name=None
    )
    return tf.image.stateless_random_brightness(image, max_delta=0.1, seed=seed), label

def random_contrast(image, label):
    seed = tf.random.uniform(
        [2], minval=0, maxval=10, dtype=tf.dtypes.int32, seed=None, name=None
    )
    return tf.image.stateless_random_contrast(image, lower=0.9, upper=1.5, seed=seed), label

def random_crop(image, label):
    fraction = tf.random.uniform(
        [], minval=0.95, maxval=1, dtype=tf.dtypes.float32, seed=None, name=None
    )
    return (tf.image.central_crop(image, central_fraction=fraction), 
            tf.image.central_crop(label, central_fraction=fraction))

def random_hue(image, label):
    seed = tf.random.uniform(
        [2], minval=0, maxval=10, dtype=tf.dtypes.int32, seed=None, name=None
    )
    return tf.image.stateless_random_hue(image, max_delta=0.08, seed=seed), label

def random_saturation(image, label):
    seed = tf.random.uniform(
        [2], minval=0, maxval=10, dtype=tf.dtypes.int32, seed=None, name=None
    )
    return tf.image.stateless_random_saturation(image, lower=0.5, upper=1.5, seed=seed), label

def make_gray(image, label):
    gray_image = tf.image.rgb_to_grayscale(image)
    return tf.image.grayscale_to_rgb(gray_image), label

def make_random_transformation(image, label):
    rand = tf.random.uniform(
        [], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=None, name=None
    )
    rand_crop = tf.random.uniform(
        [], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=None, name=None
    )
    rand_grey = tf.random.uniform(
        [], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=None, name=None
    )
    if rand_crop < 0.05:
        image, label = random_crop(image, label)
    if rand_grey < 0.02:
        return make_gray(image, label)
    if rand < 0.25:
        return random_brightness(image, label)
    if rand < 0.5:
        return random_hue(image, label)
    if rand < 0.75:
        return random_saturation(image, label)
    return random_contrast(image, label)

def resize(image, label, height, width):
    resized_image = tf.image.resize_with_pad(image, height, width, method='nearest')
    resized_label = tf.image.resize_with_pad(label, height, width, method='nearest')
    return resized_image, resized_label

def transform(image, label, height=768, width=768):
    image, label = make_random_transformation(image, label)
    image, label = resize(image, label, height, width)
    return image, label

def create_input(image, label):
    image = tf.cast(image, tf.float32) / 127.5 - 1
    label = tf.cast(label, tf.float32)
    image = tf.ensure_shape(image, [768, 768, 3])
    label = tf.ensure_shape(label, [768, 768, 1])
    return image, label
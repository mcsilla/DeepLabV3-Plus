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
    return tf.image.stateless_random_contrast(image, lower=0.8, upper=2, seed=seed), label

# def random_crop(image, label):
#     fraction = np.random.uniform(0.95, 1)
#     # fraction = tf.random.uniform(
#     #     [], minval=0.95, maxval=1, dtype=tf.dtypes.float32, seed=None, name=None
#     # )
#     return (tf.image.central_crop(image, central_fraction=fraction),
#             tf.image.central_crop(label, central_fraction=fraction))

def random_crop(image, label):
    rand = tf.random.uniform(
        [], minval=0.95, maxval=1, dtype=tf.dtypes.float32, seed=None, name=None
    )
    seed = tf.random.uniform(
        [2], minval=0, maxval=10, dtype=tf.dtypes.int32, seed=None, name=None
    )
    new_size = tf.cast(tf.round(tf.cast(tf.shape(image)[:2], dtype=tf.float32) * rand), tf.int32)
    new_image_size = tf.concat([new_size, tf.constant([3])], axis=0)
    new_label_size = tf.concat([new_size, tf.constant([1])], axis=0)
    new_image = tf.image.stateless_random_crop(
        image, new_image_size, seed=seed, name=None
    )
    new_label = tf.image.stateless_random_crop(
        label, new_label_size, seed=seed, name=None
    )
    return new_image, new_label

def random_hue(image, label):
    seed = tf.random.uniform(
        [2], minval=0, maxval=10, dtype=tf.dtypes.int32, seed=None, name=None
    )
    return tf.image.stateless_random_hue(image, max_delta=0.2, seed=seed), label

def random_saturation(image, label):
    seed = tf.random.uniform(
        [2], minval=0, maxval=10, dtype=tf.dtypes.int32, seed=None, name=None
    )
    return tf.image.stateless_random_saturation(image, lower=0.2, upper=2.5, seed=seed), label

def make_gray(image, label):
    gray_image = tf.image.rgb_to_grayscale(image)
    return tf.image.grayscale_to_rgb(gray_image), label

# def make_random_transformation(image, label):
#     rand = np.random.uniform(0, 1)
#     rand_crop = np.random.uniform(0, 1)
#     rand_grey = np.random.uniform(0, 1)
#     if rand_crop < 0.05:
#         image, label = random_crop(image, label)
#     if rand_grey < 0.02:
#         return make_gray(image, label)
#     if rand < 0.25:
#         return random_brightness(image, label)
#     if rand < 0.5:
#         return random_hue(image, label)
#     if rand < 0.75:
#         return random_saturation(image, label)
#     return random_contrast(image, label)

def make_random_transformation(image, label):
    rand = tf.random.uniform(
        [], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=None, name=None
    )
    rand_crop = tf.random.uniform(
        [], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=None, name=None
    )
    rand_gray = tf.random.uniform(
        [], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=None, name=None
    )
    crop = tf.constant(0.1)
    gray = tf.constant(0.15)
    borders = tf.constant([0.25, 0.5, 0.75, 1])
    crop_fn = lambda: random_crop(image, label)
    gray_fn = lambda: make_gray(image, label)
    hue_fn = lambda: random_hue(image, label)
    saturation_fn = lambda: random_saturation(image, label)
    brightness_fn = lambda: random_brightness(image, label)
    contrast_fn = lambda: random_contrast(image, label)
    id_fn = lambda: (image, label)
    image, label = tf.cond(
        tf.math.less(rand_crop, crop), true_fn=crop_fn, false_fn=id_fn, name=None
    )
    image, label = tf.cond(
        tf.math.less(rand_gray, gray), true_fn=gray_fn, false_fn=id_fn, name=None
    )
    image, label = tf.case([(tf.less(rand, borders[0]), hue_fn),
                            (tf.less(rand, borders[1]), saturation_fn),
                            (tf.less(rand, borders[2]), brightness_fn),
                            (tf.less(rand, borders[3]), contrast_fn)], exclusive=False)
    return image, label

def resize(image, label, height, width):
    resized_image = tf.image.resize_with_pad(image, height, width, method='nearest')
    resized_label = tf.image.resize_with_pad(label, height, width, method='nearest')
    return resized_image, resized_label

# def transform(image, label, height=768, width=768):
#     image, label = make_random_transformation(image, label)
#     image, label = resize(image, label, height, width)
#     return image, label

def create_input(image, label, height=768, width=768):
    image, label = resize(image, label, height, width)
    image = tf.cast(image, tf.float32) / 127.5 - 1
    label = tf.cast(label, tf.float32)
    image = tf.ensure_shape(image, [768, 768, 3])
    label = tf.ensure_shape(label, [768, 768, 1])
    return image, label
import numpy as np
import tensorflow as tf
from model import DeeplabV3Plus
from datasets.create_tfrecords import resize_image
import PIL.Image as Image
from pathlib import Path
import random
from glob import glob
import tqdm

def read_image(image_path, target_height, target_width):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image, channels=3)
    image.set_shape([None, None, 3])
    image = tf.image.resize_with_pad(image, target_width, target_height)
    image = tf.cast(image, tf.float32) / 127.5 - 1
    return image

def infer(model, image_tensor, height, width):
    image_tensor = tf.ensure_shape(tf.expand_dims(image_tensor, axis=0), [1, height, width, 3])
    predictions = model(image_tensor)
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions

def draw_mask_on_image_array(image, mask, color=(255, 0, 0), alpha=0.4):
    """Draws mask on an image.
    Args:
    image: uint8 numpy array with shape (img_height, img_height, 3)
    mask: a uint8 numpy array of shape (img_height, img_height) with values
      between either 0 or 1.
    color: color to draw the keypoints with. Default is red.
    alpha: transparency value between 0 and 1. (default: 0.4)
    Raises:
    ValueError: On incorrect data type for image or masks.
    """
    if image.dtype != np.uint8:
        raise ValueError('`image` not of type np.uint8')
    if mask.dtype != np.uint8:
        raise ValueError('`mask` not of type np.uint8')
    if np.any(np.logical_and(mask != 1, mask != 0)):
        raise ValueError('`mask` elements should be in [0, 1]')
    if image.shape[:2] != mask.shape:
        raise ValueError('The image has spatial dimensions %s but the mask has '
                     'dimensions %s' % (image.shape[:2], mask.shape))
    rgb = color
    pil_image = Image.fromarray(image)

    solid_color = np.expand_dims(
        np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
    pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
    pil_mask = Image.fromarray(np.uint8(255.0 * alpha * mask)).convert('L')
    pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
    np.copyto(image, np.array(pil_image.convert('RGB')))

if __name__ == '__main__':
    COLORS = sorted([
        (0, 0, 0),
        (1, 0, 150),
        (2, 50, 255),
        (3, 127, 127),
        (4, 255, 255),
        (5, 85, 0),
        (6, 170, 0),
        (7, 255, 0),
        (85, 0, 0),
        (100, 0, 255),
        (127, 0, 127),
        (170, 0, 0),
        (179, 179, 255),
        (251, 165, 0),
        (252, 255, 0),
        (253, 0, 0),
        (254, 0, 255),
        (255, 255, 220)
    ])

    model = DeeplabV3Plus(
        num_classes=18,
        backbone='resnet50'
    )
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    checkpoint_path = '/mnt/noah/dev/csilla/cv/articles/deeplab/model_augmented/ckpt_0022'
    model.load_weights(checkpoint_path)

    input_prefix = Path('/mnt/noah/dev/training_data/vision/article/')
    random_train_images = random.choices([img for img in input_prefix.glob('**/*.jpg') if not str(img.relative_to(input_prefix)).startswith(
                    ('MagyarNemzet', 'VilagIfjusaga', 'KiadokKronosz'))], k=100)
    random_val_images = random.choices(glob(str(input_prefix / 'MagyarNemzet/**/*.jpg'), recursive=True) + glob(str(input_prefix/'KiadokKronosz/**/*.jpg'), recursive=True) + glob(str(input_prefix/'VilagIfjusaga/**/*.jpg'), recursive=True), k=100)

    # image_path = '/mnt/noah/dev/training_data/vision/article/PestiHirlap/PestiHirlap_1944_12/PestiHirlap_1944_12_06/0022.jpg'

    WIDTH = 768
    HEIGHT = 768

    i = 0
    for image_path in tqdm.tqdm(random_val_images):
        image_path = str(image_path)
        image_tensor = read_image(image_path, 768, 768)
        resized_image_array = np.array(resize_image(image_path, 768, 768))

        res = infer(model, image_tensor, HEIGHT, WIDTH)
        class_ids = np.unique(res)

        for class_id in class_ids:
            if class_id == 0:
                continue
            mask = np.zeros_like(res, dtype=np.uint8)
            mask[res == class_id] = 1
            color = COLORS[class_id]
            draw_mask_on_image_array(resized_image_array, mask, color=color, alpha=0.4)

        with Image.fromarray(resized_image_array) as img:
            img.save(f'/mnt/noah/dev/csilla/cv/articles/test_images_model_augmented/ckpt_0022/val_test_{i}.jpg')
        i += 1
import numpy as np
import tensorflow as tf
from model import DeeplabV3Plus
from datasets.create_tfrecords import resize_image_from_path
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
    checkpoint_path = '/mnt/noah/dev/csilla/cv/articles/deeplab/model_new/ckpt_0022'
    model.load_weights(checkpoint_path)

    input_prefix = Path('/mnt/noah/dev/training_data/vision/article/')
    # random.seed(7)
    # random_train_images = random.choices([img for img in input_prefix.glob('**/*.jpg') if not str(img.relative_to(input_prefix)).startswith(
    #                 ('MagyarNemzet', 'VilagIfjusaga', 'KiadokKronosz'))], k=100)
    # random.seed(5)
    # random_val_images = random.choices(glob(str(input_prefix / 'MagyarNemzet/**/*.jpg'), recursive=True) + glob(str(input_prefix/'KiadokKronosz/**/*.jpg'), recursive=True) + glob(str(input_prefix/'VilagIfjusaga/**/*.jpg'), recursive=True), k=100)

    # image_path = '/mnt/noah/dev/training_data/vision/article/PestiHirlap/PestiHirlap_1944_12/PestiHirlap_1944_12_06/0022.jpg'
    random_train_images = []
    random.seed(1)
    random_train_images += random.choices(sorted(glob(str(input_prefix / 'NemzetiSport/**/*.jpg'), recursive=True)), k=20)
    random.seed(2)
    random_train_images += random.choices(sorted(glob(str(input_prefix / 'MagyarHirlap/**/*.jpg'), recursive=True)), k=20)
    random.seed(3)
    random_train_images += random.choices(sorted(glob(str(input_prefix / 'PestiNaplo/**/*.jpg'), recursive=True)), k=20)
    random.seed(4)
    random_train_images += random.choices(sorted(glob(str(input_prefix / 'VasarnapiUjsag/**/*.jpg'), recursive=True)), k=10)
    random.seed(5)
    random_train_images += random.choices(sorted(glob(str(input_prefix / '168ora/**/*.jpg'), recursive=True)), k=10)
    random.seed(6)
    random_train_images += random.choices(sorted(glob(str(input_prefix / 'KiadokTinta/**/*.jpg'), recursive=True)), k=10)

    random_val_images = []
    random.seed(7)
    random_val_images += random.choices(sorted(glob(str(input_prefix / 'MagyarNemzet/**/*.jpg'), recursive=True)), k=60)
    random.seed(8)
    random_val_images += random.choices(sorted(glob(str(input_prefix / 'VilagIfjusaga/**/*.jpg'), recursive=True)), k=20)
    random.seed(9)
    random_val_images += random.choices(sorted(glob(str(input_prefix / 'KiadokKronosz/**/*.jpg'), recursive=True)), k=10)


    jobs = {'val': random_val_images,
            'train': random_train_images}

    WIDTH = 1024
    HEIGHT = 1024

    for key in jobs:
        i = 0
        for image_path in tqdm.tqdm(jobs[key]):
            image_path = str(image_path)
            image_tensor = read_image(image_path, HEIGHT, WIDTH)
            resized_image_array = np.array(resize_image_from_path(image_path, HEIGHT, WIDTH))

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
                img.save(f'/mnt/noah/dev/csilla/cv/articles/test_images_deeplab/{key}_{i}.jpg')
            i += 1
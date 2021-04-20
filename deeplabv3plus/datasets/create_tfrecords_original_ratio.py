import tensorflow as tf
from PIL import Image
from glob import glob
import tqdm
import random
import numpy as np
import io
from pathlib import Path
import create_tfrecords as record_utils

def resize_image(image_path, longer_side=1024, mode='RGB'):
    img_orig = Image.open(image_path, 'r')
    old_width, old_height = img_orig.size
    img_ratio = old_height / old_width
    if old_height > old_width:
        height = longer_side
        width = int(height / img_ratio)
    else:
        width = longer_side
        height = int(width * img_ratio)
    img = img_orig.resize((width, height), resample=Image.NEAREST)
    if mode == 'P':
        paletteSequence = img_orig.getpalette()
        img.putpalette(paletteSequence)
    return img

def create_tf_example(image_path, label_path, longer_side = 1024):
    # PIL Images
    resized_jpg = resize_image(image_path, longer_side, mode='RGB')
    resized_png = resize_image(label_path, longer_side, mode='P')
    resized_png_grey = Image.fromarray(np.array(resized_png), 'L')
    encoded_jpg = record_utils.encode_image(resized_jpg, format='JPEG')
    encoded_png = record_utils.encode_image(resized_png_grey, format='PNG')
    width, height = resized_jpg.size

    feature_dict = {
        'image/filename':
            record_utils.bytes_feature(image_path.encode('utf8')),
        'image/encoded':
            record_utils.bytes_feature(encoded_jpg),
        'image/height':
            record_utils.int64_feature(height),
        'image/width':
            record_utils.int64_feature(width),
        'label/filename':
            record_utils.bytes_feature(label_path.encode('utf8')),
        'label/encoded':
            record_utils.bytes_feature(encoded_png)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))

if __name__ == '__main__':
    input_prefix = Path('/mnt/noah/dev/csilla/cv/articles/deeplab/article_new')
    # tfrec_prefix = Path('/mnt/noah/dev/csilla/deeplab/tfrecords/')
    tfrec_prefix = Path('/mnt/noah/dev/csilla/cv/articles/deeplab/test_records')
    for mode in ['train', 'val']:
        # if mode == 'train':
        #     images = sorted([img for img in input_prefix.glob('**/*.jpg') if not str(img.relative_to(input_prefix)).startswith(
        #             ('MagyarNemzet', 'VilagIfjusaga', 'KiadokKronosz'))])
        #     labels = sorted([img for img in input_prefix.glob('**/*.png') if not str(img.relative_to(input_prefix)).startswith(
        #             ('MagyarNemzet', 'VilagIfjusaga', 'KiadokKronosz'))])
        # else:
        #     images = sorted(glob(str(input_prefix/'MagyarNemzet/**/*.jpg'), recursive=True) +
        #                     glob(str(input_prefix/'KiadokKronosz/**/*.jpg'), recursive=True) +
        #                     glob(str(input_prefix/'VilagIfjusaga/**/*.jpg'), recursive=True))
        #     labels = sorted(glob(str(input_prefix/'MagyarNemzet/**/*.png'), recursive=True) +
        #                     glob(str(input_prefix/'KiadokKronosz/**/*.png'), recursive=True) +
        #                     glob(str(input_prefix/'VilagIfjusaga/**/*.png'), recursive=True))
        if mode == 'train':
            images = sorted(input_prefix.glob('VilagIfjusaga/VilagIfjusaga_1976/VilagIfjusaga_1976_01_01/*.jpg'))
            labels = sorted(input_prefix.glob('VilagIfjusaga/VilagIfjusaga_1976/VilagIfjusaga_1976_01_01/*.png'))
        else:
            images = sorted(input_prefix.glob('VasarnapiUjsag/VasarnapiUjsag_1921/VasarnapiUjsag_1921_szam_06/*.jpg'))
            labels = sorted(input_prefix.glob('VasarnapiUjsag/VasarnapiUjsag_1921/VasarnapiUjsag_1921_szam_06/*.png'))
        assert len(images) == len(labels)
        print(f'Number of {mode} images: ', len(images))
        example_cache = []
        example_idx = 0
        record_idx = 0
        for image_path, label_path in tqdm.tqdm(zip(images, labels), total=len(images)):
            image_path = str(image_path)
            label_path = str(label_path)
            example_cache.append(
                create_tf_example(image_path, label_path)
            )
            example_idx += 1
            if len(example_cache) >= 1000:
                file_num = '{:d}'.format(record_idx).zfill(4)
                record_utils.write_examples_to_tfrecord(example_cache, str(tfrec_prefix / f'deeplab_{mode}_{file_num}.tfrecord'))
                record_idx += 1
                example_cache = []

        if len(example_cache) > 0:
            file_num = '{:d}'.format(record_idx).zfill(4)
            record_utils.write_examples_to_tfrecord(example_cache, str(tfrec_prefix / f'deeplab_{mode}_{file_num}.tfrecord'))
        print(f"Total number of {mode} examples: ", example_idx)

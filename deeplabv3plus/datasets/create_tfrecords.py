import tensorflow as tf
from PIL import Image
from glob import glob
import tqdm
import random
import numpy as np
import io
from pathlib import Path

def write_examples_to_tfrecord(examples, record_path):
    random.shuffle(examples)
    with tf.io.TFRecordWriter(record_path) as writer:
        for example in examples:
            writer.write(example.SerializeToString())

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def resize_image(image_path, new_height, new_width, mode='RGB'):
    img_orig = Image.open(image_path, 'r')
    old_width, old_height = img_orig.size
    img_ratio = old_height / old_width
    if img_ratio >= new_height / new_width:
        height = new_height
        width = int(height / img_ratio)
    else:
        width = new_width
        height = int(new_width * img_ratio)
    img = img_orig.resize((width, height), resample=Image.NEAREST)
    new_img = Image.new(mode, (new_width, new_height))
    new_img.paste(img, ((new_width - width) // 2, (new_height - height) // 2))
    if mode == 'P':
        paletteSequence = img_orig.getpalette()
        new_img.putpalette(paletteSequence)
    return new_img

def encode_image(pil_image, format=None):
    buf = io.BytesIO()
    pil_image.save(buf, format=format)
    return buf.getvalue()

def create_tf_example(image_path, label_path, width=768, height=768):
    HEIGHT = height
    WIDTH = width
    # PIL Images
    resized_jpg = resize_image(image_path, HEIGHT, WIDTH, mode='RGB')
    resized_png = resize_image(label_path, HEIGHT, WIDTH, mode='P')
    resized_png_grey = Image.fromarray(np.array(resized_png), 'L')
    encoded_jpg = encode_image(resized_jpg, format='JPEG')
    encoded_png = encode_image(resized_png_grey, format='PNG')

    feature_dict = {
        'image/filename':
            bytes_feature(image_path.encode('utf8')),
        'image/encoded':
            bytes_feature(encoded_jpg),
        'image/height':
            int64_feature(HEIGHT),
        'image/width':
            int64_feature(WIDTH),
        'label/filename':
            bytes_feature(label_path.encode('utf8')),
        'label/encoded':
            bytes_feature(encoded_png)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))

if __name__ == '__main__':
    input_prefix = Path('/mnt/noah/dev/csilla/deeplab/article_new')
    tfrec_prefix = Path('/mnt/noah/dev/csilla/deeplab/tfrecords/')

    for mode in ['train', 'val']:
        if mode == 'train':
            images = sorted([img for img in input_prefix.glob('**/*.jpg') if not str(img.relative_to(input_prefix)).startswith(
                    ('MagyarNemzet', 'VilagIfjusaga', 'KiadokKronosz'))])
            labels = sorted([img for img in input_prefix.glob('**/*.png') if not str(img.relative_to(input_prefix)).startswith(
                    ('MagyarNemzet', 'VilagIfjusaga', 'KiadokKronosz'))])
        else:
            images = sorted(glob(str(input_prefix/'MagyarNemzet/**/*.jpg'), recursive=True) +
                            glob(str(input_prefix/'KiadokKronosz/**/*.jpg'), recursive=True) +
                            glob(str(input_prefix/'VilagIfjusaga/**/*.jpg'), recursive=True))
            labels = sorted(glob(str(input_prefix/'MagyarNemzet/**/*.png'), recursive=True) +
                            glob(str(input_prefix/'KiadokKronosz/**/*.png'), recursive=True) +
                            glob(str(input_prefix/'VilagIfjusaga/**/*.png'), recursive=True))
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
                write_examples_to_tfrecord(example_cache, str(tfrec_prefix / f'deeplab_{mode}_{file_num}.tfrecord'))
                record_idx += 1
                example_cache = []

        if len(example_cache) > 0:
            file_num = '{:d}'.format(record_idx).zfill(4)
            write_examples_to_tfrecord(example_cache, str(tfrec_prefix / f'deeplab_{mode}_{file_num}.tfrecord'))
        print(f"Total number of {mode} examples: ", example_idx)
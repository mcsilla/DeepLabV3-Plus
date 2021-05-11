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

def get_bbox(binary_mask):
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    xmin, xmax = np.where(rows)[0][[0, -1]]
    ymin, ymax = np.where(cols)[0][[0, -1]]

    return xmin, xmax, ymin, ymax

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# image: path or numpy array
def resize_image_from_path(image, new_height, new_width, mode='RGB'):
    try:
        img_orig = Image.open(image, 'r')
    except:
        img_orig = Image.fromarray(image)
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

def resize_image(image, longer_side=1024, mode='RGB'):
    img_orig = image # = Image.open(image_path, 'r')
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


def create_segmentation_mask(articles_in_img, height, width, type2index, label_palette):
    mask = np.zeros((height, width))
    blocks = itertools.chain(*[article['coords'].items() for article in articles_in_img])
    ordered_blocks = sorted(blocks, key=lambda item: type2index[item[1]])
    for block_coords_str, block_type in ordered_blocks:
        block_coords = list(map(int, block_coords_str.split()))
        if not utils.coords_valid(image, block_coords):
            return False
        left, top, right, bottom = utils.normalize_coords(block_coords)
        mask[top:bottom, left:right] = type2index[block_type]

    label = Image.new('P', image.size)
    label.putpalette(label_palette)
    label.putdata(mask.flatten())
    return label


def create_article_annotation(articles_in_img, height, width):
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    is_crowd = []
    category_names = []
    category_ids = []
    areas = []
    binary_masks = []
    for article in articles_in_img:
        mask = np.zeros((height, width), dtype=np.uint8)
        is_crowd.append(0)
        category_names.append(article['topic'])
        category_ids.append(article_type2index[article['topic']])
        for block_coords_str in article['coords']:
            block_coords = list(map(int, block_coords_str.split()))
            if not utils.coords_valid(image, block_coords):
                continue
            left, top, right, bottom = utils.normalize_coords(block_coords)
            mask[top:bottom, left:right] = 1
        binary_masks.append(mask)
        positive_pixel_count = mask.sum()
        area = positive_pixel_count / (width * height)
        areas.append(area)
        xmn, xmx, ymn, ymx = get_bbox(mask)
        xmin.append(float(xmn) / width)
        xmax.append(float(xmx) / width)
        ymin.append(float(ymn) / height)
        ymax.append(float(ymx) / height)
    return (
        category_ids,
        category_names,
        is_crowd,
        binary_masks,
        areas,
        xmin, xmax, ymin, ymax,
    )

def encode_image(pil_image, format=None):
    buf = io.BytesIO()
    pil_image.save(buf, format=format)
    return buf.getvalue()

def create_tf_example(image_path, label, source_id, mrcnn_annotations, longer_side = 1024):
    # PIL Images
    with Image.open(image_path, 'r') as pil_img:
        resized_jpg = resize_image(pil_img, longer_side, mode='RGB')
    resized_png = resize_image(label, longer_side, mode='P')
    resized_png_gray = Image.fromarray(np.array(resized_png), 'L')
    encoded_jpg = record_utils.encode_image(resized_jpg, format='JPEG')
    encoded_png = record_utils.encode_image(resized_png_gray, format='PNG')
    width, height = resized_jpg.size
    key = hashlib.sha256(encoded_jpg).hexdigest()
    img_cv2 = cv2.imread(image_path,0)
    # img_cv2 = cv2.medianBlur(img,5)
    # img_cv2 = cv2.GaussianBlur(img, (7, 7), 1)

    img_black = cv2.adaptiveThreshold(img_cv2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,49,11)

    encoded_black = record_utils.encode_image(resize_image(Image.fromarray(img_black), longer_side, mode='RGB'), format='JPEG')
    feature_dict = {

        'label/encoded':
            bytes_feature(encoded_png),
        'image/height':
            int64_feature(height),
        'image/width':
            int64_feature(width),
        'image/filename':
            bytes_feature(image_path.encode('utf8')),
        'image/source_id':
            bytes_feature(str(source_id).encode('utf8')),
        'image/key/sha256':
            bytes_feature(key.encode('utf8')),
        'image/encoded':
            bytes_feature(encoded_jpg),
        'image_black/encoded':
            bytes_feature(encoded_black),
        'image/format':
            bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin':
            float_list_feature(mrcnn_annotations['xmin']),
        'image/object/bbox/xmax':
            float_list_feature(mrcnn_annotations['xmax']),
        'image/object/bbox/ymin':
            float_list_feature(mrcnn_annotations['ymin']),
        'image/object/bbox/ymax':
            float_list_feature(mrcnn_annotations['ymax']),
        'image/object/class/text':
            bytes_list_feature(mrcnn_annotations['category_names']),
        'image/object/is_crowd':
            int64_list_feature(mrcnn_annotations['is_crowd']),
        'image/object/area':
            float_list_feature(mrcnn_annotations['areas']),
        'image/object/mask':
            bytes_list_feature(mrcnn_annotations['encoded_mask_png']),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))

if __name__ == '__main__':
    input_prefix = Path('/mnt/noah/dev/csilla/deeplab/article_new')
    # tfrec_prefix = Path('/mnt/noah/dev/csilla/deeplab/tfrecords/')
    tfrec_prefix = Path('/mnt/noah/dev/csilla/deeplab/testrecords/')
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
                write_examples_to_tfrecord(example_cache, str(tfrec_prefix / f'deeplab_{mode}_{file_num}.tfrecord'))
                record_idx += 1
                example_cache = []

        if len(example_cache) > 0:
            file_num = '{:d}'.format(record_idx).zfill(4)
            write_examples_to_tfrecord(example_cache, str(tfrec_prefix / f'deeplab_{mode}_{file_num}.tfrecord'))
        print(f"Total number of {mode} examples: ", example_idx)
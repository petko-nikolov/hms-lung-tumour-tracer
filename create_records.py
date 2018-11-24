import tensorflow as tf
import argparse
import os
import utils
from multiprocessing import Pool, cpu_count
import uuid
import skimage.transform
import numpy as np


tf_options = tf.python_io.TFRecordOptions(
    compression_type=tf.python_io.TFRecordCompressionType.ZLIB)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_input_image(input_dir, scan_id, slice_id, resize_shape, context=0):
    images = utils.get_image_range(
        input_dir, scan_id, int(slice_id) - context, int(slice_id) + context, (512, 512))
    transformed_images = []
    for image in images:
        image = skimage.transform.resize(image, resize_shape)
        transformed_images.append(image)
    input_image = np.stack(transformed_images, axis=-1)
    return input_image


def create_example(line):
    data = utils.parse_line(line.strip())

    input_image = create_input_image(
        args.input_dir, data['scan_id'], data['slice_id'], utils.INPUT_SHAPE)

    # normalize the image
    input_image = input_image / 0.0625

    if args.structure and args.structure not in data['structures']:
        return None
    mask = utils.create_joint_mask((512, 512), data['structures'], utils.LUNG_CLASSES)
    height, width = utils.INPUT_SHAPE

    classes = np.array([0] +
                       [int(any(x in data['structures'] for x in v))
                        for k, v in utils.LUNG_CLASSES], dtype=np.int32)

    print("Writing: ", data['structures'].keys(), "Classes:", classes,
          "Image max:", np.max(input_image),
          "Mask max: ", np.max(mask))

    example = tf.train.Example(features=tf.train.Features(
        feature={'height': _int64_feature(height),
                 'width': _int64_feature(width),
                 'image': _bytes_feature(input_image.tostring()),
                 'mask': _bytes_feature(mask.tostring()),
                 'scan_id': _bytes_feature(data['scan_id'].encode()),
                 'slice_id': _bytes_feature(data['slice_id'].encode()),
                 'classes': _bytes_feature(classes.tostring())}))
    return example


def create_lung_example(line):
    data = utils.parse_line(line.strip())

    input_image = create_input_image(
        args.input_dir, data['scan_id'], data['slice_id'], (256, 256), 3)

    # normalize the image
    input_image = input_image / 0.0625

    if args.structure and args.structure not in data['structures']:
        return None

    with open(os.path.join(args.input_dir, data['scan_id'], 'coordinates.txt'), 'r') as f:
        r0, c0, r1, c1 = [int(x) for x in f.read().split(",")]

    radiomics = {k: v for k, v in data['structures'].items() if k == 'radiomics_gtv'}

    mask = utils.create_mask((512, 512), radiomics)
    mask = mask[r0:r1, c0:c1]
    height, width = 256, 256

    classes = np.array([0] +
                       ['radiomics_gtv' in data['structures']], dtype=np.int32)

    print("Writing: ", data['structures'].keys(), "Classes:", classes,
          "Image max:", np.max(input_image),
          "Mask max: ", np.max(mask))

    example = tf.train.Example(features=tf.train.Features(
        feature={'height': _int64_feature(height),
                 'width': _int64_feature(width),
                 'image': _bytes_feature(input_image.tostring()),
                 'mask': _bytes_feature(mask.tostring()),
                 'scan_id': _bytes_feature(data['scan_id'].encode()),
                 'slice_id': _bytes_feature(data['slice_id'].encode()),
                 'classes': _bytes_feature(classes.tostring())}))
    return example


def create_seed_example(line):
    data = utils.parse_line(line.strip())

    input_image = create_input_image(
        args.input_dir, data['scan_id'], data['slice_id'], (512, 512), 1)

    # normalize the image
    input_image = input_image / 0.0625

    if args.structure and args.structure not in data['structures']:
        return None

    radiomics = {k: v for k, v in data['structures'].items() if k == 'radiomics_gtv'}

    examples = []

    for seed in data['seeds']:
        mask = utils.create_mask((512, 512), radiomics)
        height, width = 224, 224
        r0 = max(0, seed[1] - height // 2)
        c0 = max(0, seed[0] - width // 2)
        r1 = seed[1] + height // 2
        c1 = seed[0] + width // 2

        mask = mask[r0:r1, c0:c1]
        image = input_image[r0:r1, c0:c1, :]

        pad_h = (height - image.shape[0]) / 2
        pad_w = (width - image.shape[1]) / 2

        print("Point", seed, "Image", image.shape, "Mask", mask.shape, "Pad", pad_h, pad_w)

        if pad_h or pad_w:
            mask = np.pad(mask, ((int(np.ceil(pad_h)), int(np.floor(pad_h))),
                                 (int(np.ceil(pad_w)), int(np.floor(pad_w)))),
                          mode='constant')

            image = np.pad(image, ((int(np.ceil(pad_h)), int(np.floor(pad_h))),
                                   (int(np.ceil(pad_w)), int(np.floor(pad_w))),
                                   (0, 0)),
                           mode='constant')

        assert(image.shape[0] == height and image.shape[1] == width)
        assert(mask.shape[0] == height and mask.shape[1] == width)

        classes = np.array([0] +
                           ['radiomics_gtv' in data['structures']], dtype=np.int32)

        print("Writing: ", data['structures'].keys(), "Classes:", classes,
              "Image max:", np.max(image),
              "Mask max: ", np.max(mask))

        example = tf.train.Example(features=tf.train.Features(
            feature={'height': _int64_feature(height),
                     'width': _int64_feature(width),
                     'image': _bytes_feature(image.tostring()),
                     'mask': _bytes_feature(mask.tostring()),
                     'scan_id': _bytes_feature(data['scan_id'].encode()),
                     'slice_id': _bytes_feature(data['slice_id'].encode()),
                     'classes': _bytes_feature(classes.tostring())}))

        examples.append(example)
    return examples


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def process_lines(lines):
    writer = tf.python_io.TFRecordWriter(
        os.path.join(args.output_directory, 'data-{}'.format(uuid.uuid4())), tf_options)
    for line in lines:
        if args.type == 'normal':
            examples = create_example(line)
        elif args.type == 'lungs':
            examples = create_lung_example(line)
        elif args.type == 'seeds':
            examples = create_seed_example(line)
        for example in examples:
            writer.write(example.SerializeToString())
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert to dataset.')
    parser.add_argument(
        '--input_dir', type=str, required=True,
        help='Input directory.')
    parser.add_argument(
        '--input_annotations', type=str, required=True,
        help='Input annotations file.')
    parser.add_argument(
        '--output_directory', type=str, required=True,
        help='Output directory.')
    parser.add_argument(
        '--structure', type=str, required=False, default=None,
        help='Structure type.')
    parser.add_argument(
        '--type', choices=['lungs', 'seeds', 'normal'], type=str, required=True,
        help='Generate records from type.')
    args = parser.parse_args()

    os.makedirs(args.output_directory)

    p = Pool(processes=cpu_count())
    with open(args.input_annotations, 'r') as f:
        data = list(f.readlines())
        p.map(process_lines, chunks(data, 512))
        p.close()

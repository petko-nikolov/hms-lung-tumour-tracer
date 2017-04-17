import tensorflow as tf
import argparse
import glob
import os
from model import (
    ConvolutionalSegmentationFirst256, ConvolutionalSegmentationModel,
    VGGModel, ResnetV2Segmentation, ConvolutionalSegmentationModel3D)
import utils
import numpy as np
import pickle
from copy import deepcopy
import pprint


STRUCTURE_CLASS = utils.STRUCTURE_CLASS


def update_segmentation_metrics(current_metrics, metrics):
    for k, v in metrics['losses'].items():
        metrics['losses'][k] = v + current_metrics['losses'][k]

    segmentation = metrics['accuracy']['segmentation']
    current_segmentation = current_metrics['accuracy']['segmentation']
    for k, v in segmentation.items():
        segmentation[k] = (v[0] + current_segmentation[k][0], v[1] + current_segmentation[k][1])

    return metrics


def normalize_metrics(metrics, example_count):
    metrics = deepcopy(metrics)

    for k, v in metrics['losses'].items():
        metrics['losses'][k] = v / example_count

    segmentation = metrics['accuracy']['segmentation']
    for k, v in segmentation.items():
        segmentation[k] = v[0] / v[1] if v[1] > 0 else float("-inf")

    return metrics


def init_metrics(classes):
    metrics = {}

    metrics['losses'] = {'segmentation': 0.0}

    metrics['accuracy'] = {
        'segmentation': {k: (0.0, 0) for k, v in classes}}

    return metrics


def compute_jacard(preds, mask, classes):
    result = {}
    for i, (k, v) in enumerate(classes):
        positive_preds = (preds == (i + 1))
        positive_gt = (mask == (i + 1))
        total_pixels = np.sum(
            np.logical_or(positive_preds, np.squeeze(positive_gt)),
            axis=(1, 2))
        matched_pixels = np.sum(
            np.logical_and(positive_preds, np.squeeze(positive_gt)),
            axis=(1, 2))
        result[k] = (np.sum(matched_pixels), np.sum(total_pixels))
    return result


class RadiomicsData(object):
    def get_tensors(self, serialized_example):
        feature_map = {
          'height': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
          'width': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
          'image': tf.VarLenFeature(dtype=tf.string),
          'mask': tf.VarLenFeature(dtype=tf.string),
          'scan_id': tf.VarLenFeature(dtype=tf.string),
          'slice_id': tf.VarLenFeature(dtype=tf.string),
          'xs': tf.VarLenFeature(dtype=tf.string),
          'ys': tf.VarLenFeature(dtype=tf.string)}
        features = tf.parse_single_example(serialized_example, feature_map)
        example = {
          'height': tf.to_int32(features['height']),
          'width': tf.to_int32(features['width'])}

        input_shape = (256, 256) + (7,)
        example['image'] = tf.decode_raw(features['image'].values[0], tf.float64)
        example['image'] = tf.reshape(example['image'], input_shape)
        example['image'] = tf.cast(example['image'], tf.float32)
        example['image'] = tf.slice(example['image'], [0, 0, 2], [-1, -1, 3])

        example['mask'] = tf.decode_raw(features['mask'].values[0], tf.float64)
        example['mask'] = tf.reshape(example['mask'], (256, 256, 1))
        example['mask'] = tf.cast(example['mask'], tf.float32)

        example['scan_id'] = features['scan_id']
        example['slice_id'] = features['slice_id']
        example['classes'] = tf.decode_raw(features['classes'].values, tf.int32)

        example['classes'] = tf.reshape(example['classes'], [2])

        mean_image = tf.reduce_mean(example['image'], axis=[0, 1, 2])
        example['image'] = example['image'] - tf.reshape(mean_image, [1, 1, 1])
        print(example['image'])

        return example


class RadiomicsData3D(object):
    def get_tensors(self, serialized_example):
        feature_map = {
          'height': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
          'width': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
          'image': tf.VarLenFeature(dtype=tf.string),
          'mask': tf.VarLenFeature(dtype=tf.string),
          'scan_id': tf.VarLenFeature(dtype=tf.string),
          'slice_id': tf.VarLenFeature(dtype=tf.string),
          'xs': tf.VarLenFeature(dtype=tf.string),
          'ys': tf.VarLenFeature(dtype=tf.string)}
        features = tf.parse_single_example(serialized_example, feature_map)
        example = {
          'height': tf.to_int32(features['height']),
          'width': tf.to_int32(features['width'])}

        input_shape = (256, 256) + (7,)
        example['image'] = tf.decode_raw(features['image'].values[0], tf.float64)
        example['image'] = tf.reshape(example['image'], input_shape)
        example['image'] = tf.cast(example['image'], tf.float32)

        example['mask'] = tf.decode_raw(features['mask'].values[0], tf.float64)
        example['mask'] = tf.reshape(example['mask'], (256, 256, 1))
        example['mask'] = tf.cast(example['mask'], tf.float32)

        example['scan_id'] = features['scan_id']
        example['slice_id'] = features['slice_id']

        return example


class LungData(object):
    def get_tensors(self, serialized_example):
        feature_map = {
          'height': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
          'width': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
          'image': tf.VarLenFeature(dtype=tf.string),
          'mask': tf.VarLenFeature(dtype=tf.string),
          'classes': tf.VarLenFeature(dtype=tf.string),
          'scan_id': tf.VarLenFeature(dtype=tf.string),
          'slice_id': tf.VarLenFeature(dtype=tf.string),
          'xs': tf.VarLenFeature(dtype=tf.string),
          'ys': tf.VarLenFeature(dtype=tf.string)}
        features = tf.parse_single_example(serialized_example, feature_map)
        example = {
          'height': tf.to_int32(features['height']),
          'width': tf.to_int32(features['width'])}

        input_shape = (512, 512) + (1,)
        example['image'] = tf.decode_raw(features['image'].values[0], tf.float64)
        example['image'] = tf.reshape(example['image'], input_shape)
        example['image'] = tf.cast(example['image'], tf.float32)

        example['mask'] = tf.decode_raw(features['mask'].values[0], tf.float64)
        example['mask'] = tf.reshape(example['mask'], (512, 512, 1))
        example['mask'] = tf.cast(example['mask'], tf.float32)

        example['scan_id'] = features['scan_id']
        example['slice_id'] = features['slice_id']
        example['classes'] = tf.decode_raw(features['classes'].values, tf.int32)

        example['classes'] = tf.reshape(example['classes'], [2])

        return example


def forward(data_dir, model, data_reader, is_training, batch_size):
    with tf.Graph().as_default():
        filename_queue = tf.train.string_input_producer(
            glob.glob(data_dir + '*'), shuffle=is_training, num_epochs=num_epochs)

        options = tf.python_io.TFRecordOptions(
            compression_type=tf.python_io.TFRecordCompressionType.ZLIB)

        reader = tf.TFRecordReader(options=options)
        _, serialized_example = reader.read(filename_queue)

        example = data_reader.get_tensors(serialized_example)

        images, masks, scan_ids, slice_ids = tf.train.shuffle_batch(
            [example['image'], example['mask'], example['scan_id'],
             example['slice_id']],
            batch_size=batch_size,
            capacity=1000,
            num_threads=2,
            allow_smaller_final_batch=True,
            min_after_dequeue=10)

        slice_images = tf.unstack(images, axis=3)
        for i, slice_image in enumerate(slice_images):
            tf.summary.image("images_{}".format(i),
                             tf.expand_dims(slice_image, axis=-1))

        tf.summary.image("masks", masks)

        logits = model.forward(images, is_training)
        return {'logits': logits,
                'masks': masks,
                'images': images,
                'scan_ids': scan_ids,
                'slice_ids': slice_ids}


def add_image_summary(classes, probabilities, predictions, masks):
    for i, (k, v) in enumerate(classes):
        cls_prob = tf.slice(probabilities, [0, 0, 0, i + 1], [-1, -1, -1, 1])
        cls_pred = tf.to_float(tf.equal(predictions, i + 1))
        target = tf.to_float(tf.equal(masks, i + 1))

        tf.summary.image('{}_probs'.format(k),
                         tf.to_float(cls_prob))
        tf.summary.image('{}_preds'.format(k),
                         tf.expand_dims(tf.to_float(cls_pred), axis=3))
        tf.summary.image('{}_target'.format(k), target)


def get_train_op(loss):
    optimizer = tf.train.AdamOptimizer(1e-4)
    grads_and_vars = optimizer.compute_gradients(loss, tf.trainable_variables())
    grads, vars = zip(*grads_and_vars)
    clipped_grads, grads_global_norm = tf.clip_by_global_norm(grads, 1.0)
    train_op = optimizer.apply_gradients(zip(clipped_grads, vars))
    return train_op


def restore_checkpoint(checkpoint, ignore_missing_vars):
    vars_to_restore = tf.all_variables()
    if ignore_missing_vars:
        reader = tf.train.NewCheckpointReader(checkpoint)
        var_to_shape_map = reader.get_variable_to_shape_map()
        vars_to_restore = [v for v in vars_to_restore
                           if v.op.name in var_to_shape_map]
    print("Vars to restore:", [v.op.name for v in vars_to_restore])


def train(data_dir, model, data_reader, is_training, sampling, batch_size, checkpoint,
          ignore_missing_vars, model_dir, validate_output_dir, max_iters):
        loss = tf.constant(0)

        tensors = forward(data_dir, model, data_reader, is_training, batch_size)

        probabilities = tf.nn.softmax(tensors['logits'])

        loss = model.error(tensors['logits'], tensors['masks'])

        segmentation_tensors = {
            'probabilities': probabilities,
            'predictions': tf.argmax(probabilities, axis=3)}

        add_image_summary(model.classes, probabilities,
                          segmentation_tensors['predictions'],
                          tensors['masks'])

        train_op = get_train_op(loss)

        init_op = tf.global_variables_initializer()
        local_init_op = tf.local_variables_initializer()

        summary_op = tf.summary.merge_all()

        session = tf.Session()
        session.run([init_op, local_init_op])


        saver = tf.train.Saver(tf.all_variables(), keep_checkpoint_every_n_hours=1, max_to_keep=30)

        if checkpoint:
            restore_checkpoint(checkpoint, ignore_missing_vars)

        global_step_tensor = tf.train.get_global_step()

        summary_writer = tf.summary.FileWriter(model_dir)

        if checkpoint:
            print("Restoring..")
            tf.train.Saver(vars_to_restore).restore(session, checkpoint)

        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=session, coord=coord)

        global_example_counter = 0
        global_metrics = init_metrics(model.classes)

        example_counter = 0
        metrics = init_metrics(model.classes)

        if validate_output_dir:
            os.makedirs(validate_output_dir)

        for i in range(max_iters):
            fetches = [loss, summary_op, segmentation_tensors, tensors['masks']]
            if not validate_mode:
                fetches.append(train_op)
            else:
                fetches.extend([tensors['scan_ids'], tensors['slice_ids']])

            try:
                if not validate_mode:
                    loss, summary, pred_data, gt, gt_classes, _, = session.run(fetches)
                else:
                    loss, summary, pred_data, gt, gt_classes, scans, slices = session.run(fetches)

                    predictions_objects = utils.split_batch(
                        scans, slices, pred_data['predictions'], pred_data['probabilities'])

                    for p in predictions_objects:
                        with open(
                                os.path.join(validate_output_dir,
                                             p.scan_id + '.' + p.slice_id), 'wb') as f:
                            pickle.dump(p.predictions.astype(np.uint8), f)

                current_metrics = {'loss': loss}
                current_metrics['accuracy'] = {
                    'segmentation': compute_jacard(pred_data['predictions'], gt,
                                                   model.classes)}

                metrics = update_segmentation_metrics(current_metrics, metrics)
                global_metrics = update_segmentation_metrics(current_metrics, global_metrics)

                example_counter += pred_data['predictions'].shape[0]
                global_example_counter += pred_data['predictions'].shape[0]

                if(i % 20 == 0):
                    print("Step:", i, ",", "Metrics:")
                    pprint.pprint(normalize_metrics(metrics, example_counter))
                    metrics = init_metrics(model.classes)
                    example_counter = 0
                    summary_writer.add_summary(summary)

                if i > 0 and i % 1000 == 0 and not validate_mode:
                    saver.save(session, os.path.join(model_dir, 'model'), i)
            except tf.errors.OutOfRangeError:
                break

        print("Final metrics:", normalize_metrics(global_metrics, global_example_counter))

        session.close()


data_readers = {'lung': LungData,
                'radiomics': RadiomicsData,
                'radiomics_3d': RadiomicsData3D,
                'radiomics_first': RadiomicsData,
                'radiomics_resnet': RadiomicsData}

models_map = {'lung': ConvolutionalSegmentationModel,
              'radiomics': VGGModel,
              'radiomics_3d': ConvolutionalSegmentationModel3D,
              'radiomics_first': ConvolutionalSegmentationFirst256,
              'radiomics_resnet': ResnetV2Segmentation}

if '__main__' == __name__:
    parser = argparse.ArgumentParser(description='Convert to dataset.')
    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='Train tf records prefix.')
    parser.add_argument(
        '--model_dir', type=str, required=True,
        help='Model directory.')
    parser.add_argument(
        '--batch_size', type=int, required=False, default=4,
        help='Batch size.')
    parser.add_argument(
        '--max_iters', type=int, required=False, default=100000,
        help='Max model iterations.')
    parser.add_argument(
        '--checkpoint', type=str, required=False, default=None,
        help='Checkpoint to restore.')
    parser.add_argument(
        '--ignore_missing_vars', action='store_true', required=False, default=None,
        help='Ignore missing vars')
    parser.add_argument(
        '--sampling', type=int, required=False, default=None,
        help='Sample with dropout ')
    parser.add_argument(
        '--model_variation', type=str, required=False, default='lung',
        choices=list(models_map.keys()),
        help='Model type to train (classification or regression).')
    parser.add_argument(
        '--validate_output_dir', type=str, required=False, default=None,
        help='Validata output directory.')

    args = parser.parse_args()

    validate_mode = args.validate_output_dir is not None
    is_training = args.validate_output_dir is None
    is_training = is_training or args.sampling is not None

    num_epochs = 1 if validate_mode else None
    shuffle = (not validate_mode)

    train(args.data_dir, models_map[args.model_variation](),
          data_readers[args.model_variation](), not validate_mode,
          args.sampling, args.batch_size, args.checkpoint,
          args.ignore_missing_vars, args.model_dir, args.validate_output_dir, args.max_iters)

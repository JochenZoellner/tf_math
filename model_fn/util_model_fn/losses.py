from itertools import permutations

import numpy as np
import tensorflow as tf

from input_fn.input_fn_2d.data_gen_2dt.util_2d.misc_tf import center_triangle, flip_along_axis
from model_fn.util_model_fn import keras_compatible_layers


def point3_loss(target, prediction):
    """target and prediction are tensors with shape (3,2) holding coordinates of 3 points of a triangle
        -find best permutation of prediction against target points in terms of linear-summed-point-distance
    :return scalar loss tensor"""
    loss_list = []
    target = tf.squeeze(target, axis=0)
    prediction = tf.squeeze(prediction, axis=0)
    for permutation in set(permutations([0, 1, 2])):
        permuted_prediction = tf.gather_nd(prediction, np.expand_dims(np.array(permutation), axis=1))
        loss_by_chance = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(permuted_prediction - target), axis=1)))
        loss_list.append(loss_by_chance)

    return tf.reduce_min(tf.stack(loss_list))


def map_point3_loss(in_tuple):
    """target and prediction are tensors with shape (3,2) holding coordinates of 3 points of a triangle
        -find best permutation of prediction against target points in terms of linear-summed-point-distance
    :return scalar loss tensor"""
    target, prediction = in_tuple
    loss_list = []
    target = tf.cast(tf.reshape(target, [3, 2]), tf.float32)
    prediction = tf.reshape(prediction, [3, 2])
    for permutation in set(permutations([0, 1, 2])):
        permuted_prediction = tf.gather_nd(prediction, np.expand_dims(np.array(permutation), axis=1))
        loss_by_chance = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(permuted_prediction - target), axis=1)))
        loss_list.append(loss_by_chance)

    return tf.reduce_min(tf.stack(loss_list))


def map_point3_loss_absolute_invariant(in_tuple):
    """calc best point loss with centered triangle and mirrored at x axis, later leads to 12 permutations"""
    target, prediction = in_tuple
    tgt_points = tf.reshape(target, [3, 2])
    pred_points = tf.reshape(prediction, [3, 2])
    tgt_points = center_triangle(tgt_points)
    pred_points = center_triangle(pred_points)

    pred_points_x_flip = flip_along_axis(pred_points, axis="x")
    no_flip_res = map_point3_loss((tf.keras.backend.flatten(pred_points), tf.keras.backend.flatten(tgt_points)))
    flip_res = map_point3_loss((tf.keras.backend.flatten(pred_points_x_flip), tf.keras.backend.flatten(tgt_points)))
    return tf.minimum(no_flip_res, flip_res)






def no_match_loss(target, prediction, batch_size):
    target_lined = tf.reshape(target, [batch_size, 6])
    prediction_lined = tf.reshape(prediction, [batch_size, 6])
    return tf.reduce_sum(tf.losses.absolute_difference(target_lined, prediction_lined))


def map_ordered_point3_loss(in_tupel):
    target, prediction = in_tupel
    target = tf.cast(tf.reshape(target, [3, 2]), tf.float32)
    prediction = keras_compatible_layers.reshape(prediction, [3, 2])
    return tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(prediction - target), axis=1)))


def batch_point3_loss(target, prediction, batch_size):
    target_lined = tf.reshape(target, [batch_size, 6])
    # target_lined = tf.Print(target_lined, [tf.shape(target_lined)])
    prediction_lined = tf.reshape(prediction, [batch_size, 6])

    sample_losses = tf.map_fn(map_point3_loss, (target_lined, prediction_lined), dtype=tf.float32)
    return sample_losses


def ordered_point3_loss(target, prediction, batch_size):
    target_lined = keras_compatible_layers.reshape(target, [batch_size, 6])
    # target_lined = tf.Print(target_lined, [tf.shape(target_lined)])
    prediction_lined = keras_compatible_layers.reshape(prediction, [batch_size, 6], )

    sample_losses = tf.map_fn(map_ordered_point3_loss, (target_lined, prediction_lined), dtype=tf.float32)
    return sample_losses


def batch_point3_loss_absolute_invariant(target, prediction, batch_size):
    target_lined = keras_compatible_layers.reshape(target, [batch_size, 6])
    prediction_lined = keras_compatible_layers.reshape(prediction, [batch_size, 6], )

    sample_losses = tf.map_fn(map_point3_loss_absolute_invariant, (target_lined, prediction_lined), dtype=tf.float32)
    return sample_losses
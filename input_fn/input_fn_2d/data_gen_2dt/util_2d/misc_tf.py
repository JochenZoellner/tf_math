import logging

import numpy as np
import tensorflow as tf

logger = logging.getLogger("util_2d/misc_tf.py")

def center_triangle(points):
    d_xy = tf.reduce_sum(points, axis=-2) / 3.0
    return points - tf.expand_dims(d_xy, axis=-2)

def flip_along_axis(points, axis="x"):
    """flip along x axis, [(batch), points, xy] - [None, None, 2]"""
    if axis == "x":
        mask = tf.constant([-1.0, 1.0])
    elif axis == "y":
        mask = tf.constant([1.0, -1.0])
    else:
        raise AttributeError("only axis='x' or axis='y' is possible as flip axis")
    res = tf.math.multiply(points, mask)
    return res


def get_spin_batched(batched_point_squence, dtype=tf.float32):
    """[batch, point, coordinate]
        [None, None, 2]
    eg. 12 samples of triangle in 2D ->[12,3,2]
    -find point with max X, (
    :raises ValueError if 3 neighbouring points have the same max x-coordinate"""
    # find max x-coordinate
    batched_point_squence = tf.cast(batched_point_squence, dtype)
    max_x_arg = tf.argmax(batched_point_squence[:, :, 0], axis=1)
    # construct gather indices for 3 Points with max x-Point centered
    ranged = tf.range(max_x_arg.shape[0], dtype=tf.int64)
    max_x_minus = (max_x_arg - 1) % batched_point_squence.shape[1]
    max_x = (max_x_arg) % batched_point_squence.shape[1]
    max_x_plus = (max_x_arg + 1) % batched_point_squence.shape[1]
    indices = tf.stack((ranged, max_x), axis=1)
    indices_minus = tf.stack((ranged, max_x_minus), axis=1)
    indices_plus = tf.stack((ranged, max_x_plus), axis=1)

    # construct 3 Points with max x-Point centered
    # print("indices_minus",indices_minus)
    Pm = tf.gather_nd(params=batched_point_squence, indices=indices_minus)
    P = tf.gather_nd(params=batched_point_squence, indices=indices)
    Pp = tf.gather_nd(params=batched_point_squence, indices=indices_plus)

    # calc scalar product of 'Pm->P'-normal and 'P->Pp'
    cross = tf.constant([[0.0, -1.0], [1.0, 0.0]], dtype)
    PmP = P - Pm
    PPp = Pp - P
    PPm_cross = tf.matmul(PmP, cross)
    orientation = tf.einsum("...i,...i->...", PPm_cross, PPp)
    # check if orientation contains zero which means adjacent 3 points on max x-straight
    # assertion = tf.assert_greater(tf.abs(orientation), tf.constant(0.0, dtype), message="get orientation failed, probably 3 points on a straight")
    # with tf.control_dependencies([assertion]):
    return orientation


@tf.function
def make_spin_positive(batched_point_squence, dtype=tf.float32):
    orientation = get_spin_batched(batched_point_squence, dtype=dtype)
    orientation_bool_vec = orientation > tf.constant([0.0], dtype)
    orientation_arr = tf.broadcast_to(tf.expand_dims(tf.expand_dims(orientation_bool_vec, axis=-1), axis=-1),
                                      batched_point_squence.shape)
    batched_point_squence = tf.where(orientation_arr, tf.reverse(batched_point_squence, axis=[1]),
                                     batched_point_squence)
    return batched_point_squence


def get_area_of_triangle(points, small_area_warning=10.0):
    """points is tensor with shape [3x2]"""
    logger.debug("get_area_of_triangle...")
    assert tf.is_tensor(points)
    distances = points - tf.roll(points, shift=-1, axis=-2)
    logger.debug("distances: {}".format(distances))
    euclid_distances = tf.math.reduce_euclidean_norm(distances, axis=-1)
    logger.debug("euclidean_norm_distances: {}".format(euclid_distances))
    s = 0.5 * tf.reduce_sum(euclid_distances, axis=-1)
    logger.debug("s: {}".format(s))
    red_prod = tf.reduce_prod(
        tf.broadcast_to(s, tf.shape(tf.transpose(euclid_distances))) - tf.transpose(euclid_distances), axis=0)
    area = tf.sqrt(s * red_prod)
    logger.debug("area: {}".format(area))
    if tf.executing_eagerly():
        if np.min(area) < small_area_warning:
            logger.warning("small area detected!")

            if tf.rank(area) >= 1:
                logger.warning("sorted areas: {}".format(np.sort(area)))
            else:
                logger.warning("sorted areas: {}".format(area))
    else:
        if tf.reduce_min(area) < small_area_warning:
            logger.warning("small area detected!")
    logger.debug("get_area_of_triangle... Done.")
    return area


if __name__ == "__main__":
    pass

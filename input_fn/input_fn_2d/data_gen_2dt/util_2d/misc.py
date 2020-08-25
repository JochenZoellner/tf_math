import logging

import numpy as np
import shapely.geometry as geometry
import matplotlib.pyplot as plt

logger = logging.getLogger("util_2d/misc.py")


def rotate_triangle_list(p1, p2, p3, phi):
    p_list = np.zeros((3, 2))
    for index, p in enumerate([p1, p2, p3]):
        p_list[index, 0] = p[0] * np.cos(phi) + p[1] * np.sin(phi)
        p_list[index, 1] = -1.0 * p[0] * np.sin(phi) + p[1] * np.cos(phi)
    return p_list[0], p_list[1], p_list[2]


def rotate_triangle(points, phi, unit="deg", fix_point="center"):
    if unit == "deg":
        x = 180.0j / np.pi
    elif unit == "rad":
        x = 1.0j
    else:
        raise AttributeError
    if fix_point == "center":
        d_xy = np.sum(points, axis=0) / 3.0
    elif fix_point == "zero":
        d_xy = np.array([0.0, 0.0])
    else:
        raise AttributeError

    center_points = points - d_xy
    r_points = (center_points[:, 0] + 1.0j * center_points[:, 1]) * np.e**(x * phi)
    return np.array([np.real(r_points), np.imag(r_points)]).transpose() + d_xy


def center_triangle_list(p1, p2, p3):
    x_c = (p1[0] + p2[0] + p3[0]) / 3.0
    y_c = (p1[1] + p2[1] + p3[1]) / 3.0
    p_list = np.zeros((3, 2), dtype=np.float32)
    for index, p in enumerate([p1, p2, p3]):
        p_list[index, 0] = p[0] - x_c
        p_list[index, 1] = p[1] - y_c

    return p_list[0], p_list[1], p_list[2]


def center_triangle(points):
    d_xy = np.sum(points, axis=0) / 3.0
    return points - np.expand_dims(d_xy, axis=0)


def translate_triangle_list(p1, p2, p3, delta_x_y):
    p_list = np.zeros((3, 2))
    for index, p in enumerate([p1, p2, p3]):
        p_list[index, 0] = p[0] + delta_x_y[0]
        p_list[index, 1] = p[1] + delta_x_y[1]
    return p_list[0], p_list[1], p_list[2]


def translate_triangle(points, delta_x_y):
    return points + delta_x_y


def get_spin(point_list):
    """sums all angles of a point_list/array (simple linear ring).
    If positive the direction is counter-clockwise and mathematical positive"""
    # if type(point_list) == list:
    #     arr = convert.tuples_to_array(point_list)
    # else:
    #     arr = point_list
    direction = 0.0
    # print(point_list)
    for index in range(len(point_list)):
        p0 = np.array(list(point_list[index - 2]))
        p1 = np.array(list(point_list[index - 1]))
        p2 = np.array(list(point_list[index]))
        s1 = p1 - p0
        s1_norm = s1 / np.sqrt(np.dot(s1, s1))
        s2 = p2 - p1
        s2_norm = s2 / np.sqrt(np.dot(s2, s2))
        s1_bar = (-s1_norm[1], s1_norm[0])
        direction += np.dot(s1_bar, s2_norm)
    # print(direction)
    return True if direction < 0.0 else False


def angle_between(p1, p2):
    ang1 = -np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return -1 * (np.rad2deg((ang1 - ang2) % (2 * np.pi)) - 180.0)


def min_angle(v1, v2):
    """ Returns the angle in DEGREE between vectors 'v1' and 'v2'
      result is given in DEG!

    >>> a = np.array([1.0, 0.0])
    >>> b = np.arange(0, 7) * 2 * np.pi / 7.0
    >>> b_ = np.array([np.real(np.e**(1.0j*b)), np.imag(np.e**(1.0j*b))])
    >>> [print(min_angle(a, b_[:, i])) for i in range(7)]
    0.0
    51.42857142857142
    77.14285714285715
    25.714285714285722
    25.714285714285708
    77.14285714285712
    51.428571428571445
    [None, None, None, None, None, None, None]
    """
    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2)) * np.sign(cosang)
    res = np.arctan2(sinang, cosang * np.sign(cosang)) / np.pi * 180
    return np.abs(res)


def min_angle_batched(a_batch, b_batch):
    """calculate the minimal angle between two vectors"""
    logger.info(f"a_batch: {a_batch.shape}; b_batch: {b_batch.shape}")

    dot_batched = np.sum(a_batch * b_batch, axis=-1)
    normed_product_batched = np.linalg.norm(a_batch, axis=-1) * np.linalg.norm(b_batch, axis=-1)
    angle_batched = np.arccos(dot_batched / normed_product_batched) * 180 / np.pi
    logger.debug(f"raw angle: {angle_batched}")
    min_angle_ = np.minimum(np.abs(angle_batched % 180.0), np.abs(180.0 - np.abs(angle_batched % 180.0)))
    # min_angle_ = 90.0 - np.abs(angle_batched % 180 - 90.0)
    logger.info(f"min angle batched: {min_angle_}")
    return min_angle_


def has_min_angle(points, min_angle):
    """check if polygon satisfies the min_angle condition
        calculate all "min_angle" between adjacent edges return True if all greater_equal than min_angle"""
    batch_a = points - np.roll(points, shift=1, axis=0)
    batch_b = np.roll(points, shift=-1, axis=0) - points
    min_angle_of_polygon = np.min(min_angle_batched(batch_a, batch_b))
    return True if min_angle_of_polygon >= min_angle else False


def point_distance_batched(point_array):
    res = np.linalg.norm(point_array - np. roll(point_array, shift=1, axis=0), axis=-1)
    logger.debug(f"min_distance_batched: {res}")
    return res


def has_min_point_distance_batched(point_array, min_dist):
    return True if np.min(point_distance_batched(point_array)) > min_dist else False


def get_area_of_triangle2(points):
    """deprecated just for test purpose"""
    a_x = points[0, 0]
    a_y = points[0, 1]
    b_x = points[1, 0]
    b_y = points[1, 1]
    c_x = points[2, 0]
    c_y = points[2, 1]
    return np.abs((a_x * (b_y - c_y) + b_x * (c_y - a_y) + c_x * (a_y - b_y)) / 2.0)


def get_area_of_triangle(points):
    assert type(points) == np.ndarray
    assert points.shape == (3, 2)
    distances = points - np.roll(points, axis=0, shift=1)
    # abs_distances = np.sqrt(np.sum(np.square(distances), axis=1))
    abs_distances = np.linalg.norm(distances, axis=-1)
    # assert np.allclose(abs_distances, abs_distances_2)
    s = 0.5 * np.sum(abs_distances)
    under_root = s * np.prod(s - abs_distances)
    return np.sqrt(under_root) if under_root >= 0.0 else -1.0


def get_min_aspect_ratio(points):
    assert type(points) == np.ndarray
    assert points.shape == (3, 2)
    distances = points - np.roll(points, axis=0, shift=1)
    abs_distances = np.sqrt(np.sum(np.square(distances), axis=1))
    s = 0.5 * np.sum(abs_distances)
    area = np.sqrt(s * np.prod(s - abs_distances))
    h_c = 2 / np.max(abs_distances) * area
    logger.debug("get_min_aspect_ratio:")
    logger.debug("points {}".format(points))
    logger.debug("distances: {}".format(abs_distances))
    logger.debug("A: {}".format(area))
    logger.debug("H_min: {}".format(h_c))
    logger.info("min_aspect_ratio: {}".format(h_c / np.max(abs_distances)))
    return h_c / np.max(abs_distances)


def has_min_aspect_ratio(points, min_ratio):
    return True if get_min_aspect_ratio(points) > min_ratio else False


def has_min_point_edge_distance(points, min_distance):
    bool_array = np.empty(points.shape[0], np.bool)
    for idx in range(points.shape[0]):
        shift_points = np.roll(points, shift=idx, axis=0)
        l_string = geometry.asLineString(shift_points[1:])
        test_point = geometry.asPoint(shift_points[0]).buffer(min_distance)
        bool_array[idx] = not test_point.intersects(l_string)
    return np.all(bool_array)


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=1)

    a = np.array([-1.0, 0.0])
    b = np.arange(0, 100) * 2 * np.pi / 100.0
    b_ = np.array([np.real(np.e**(1.0j*b)), np.imag(np.e**(1.0j*b))])
    plt.figure()
    plt.scatter(b_[0], b_[1])
    angle = np.empty(100)
    angle2 = np.empty(100)
    for i in range(100):
        angle[i] = (min_angle(a, b_[:, i]))
        angle2[i] = angle_between(a, b_[:, i])
            # if i % 80 == 0:
            #     print(f"{b_[:, i]}")
        if i % 80 == 0:
            print(f"{angle[i]}")
    plt.plot(angle)
    plt.plot(angle2)

    plt.show()

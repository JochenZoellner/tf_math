import logging

import numpy as np
from shapely import geometry

logger = logging.getLogger("convert 2D")


# logger.setLevel("DEBUG")
# logger.setLevel("INFO")


def tuples_to_array(t):
    """converts a list of point-tuple into np.ndarray with shape (?,2)"""
    assert type(t) is list
    assert len(t) != 0
    length = len(t)
    a = np.empty((length, 2))
    for i_, tuple_ in enumerate(t):
        a[i_] = np.array([tuple_[0], tuple_[1]])
    return a


def array_to_tuples(a):
    """converts a numpy array (shape (?,2) )into a list where each element is a point-tuple"""
    assert type(a) == np.ndarray
    t = []
    for i_ in range(a.shape[0]):
        t.append(tuple((a[i_, 0], a[i_, 1])))
    return t


def polygon_to_tuples(polygon):
    """point coordinates as tuple-list of a shapley.geometry.Polygon"""
    return [x for x in geometry.mapping(polygon)["coordinates"][0]]


def rphi_array_to_points_array(rphi_array):
    z_arr = rphi_array[:, 0] * np.exp(1.0j * 2.0 * np.pi * rphi_array[:, 1] / 360.0)
    points_array = np.array([z_arr.real, z_arr.imag]).transpose()
    return points_array


def points_array_to_rphi_array(points_array):
    z_array = points_array[:, 0] + 1.0j * points_array[:, 1]
    r_array = np.abs(z_array)
    phi_array = ((np.arctan2(z_array.imag, z_array.real) * 360 / (2.0 * np.pi)) + 360.0) % 360.0
    rphi_array = np.stack((r_array, phi_array), axis=1)
    return rphi_array


def rpf_to_points_array(rpf_dict):
    z_move = 1.0j * rpf_dict['translation'][1] + rpf_dict['translation'][0]
    pts_arr_rp = np.empty([rpf_dict['edges'], 2])
    for index in range(rpf_dict['edges']):
        alpha = 2.0 * np.pi * float(index) / float(rpf_dict['edges']) + rpf_dict['rotation']
        z = rpf_dict['radius'] * np.exp(1.0j * alpha) + z_move
        pts_arr_rp[index, :] = np.array([z.real, z.imag])
    return pts_arr_rp


def regular_polygon_points_array_to_rpf(points_array):
    translation = np.mean(points_array, axis=0)
    centered_points_array = points_array - translation
    edges = points_array.shape[0]
    radius = np.sqrt(np.sum(np.square(centered_points_array[0])))
    y_max_pt_index = np.argmax(centered_points_array[:, 0])
    alpha_ = np.arctan(centered_points_array[y_max_pt_index][1] / centered_points_array[y_max_pt_index][0])
    rotation = (alpha_ + 2.0 * np.pi) % (2.0 * np.pi / float(edges))
    rpf_dict = {"radius": radius, "rotation": rotation, "translation": translation, "edges": edges}

    return rpf_dict

import matplotlib
import numpy as np
import pytest
import logging
import tensorflow as tf
import shapely.geometry as geometry
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from input_fn.input_fn_2d.data_gen_2dt.util_2d import misc, misc_tf, convert, object_generator




def test_min_angle():
    np.set_printoptions(precision=4, suppress=True)
    angle_array, _, _ = min_angle_array()
    logging.info(np.abs(angle_array - np.roll(angle_array, shift=1)))
    logging.info(np.abs(angle_array - np.roll(angle_array, shift=1)) < 51.5)
    assert np.allclose(np.abs(angle_array - np.roll(angle_array, shift=1)), 3.6)
    assert np.all(angle_array >= 0.0), "Negative value in 'min_angle' results!'"
    assert np.all(angle_array <= 90.0), "More than 90° in 'min_angle' results!'"


def min_angle_array():
    """helper function for test min_angle and min_angle_batched"""
    N = 100
    a = np.array([1.0, 0.0])
    b = np.arange(0, N) * 2 * np.pi / 100.0
    b = np.array([np.real(np.e**(1.0j*b)), np.imag(np.e**(1.0j*b))]).transpose()
    angle_array = np.empty(N)
    for i in range(N):
        angle_array[i] = misc.min_angle(a, b[i])
    return angle_array, a, b


def test_min_angle_batched():
    angle_array, a, b = min_angle_array()
    angle_array_batched =misc.min_angle_batched(np.broadcast_to(np.expand_dims(a, axis=0), shape=b.shape), b)
    assert np.allclose(angle_array, angle_array_batched)


def test_get_orientation_vs_get_spin():

    rng = np.random.Generator(np.random.PCG64(1233))
    for i in range(10):
        triangle = object_generator.generate_target_triangle(rng=rng)
        orientation_b = misc_tf.get_spin_batched(np.expand_dims(triangle, axis=0))
        orientation_bool_vec = orientation_b > tf.constant([0.0], tf.float32)
        logging.info(f"orientation_b: {orientation_b}")
        logging.info(f"orientation_bool: {orientation_bool_vec}")
        get_spin_bool_vec = np.array([misc.get_spin(convert.array_to_tuples(triangle))])
        logging.info(f"get_spin: {get_spin_bool_vec}")
        assert np.any(get_spin_bool_vec == orientation_bool_vec)


def test_rotate_translate_center_triangle():
    rng = np.random.Generator(np.random.PCG64(1234))
    for sample in range(10):
        triangle = object_generator.generate_target_triangle(rng=rng)
        assert np.any(triangle != misc.center_triangle(triangle))


def test_center_triangle():
    rng = np.random.Generator(np.random.PCG64(1235))
    for sample in range(10):
        triangle = object_generator.generate_target_triangle(rng=rng)
        assert np.allclose(misc.center_triangle(triangle),
                           np.array([misc.center_triangle_list(triangle[0], triangle[1], triangle[2])], dtype=np.float32))


def test_rotate_triangle():
    rng = np.random.Generator(np.random.PCG64(1234))
    for sample in range(10):
        triangle = object_generator.generate_target_triangle(rng=rng)
        phi = rng.uniform(0, 4 * np.pi)
        rt_array = misc.rotate_triangle(triangle, -phi, unit="rad")
        d_xy = np.sum(triangle, axis=0) / 3.0
        rt_list = np.stack((misc.rotate_triangle_list(triangle[0]-d_xy, triangle[1]-d_xy, triangle[2]-d_xy, phi)))+d_xy
        assert np.allclose(rt_list, rt_array)
        rt_list = np.stack((misc.rotate_triangle_list(triangle[0], triangle[1], triangle[2], phi)))
        rt_array = misc.rotate_triangle(triangle, -phi, unit="rad", fix_point="zero")
        assert np.allclose(rt_list, rt_array)

        # print(rt_list)
        # print(rt_array)
        # plt.figure()
        # plt.scatter(rt_list[:, 0], rt_list[:, 1], marker="^")
        # plt.scatter(rt_array[:, 0], rt_array[:, 1], marker="+")
        # plt.scatter(triangle[:, 0], triangle[:, 1], marker="o")
        # plt.show()


def test_get_area_triangle():
    rng = np.random.Generator(np.random.PCG64(1234))
    for sample in range(100):
        triangle = object_generator.generate_target_triangle(rng=rng).astype(np.float64)
        assert np.isclose(misc.get_area_of_triangle(triangle), misc.get_area_of_triangle2(triangle))


def test_translate_triangle():
    from ..object_generator import generate_target_triangle
    from ..misc import translate_triangle, translate_triangle_list

def test_has_min_point_edge_distance():
    small_dist_array = misc.translate_triangle(np.array([(0, 0),  (1, 0), (1, 1), (0.5, 0.1), (0, 1)]), np.array([(2, 2)]))
    no_dist_array = misc.translate_triangle(np.array([(0, 0),  (1, 0), (1, 1), (0.5, 0.0), (0, 1)]), np.array([(2, 1)]))
    not_simple_array = misc.translate_triangle(np.array([(0, 0),  (1, 0), (1, 1), (0.5, -0.5), (0, 1)]), np.array([(1, 1)]))
    right_dist_array = misc.translate_triangle(np.array([(0, 0),  (1, 0), (1, 1), (0.5, 0.5), (0, 1)]), np.array([(1, 2)]))
    logging.info(f"small_distance polygon: {misc.has_min_point_edge_distance(small_dist_array, 0.2)}")
    logging.info(f"no_distance polygon: {misc.has_min_point_edge_distance(no_dist_array, 0.2)}")
    logging.info(f"right_distance polygon: {misc.has_min_point_edge_distance(right_dist_array, 0.2)}")
    logging.info(f"not_simple polygon: {misc.has_min_point_edge_distance(not_simple_array, 0.2)}")
    assert not misc.has_min_point_edge_distance(small_dist_array, 0.2)
    assert misc.has_min_point_edge_distance(small_dist_array, 0.05)
    assert not misc.has_min_point_edge_distance(no_dist_array, 0.2)
    assert misc.has_min_point_edge_distance(right_dist_array, 0.2)
    assert not geometry.asPolygon(not_simple_array).is_simple

    small_dist_polygon = Polygon(small_dist_array, True)
    no_dist_polygon = Polygon(no_dist_array, True)
    not_simple_polygon = Polygon(not_simple_array, True)
    right_dist_polygon = Polygon(right_dist_array, True)
    patches = [small_dist_polygon, no_dist_polygon, right_dist_polygon, not_simple_polygon]
    fig, ax = plt.subplots()
    colors = 100 * np.random.rand(len(patches))
    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)
    p.set_array(np.array(colors))
    ax.add_collection(p)
    plt.xlim(0.0, 4.0)
    plt.ylim(0.0, 4.0)
    plt.show()


if __name__ == '__main__':
    # logging.basicConfig(level="WARNING")
    # pytest.main(["test_misc.py", "-v", "-s"])
    logging.basicConfig(level="INFO")
    # pytest.main(["test_misc.py", "-k test_get_area_triangle", "-v", "-s"])
    # test_min_angle()
    test_has_min_point_edge_distance()

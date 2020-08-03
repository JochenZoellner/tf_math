import logging
import os
import time
import pytest

import matplotlib.pyplot as plt
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from shapely import geometry

import input_fn.input_fn_2d.input_fn_2d_util as if2d_util
from input_fn.input_fn_2d.data_gen_2dt.util_2d import  object_generator, convert, scatter, misc

logger = logging.getLogger("test_scatter_2d")
# logger.setLevel("DEBUG")
# logger.setLevel("INFO")

if __name__ == "__main__":
    # logging.basicConfig()
    # logging.basicConfig(level="DEBUG")
    # logging.basicConfig(level="INFO")
    np.set_printoptions(precision=6, suppress=True)




def debug_regular_polygon1():
    for i in range(10):
        pts_arr_rp, _ = object_generator.generate_target_regular_polygon()
        print(pts_arr_rp)
        fig, ax1 = plt.subplots(1, 1, figsize=(9.5, 9.5))
        ax1.fill(pts_arr_rp.transpose()[0], pts_arr_rp.transpose()[1])
        ax1.set_xlim((-50, 50))
        ax1.set_ylim((-50, 50))
        ax1.set_aspect(aspect=1.0)
        plt.show()


def debug_regular_polygon2():
    for i in range(10):
        pts_arr_rp, _ = object_generator.generate_target_regular_polygon()
        points = convert.array_to_tuples(pts_arr_rp)
        polygon_calculator = scatter.Fcalculator(points)
        phi_array = np.arange(np.pi / 2 - 1.5, np.pi / 2 + 1.5, 0.01)
        polygon_scatter_res = polygon_calculator.f_of_phi(phi=phi_array).astype(dtype=np.complex64)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 14))
        ax1.fill(pts_arr_rp.transpose()[0], pts_arr_rp.transpose()[1])
        ax1.set_xlim((-50, 50))
        ax1.set_ylim((-50, 50))
        ax1.set_aspect(aspect=1.0)
        ax2.plot(phi_array, polygon_scatter_res.real, "-b", label="real_polygon")
        ax2.plot(phi_array, polygon_scatter_res.imag, "-y", label="imag_polygon")
        ax2.plot(phi_array, np.abs(polygon_scatter_res), "-r", label="abs_polygon")
        ax2.legend(loc=2)
        plt.show()


def test_regular_polygon_transformation():
    for i in range(10000):
        rpf_dict = object_generator.generate_rpf()
        rp_points_array = convert.rpf_to_points_array(rpf_dict)
        rpf_dict2 = convert.regular_polygon_points_array_to_rpf(rp_points_array)
        for key in rpf_dict.keys():
            # print(key, np.array(rpf_dict[key]),  np.array(rpf_dict2[key]))
            assert np.isclose(np.sum(np.array(rpf_dict[key])), np.sum(np.array(rpf_dict2[key]))), \
                "{} is differen after transformation! before: {}, after: {}".format(key, rpf_dict[key], rpf_dict2[key])
    return 0


# def test_star_polygon_transformation(self):
#     for i in range(100):
#         rphi_array = object_generator.generate_target_star_polygon2(edges=8, max_radius=50, min_radius=5, angle_epsilon=10.0)
#         points_array = convert.rphi_array_to_points_array(rphi_array)
#         rphi_array2 = convert.points_array_to_rphi_array(points_array)
#         assert np.allclose(rphi_array,
#                            rphi_array2), "\n{}\n is differen after transformation! before: \n{}, after: \n{}".format(
#             points_array, rphi_array, rphi_array2)
#     return 0


def test_star_polygon():
    for i in range(100):
        rphi_arr = object_generator.generate_target_star_polygon(max_radius=30, edges=10)
        # print("  radius\t phi in deg")
        # print(rphi_arr)
        z_arr = rphi_arr[:, 0] * np.exp(1.0j * 2.0 * np.pi * rphi_arr[:, 1] / 360.0)
        points_array = np.array([z_arr.real, z_arr.imag]).transpose()
        # print("  X\t\t\t Y")
        # print(points_array)
        # print("mean x = {}".format(np.sum(points_array[:, 0])))
        # print("mean y = {}".format(np.sum(points_array[:, 1])))
        # fig, ax1 = plt.subplots(1, 1, figsize=(9.5, 14))
        # ax1.fill(points_array.transpose()[0], points_array.transpose()[1])
        # # ax1.plot(points_array.transpose()[0, :2], points_array.transpose()[1, :2], "r")
        # ax1.set_xlim((-50, 50))
        # ax1.set_ylim((-50, 50))
        # ax1.set_aspect(aspect=1.0)
        # ax1.grid()
        # plt.show()


def test_arbitrary_polygon():
    from input_fn.input_fn_2d.data_gen_2dt.util_2d.object_generator import generate_target_polygon
    for sample in range(100):
        polygon_points, corners = generate_target_polygon(max_edges=12, min_edges=3, min_angle=20.0, min_distance=5.0)

        polygon_lr = geometry.asPolygon(polygon_points)
        plt.figure("arbirary polygon plot")
        plt.xlim((-50.0, 50.0))
        plt.ylim((-50.0, 50.0))
        plt.plot(*polygon_lr.exterior.xy)
        plt.show()


def test_arbitrary_polygon2():
    rng = np.random.Generator(np.random.PCG64(1236))
    for sample in range(10):
        polygon, corners = object_generator.generate_target_polygon(rng=rng, min_distance=2.0, min_angle=15.0, min_area=11.0)
        assert polygon.shape[0] == corners
        assert misc.has_min_point_edge_distance(polygon, min_distance=2.0)
        assert misc.has_min_point_distance_batched(polygon, min_dist=2.0)
        assert misc.has_min_angle(polygon, min_angle=15.0)
        assert geometry.asPolygon(polygon).area >= 11.0


if __name__ == "__main__":
    print("run tf_polygon_2d_helper.py as main")
    time.sleep(0.001)
    logging.basicConfig(level="DEBUG")

    # pytest.main(["test_objects.py", "-k arbirary", "-v", "-s", "--plots"])
    test_arbitrary_polygon()
    # logger.setLevel("INFO")
    # test_result = unittest.main(verbosity=4)

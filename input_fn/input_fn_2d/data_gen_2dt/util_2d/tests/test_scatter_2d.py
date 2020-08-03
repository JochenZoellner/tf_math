import logging
import os
import time
import unittest

import matplotlib.pyplot as plt
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from shapely import geometry

import input_fn.input_fn_2d.input_fn_2d_util as if2d_util
import input_fn.input_fn_2d.data_gen_2dt.util_2d.scatter as scatter
import input_fn.input_fn_2d.data_gen_2dt.util_2d.convert as convert
import input_fn.input_fn_2d.data_gen_2dt.util_2d.misc as misc
import input_fn.input_fn_2d.data_gen_2dt.util_2d.misc_tf as misc_tf
import input_fn.input_fn_2d.data_gen_2dt.util_2d.object_generator as object_generator

logger = logging.getLogger("test_scatter_2d")
# logger.setLevel("DEBUG")
# logger.setLevel("INFO")

if __name__ == "__main__":
    # logging.basicConfig()
    # logging.basicConfig(level="DEBUG")
    # logging.basicConfig(level="INFO")
    np.set_printoptions(precision=6, suppress=True)


class TestPolygon2DHelper(unittest.TestCase):

    # def test_track_gradient_graph_mode(self):
    #     convex_polygon_arr = object_generator.generate_target_polygon(max_edge=3)
    #     convex_polygon_arr = tf.constant(convex_polygon_arr, dtype=tf.float64)
    #     DEBUG = False
    #     polygon_calculator_target = scatter.ScatterCalculator2D(points=convex_polygon_arr, debug=DEBUG)
    #
    #     dphi = 0.0001
    #     har = 1.0 / 180.0 * np.pi  # hole_half_angle_rad
    #     mac = 1.0 / 180.0 * np.pi  # max_angle_of_view_cut_rad
    #     phi_array = np.concatenate((np.arange(0 + har, np.pi / 2 - mac, dphi),
    #                                 np.arange(np.pi / 2 + har, np.pi - mac, dphi)))
    #
    #     polygon_scatter_res_target = polygon_calculator_target.fc_of_phi(phi=phi_array)
    #     convex_polygon_tensor = tf.Variable(convex_polygon_arr + np.random.uniform(-0.1, 0.1, convex_polygon_arr.shape))
    #
    #     with tf.GradientTape() as tape:
    #         polygon_calculator = scatter.ScatterCalculator2D(points=convex_polygon_tensor, debug=DEBUG)
    #         polygon_scatter_res = polygon_calculator.fc_of_phi(phi=phi_array)
    #         loss = tf.keras.losses.mean_absolute_error(polygon_scatter_res_target, polygon_scatter_res)
    #         tf.print(loss)
    #         gradient = tape.gradient(loss, convex_polygon_tensor)
    #         tf.print(gradient)
    #     tf.assert_greater(tf.reduce_sum(tf.abs(gradient)), tf.constant(0.0, dtype=tf.float64))
    #     return 0

    def test_get_area_of_triangle(self):
        logger.info("Run test_get_area_of_triangle...")
        test_points = tf.constant([[3.0, 0.0],
                                   [0.0, 4.0],
                                   [0.0, 0.0]])
        logger.debug("test_points:\n{}".format(test_points))

        triangle_area = misc_tf.get_area_of_triangle(points=test_points)
        assert tf.is_tensor(triangle_area)
        logger.debug("triangle area: {}".format(triangle_area))
        tf.assert_equal(triangle_area, tf.constant(6.0, dtype=tf.float32), summarize=10)
        logger.info("Run test_get_area_of_triangle... Done.")
        return 0

    def test_get_area_of_triangle_batched(self):
        logger.info("Run test_get_area_of_triangle_batched...")
        test_points = tf.constant([[[3.0, 0.0],
                                    [0.0, 4.0],
                                    [0.0, 0.0]],
                                   [[6.0, 0.0],
                                    [0.0, 8.0],
                                    [0.0, 0.0]],
                                   [[1.0, 7.1],
                                    [-2.0, 5.5],
                                    [-3.0, -2.0]]
                                   ], dtype=tf.float64)
        logger.debug("test_points:\n{}".format(test_points))

        triangle_area = misc_tf.get_area_of_triangle(points=test_points)
        assert tf.is_tensor(triangle_area)
        logger.debug("triangle area: {}".format(triangle_area))
        tf.debugging.assert_near(triangle_area, tf.constant([6.0, 24.0, 10.45], dtype=tf.float64), summarize=10)
        logger.info("Run test_get_area_of_triangle_batched... Done.")
        return 0

    def test_compare_np_vs_tf_scatter_polygon(self):
        logger.info("Run test_compare_np_vs_tf_scatter_polygon...")

        for phi_function in ["phi_array_open_symetric_no90", "phi_array_open_no90", "phi_array_open", "phi_array"]:
            logger.debug("use phi function: {}".format(phi_function))
            max_edges = 6
            convex_polygon_arr, _ = object_generator.generate_target_polygon(max_edges=max_edges)
            convex_polygon_arr = np.pad(convex_polygon_arr, pad_width=[(0, max_edges - convex_polygon_arr.shape[0]), (0, 0)],
                                   mode='edge').astype(np.float32)
            convex_polygon_arr = tf.constant(convex_polygon_arr, dtype=tf.float64)
            phi_array = getattr(if2d_util, phi_function)(delta_phi=0.001)
            tf.config.experimental_run_functions_eagerly(run_eagerly=True)
            polygon_calculator = scatter.ScatterCalculator2D(points=convex_polygon_arr, debug=True, allow_variable_edges=True)
            polygon_scatter_res = polygon_calculator.fc_of_phi(phi=phi_array)
            logger.debug("polygon_scatter_res: {}".format(polygon_scatter_res))

        logger.info("Run test_compare_np_vs_tf_scatter_polygon... Done.")
        return 0

    def test_scatter_polygon_2d_tf_vs_layer(self):
        import model_fn.util_model_fn.custom_layers as c_layer
        import input_fn.input_fn_2d.input_fn_2d_util as if2d_util
        np.random.seed(1)

        DTYPE = np.float64
        DTYPE_TF = tf.float64
        # phi_functions = [if2d_util.phi_array_open_symetric_no90,
        #                  if2d_util.phi_array_open_no90,
        #                  if2d_util.phi_array_open,
        #                  if2d_util.phi_array]
        phi_functions = [if2d_util.phi_array_open_symetric_no90]
        np_rng = np.random.Generator(np.random.PCG64(1234))
        rnd_ints = np_rng.integers(0, 100, size=12)
        logger.debug("test random integers: {}".format(rnd_ints))
        for db_mode in [True, False]:
            logger.info("Debug mode: {}".format(db_mode))
            tf.config.experimental_run_functions_eagerly(run_eagerly=db_mode)

            for target in range(2):
                for phi_function in phi_functions:
                    phi_array = phi_function(delta_phi=0.001, dtype=DTYPE)
                    max_edges = 6
                    convex_polygon_arr, _ = object_generator.generate_target_polygon(max_edges=max_edges, rng=np_rng)
                    convex_polygon_arr = np.pad(convex_polygon_arr,
                                                pad_width=[(0, max_edges - convex_polygon_arr.shape[0]), (0, 0)],
                                                mode='edge').astype(np.float32)
                    # convex_polygon_tuple = convert.array_to_tuples(convex_polygon_arr)
                    scatter_calculator_2d_tf = scatter.ScatterCalculator2D(points=convex_polygon_arr, debug=db_mode,
                                                                           dtype=DTYPE_TF, allow_variable_edges=True)

                    if not db_mode:
                        phi_array = tf.cast(phi_array, dtype=DTYPE_TF)

                    polygon_scatter_res = scatter_calculator_2d_tf.fc_of_phi(phi=phi_array)
                    if isinstance(polygon_scatter_res, tf.Tensor):
                        polygon_scatter_res = polygon_scatter_res.numpy().astype(dtype=np.complex64)
                    else:
                        polygon_scatter_res = polygon_scatter_res.astype(dtype=np.complex64)

                    test_reference_mean = np.mean(polygon_scatter_res)
                    logger.info("test reference : {}".format(test_reference_mean))
                    scatter_polygon_2d_layer = c_layer.ScatterPolygon2D(tf.expand_dims(phi_array, axis=0),
                                                                        with_batch_dim=False, allow_variable_edges=True)
                    sp2dl_res = scatter_polygon_2d_layer(tf.constant(convex_polygon_arr, dtype=DTYPE_TF))
                    test_layer_mean = np.mean(sp2dl_res[0].numpy() + 1.0j * sp2dl_res[1].numpy())
                    logger.info("test Layer     : {}".format(test_layer_mean))
                    phi_tf = tf.expand_dims(phi_array, axis=0)
                    fc_one = tf.concat((phi_tf, tf.zeros_like(phi_tf), tf.ones_like(phi_tf)), axis=0)
                    fc_one_b = tf.expand_dims(fc_one, axis=0)
                    fc_batch = tf.concat((fc_one_b, fc_one_b, fc_one_b, fc_one_b), axis=0)
                    convex_polygon_arr_b = tf.expand_dims(convex_polygon_arr, axis=0)
                    convex_polygon_arr_batch = tf.concat(
                        (convex_polygon_arr_b, convex_polygon_arr_b, convex_polygon_arr_b, convex_polygon_arr_b), axis=0)
                    scatter_polygon_2d_layer_batch = c_layer.ScatterPolygon2D(fc_batch, with_batch_dim=True, allow_variable_edges=True)
                    sp2dl_b_res = scatter_polygon_2d_layer_batch(tf.cast(convex_polygon_arr_batch, dtype=DTYPE_TF))
                    test_layerbatch_mean = np.mean(sp2dl_b_res[:, 0].numpy() + 1.0j * sp2dl_b_res[:, 1].numpy())
                    logger.info("test BatchLayer: {}".format(test_layerbatch_mean))
                    assert np.allclose(test_reference_mean, test_layer_mean)
                    assert np.allclose(test_layerbatch_mean, test_layer_mean)

                    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 14))
                    # ax1.plot(phi_array, polygon_scatter_res.real, "-b", label="test_real")
                    # ax1.plot(phi_array, sp2dl_b_res.numpy()[0, 0, :], "+r", label="batch_real")
                    # ax1.plot(phi_array, sp2dl_res.numpy()[0, :], ".y", label="layer_real")
                    # ax2.fill(convex_polygon_arr.transpose()[0], convex_polygon_arr.transpose()[1])
                    # ax2.set_xlim((-50, 50))
                    # ax2.set_ylim((-50, 50))
                    # ax2.set_aspect(aspect=1.0)
                    # plt.show()
        return 0


if __name__ == "__main__":
    print("run tf_polygon_2d_helper.py as main")
    time.sleep(0.001)
    # logger.setLevel("INFO")
    test_result = unittest.main(verbosity=1)

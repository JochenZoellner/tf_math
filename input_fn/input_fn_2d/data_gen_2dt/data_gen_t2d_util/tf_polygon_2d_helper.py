import logging
import os
import time
import unittest

import matplotlib.pyplot as plt
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from shapely import geometry

import input_fn.input_fn_2d.data_gen_2dt.data_gen_t2d_util.polygone_2d_helper as old_helper
import input_fn.input_fn_2d.input_fn_2d_util as if2d_util
logger = logging.getLogger("tf_polygone_2d_helper")
# logger.setLevel("DEBUG")
# logger.setLevel("INFO")

if __name__ == "__main__":
    # logging.basicConfig()
    # logging.basicConfig(level="DEBUG")
    # logging.basicConfig(level="INFO")
    np.set_printoptions(precision=6, suppress=True)


class ScatterCalculator2D:
    def __init__(self, points, epsilon=np.array(0.0001), debug=False, dtype=tf.float64):
        """points is tensor with shape [3x2]"""
        self.logger = logging.getLogger("ScatterCalculator2D")
        self.dtype = dtype
        if self.dtype == tf.float64:
            self.c_dtype = tf.complex128
            self.np_dtype = np.float64
            self.np_c_dtype = np.complex128
        elif self.dtype == tf.float32:
            self.logger.warning("using float32 is not tested for numerical stability")
            self.c_dtype = tf.complex64
            self.np_dtype = np.float32
            self.np_c_dtype = np.complex64
        else:
            raise TypeError("Unsupported dtype in ScatterCalculator2D, only tf.float32 and tf.float62 is supported.")

        self.epsilon_tf = tf.constant(epsilon, dtype=self.dtype)
        self.points_tf = tf.constant(points, dtype=self.dtype)
        self._debug = debug
        self._cross = tf.constant([[0.0, 1.0], [-1.0, 0.0]], dtype=self.dtype)

        self.epsilon = epsilon
        self.points = points

    def update_points(self, points):
        self.points_tf = points

    def q_of_phi(self, phi):
        phi_tf = tf.cast(phi, dtype=self.dtype)
        a_tf = tf.math.cos(phi_tf)
        b_tf = tf.math.sin(phi_tf) - 1.0
        q_tf = tf.stack([a_tf, b_tf])
        if self._debug:
            phi = np.array(phi, dtype=self.np_dtype)
            tf.assert_equal(phi_tf, phi)
            a_ = np.cos(phi)
            b_ = np.sin(phi) - 1.0
            q = np.array([a_, b_])
            self.logger.debug("q^2: {}".format(tf.math.abs(q[0] ** 2 + q[1] ** 2)))
            assert np.array_equal(a_tf, a_)
            assert np.array_equal(q, q_tf.numpy())
            return q
        else:
            return q_tf

    @tf.function
    def fc_of_qs_arr(self, q, p0_, p1_, c=0.0):
        j_tf = tf.cast(tf.complex(0.0, 1.0), dtype=self.c_dtype)
        p0_tf = tf.cast(p0_, dtype=self.dtype)
        p1_tf = tf.cast(p1_, dtype=self.dtype)
        q_tf = tf.cast(q, dtype=self.dtype)
        c_tf = tf.cast(c, dtype=self.dtype)
        c_tfc = tf.cast(c, dtype=self.c_dtype)
        q_cross_tf = tf.matmul(tf.cast([[0.0, -1.0], [1.0, 0.0]], dtype=tf.float64), q_tf)

        p0p1_tf = p1_tf - p0_tf
        scale_tf = tf.cast(1.0 / tf.math.abs(q_tf[0] ** 2 + q_tf[1] ** 2), dtype=self.dtype)
        f_p0_tf = -tf.cast(1.0, dtype=tf.complex128) * tf.math.exp(j_tf * (tf.cast(complex_dot(p0_tf, q_tf), dtype=tf.complex128) + c_tfc))
        f_p1_tf = -tf.cast(1.0, dtype=tf.complex128) * tf.math.exp(j_tf * (tf.cast(complex_dot(p1_tf, q_tf), dtype=tf.complex128) + c_tfc))
        case1_array_tf = tf.cast(scale_tf * complex_dot(p0p1_tf, q_cross_tf), dtype=tf.complex128) * (f_p1_tf - f_p0_tf) / tf.cast(complex_dot(p0p1_tf, q_tf), dtype=tf.complex128)
        case2_array_tf = tf.cast(scale_tf * complex_dot(p0p1_tf, q_cross_tf), dtype=tf.complex128) * -j_tf * tf.math.exp(j_tf * tf.cast(complex_dot(p0_tf, q_tf) + c_tf, dtype=tf.complex128))
        res_array_tf = tf.where(tf.math.abs(complex_dot(p0p1_tf, q_tf)) >= 0.0001, case1_array_tf, case2_array_tf)

        if not self._debug:
            return res_array_tf
        else:
            self.logger.debug("start np vs. tf comparison in debug mode")
            j_ = np.array(1.0j, dtype=self.np_c_dtype)
            p0 = np.array(p0_, dtype=self.np_dtype)
            p1 = np.array(p1_, dtype=self.np_dtype)
            q = np.array(q, dtype=self.np_dtype)
            q_cross = np.array([-q[1], q[0]])
            c = np.array(c)
            scale = 1.0 / np.abs(q[0] ** 2 + q[1] ** 2)
            p0p1 = p1 - p0
            assert np.array_equal(p0, p0_tf.numpy())
            assert np.array_equal(p1, p1_tf.numpy())
            assert np.array_equal(q, q_tf.numpy())
            assert np.array_equal(q_cross, q_cross_tf.numpy())
            assert np.array_equal(c, c_tf.numpy())
            assert np.array_equal(scale, scale_tf.numpy())

            f_p0 = -np.array(1.0, dtype=self.np_c_dtype) * np.exp(j_ * (np.dot(p0, q) + c))
            f_p1 = -np.array(1.0, dtype=self.np_c_dtype) * np.exp(j_ * (np.dot(p1, q) + c))

            assert np.array_equal(f_p0, f_p0_tf.numpy())
            assert np.array_equal(f_p1, f_p1_tf.numpy())
            case1_array = np.array(scale * np.dot(p0p1, q_cross), dtype=self.np_c_dtype) * (f_p1 - f_p0) / tf.cast(np.dot(p0p1, q), dtype=self.np_c_dtype)
            case2_array = np.array(scale * np.dot(p0p1, q_cross), dtype=self.np_c_dtype) * -1.0j * np.exp(1.0j * (tf.cast(np.dot(p0, q), dtype=self.np_c_dtype) + c))
            # print("complex dot in",p0p1_tf,q_cross_tf )
            # print("complex dot out", complex_dot(p0p1_tf, q_cross_tf))
            self.logger.debug("case1_array_np:\n{}\ncase1_array_tf:\n{}".format(case1_array, case1_array_tf.numpy()))
            assert np.array_equal(case1_array, case1_array_tf.numpy())
            assert np.array_equal(case2_array, case2_array_tf.numpy())
            res_array = np.where(np.abs(np.dot(p0p1, q)) >= 0.0001, case1_array, case2_array)
            # if np.max(scale) >= 1000.0 / self.epsilon:
            #     logger.debug("Scale == NONE")
            #     polygon = geometry.Polygon(self.points)
            #     area = np.array(polygon.area, dtype=np.complex)
            #     logger.debug("area: {}".format(area))
            #     s_value = area / len(self.points)
            #     case3_array = np.ones_like(q[0]) * s_value
            #     res_array = np.where(scale >= 1000.0 / self.epsilon, case3_array, res_array)
            #
            # if tf.math.reduce_max(scale_tf) >= 1000.0 / self.epsilon_tf:
            #     logger.debug("Scale_tf == NONE")
            #     polygon = geometry.Polygon(self.points_tf)
            #     area = np.array(polygon.area, dtype=np.complex)
            #     logger.debug("area: {}".format(area))
            #     s_value = area / len(self.points)
            #     case3_array = np.ones_like(q[0]) * s_value
            #     res_array_tf = np.where(scale >= 1000.0 / self.epsilon, case3_array, res_array)
            # print("res_array", res_array)
            # print("res_array_tf", res_array_tf)
            assert np.array_equal(res_array, res_array_tf.numpy())
            self.logger.debug("np_res_array: {}".format(res_array))
            return res_array

    @tf.function
    def fc_of_phi(self, phi):
        self.logger.debug("###########################################")
        self.logger.info("phi: {}".format(phi))
        q = self.q_of_phi(phi)
        c = 0.0
        if not self._debug:
            sum_res = tf.cast(phi * tf.cast(0.0, dtype=self.dtype), dtype=self.c_dtype)
            c = tf.cast(c, dtype=self.dtype)
            for index in range(self.points_tf.shape[0]):
                p0 = self.points_tf[index - 1]
                p1 = self.points_tf[index]
                sum_res += self.fc_of_qs_arr(q, p0, p1, c=c)
        else:
            sum_res = tf.zeros_like(phi, dtype=self.c_dtype)
            for index in range(len(self.points)):
                self.logger.debug("index: {}".format(index))
                p0 = self.points[index - 1]
                p1 = self.points[index]
                logger.debug("p0: {}; p1: {}".format(p0, p1))
                sum_res += self.fc_of_qs_arr(q, p0, p1, c=c)
                logger.debug("sum_res {}".format(sum_res))
        final_res = sum_res
        self.logger.debug("sum_res.dtype: {}".format(sum_res.dtype))
        self.logger.info("final value: {}".format(final_res))
        return final_res


def complex_dot(a, b):
    return tf.einsum('i,i...->...', a, b)


def get_orientation_batched(batched_point_squence, dtype=tf.float32):
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
    Pm = tf.gather_nd(params=batched_point_squence,indices=indices_minus)
    P = tf.gather_nd(params=batched_point_squence, indices=indices)
    Pp = tf.gather_nd(params=batched_point_squence,indices=indices_plus)

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
def make_positiv_orientation(batched_point_squence, dtype=tf.float32):
    orientation = get_orientation_batched(batched_point_squence, dtype=dtype)
    orientation_bool_vec = orientation > tf.constant([0.0], dtype)
    orientation_arr = tf.broadcast_to(tf.expand_dims(tf.expand_dims(orientation_bool_vec, axis=-1), axis=-1),
                                      batched_point_squence.shape)
    batched_point_squence = tf.where(orientation_arr, tf.reverse(batched_point_squence, axis=[1]), batched_point_squence)
    return batched_point_squence


def get_area_of_triangle(points):
    """points is tensor with shape [3x2]"""
    logger.debug("get_area_of_triangle...")
    assert tf.is_tensor(points)
    distances = points - tf.roll(points, shift=-1, axis=-2)
    logger.debug("distances: {}".format(distances))
    euclid_distances = tf.math.reduce_euclidean_norm(distances, axis=-1)
    logger.debug("euclidean_norm_distances: {}".format(euclid_distances))
    s = 0.5 * tf.reduce_sum(euclid_distances, axis=-1)
    logger.debug("s: {}".format(s))
    red_prod = tf.reduce_prod(tf.broadcast_to(s, tf.shape(tf.transpose(euclid_distances))) - tf.transpose(euclid_distances), axis=0)
    area = tf.sqrt(s * red_prod)
    logger.debug("area: {}".format(area))
    logger.debug("get_area_of_triangle... Done.")
    return area


def test_track_gradient_graph_mode():
    convex_polygon_arr = old_helper.generate_target_polygon(max_edge=3)
    convex_polygon_arr = tf.constant(convex_polygon_arr, dtype=tf.float64)
    DEBUG = False
    polygon_calculator_target = ScatterCalculator2D(points=convex_polygon_arr, debug=DEBUG)

    dphi = 0.0001
    har = 1.0 / 180.0 * np.pi  # hole_half_angle_rad
    mac = 1.0 / 180.0 * np.pi  # max_angle_of_view_cut_rad
    phi_array = np.concatenate((np.arange(0 + har, np.pi / 2 - mac, dphi),
                                np.arange(np.pi / 2 + har, np.pi - mac, dphi)))

    polygon_scatter_res_target = polygon_calculator_target.fc_of_phi(phi=phi_array)
    convex_polygon_tensor = tf.Variable(convex_polygon_arr + np.random.uniform(-0.1, 0.1, convex_polygon_arr.shape))

    with tf.GradientTape() as tape:
        polygon_calculator = ScatterCalculator2D(points=convex_polygon_tensor, debug=DEBUG)
        polygon_scatter_res = polygon_calculator.fc_of_phi(phi=phi_array)
        loss = tf.keras.losses.mean_absolute_error(polygon_scatter_res_target, polygon_scatter_res)
        tf.print(loss)
        gradient = tape.gradient(loss, convex_polygon_tensor)
        tf.print(gradient)
    tf.assert_greater(tf.reduce_sum(tf.abs(gradient)), tf.constant(0.0, dtype=tf.float64))
    return 0




class TestPolygon2DHelper(unittest.TestCase):
    # def __index__(self):
    #     super(TestPolygon2DHelper).__init__()
    #
    #

    def test_get_area_of_triangle(self):
        logger.info("Run test_get_area_of_triangle...")
        test_points = tf.constant([[3.0, 0.0],
                                   [0.0, 4.0],
                                   [0.0, 0.0]])
        logger.debug("test_points:\n{}".format(test_points))

        triangle_area = get_area_of_triangle(points=test_points)
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

        triangle_area = get_area_of_triangle(points=test_points)
        assert tf.is_tensor(triangle_area)
        logger.debug("triangle area: {}".format(triangle_area))
        tf.debugging.assert_near(triangle_area, tf.constant([6.0, 24.0, 10.45], dtype=tf.float64), summarize=10)
        logger.info("Run test_get_area_of_triangle_batched... Done.")
        return 0

    def test_compare_np_vs_tf_scatter_polygon(self):
        logger.info("Run test_compare_np_vs_tf_scatter_polygon...")

        for phi_function in ["phi_array_open_symetric_no90", "phi_array_open_no90", "phi_array_open", "phi_array"]:
            logger.debug("use phi function: {}".format(phi_function))
            convex_polygon_arr = old_helper.generate_target_polygon(max_edge=6)
            convex_polygon_arr = tf.constant(convex_polygon_arr, dtype=tf.float64)
            phi_array = getattr(if2d_util, phi_function)(delta_phi=0.001)
            tf.config.experimental_run_functions_eagerly(run_eagerly=True)
            polygon_calculator = ScatterCalculator2D(points=convex_polygon_arr, debug=True)
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
                    convex_polygon_arr = old_helper.generate_target_polygon(max_edge=6, rng=np_rng)
                    convex_polygon_tuple = old_helper.array_to_tuples(convex_polygon_arr)
                    scatter_calculator_2d_tf = ScatterCalculator2D(points=convex_polygon_tuple, debug=db_mode, dtype=DTYPE_TF)

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
                                                                        with_batch_dim=False)
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
                    scatter_polygon_2d_layer_batch = c_layer.ScatterPolygon2D(fc_batch, with_batch_dim=True)
                    sp2dl_b_res = scatter_polygon_2d_layer_batch(tf.constant(convex_polygon_arr_batch, dtype=DTYPE_TF))
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

    # # test_track_gradient_graph_mode()
    TestPolygon2DHelper().test_get_area_of_triangle_batched()
    # test_result = unittest.main(verbosity=2)
    # print(test_result)
    # exit(0)







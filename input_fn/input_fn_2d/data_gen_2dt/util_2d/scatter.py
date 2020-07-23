import logging

import numpy as np
import tensorflow as tf
from shapely import geometry

logger = logging.getLogger("util_2d/scatter.py")

if __name__ == "__main__":
    logging.basicConfig()
    np.set_printoptions(precision=6, suppress=True)


# logger.setLevel("DEBUG")
# logger.setLevel("INFO")


class ScatterCalculator2D:
    def __init__(self, points, epsilon=np.array(0.0001), debug=False, dtype=tf.float64, allow_variable_edges=False):
        """points is tensor with shape [3x2]
        numpy and tensorflow implementation for debuggging purpose
        DO NOT USE, use Keras layer object instead.
        not graph mode compatible
        no batch support"""
        self.logger = logging.getLogger("ScatterCalculator2D")
        self.dtype = dtype
        self.allow_variable_edges = allow_variable_edges
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
        f_p0_tf = -tf.cast(1.0, dtype=tf.complex128) * tf.math.exp(
            j_tf * (tf.cast(complex_dot(p0_tf, q_tf), dtype=tf.complex128) + c_tfc))
        f_p1_tf = -tf.cast(1.0, dtype=tf.complex128) * tf.math.exp(
            j_tf * (tf.cast(complex_dot(p1_tf, q_tf), dtype=tf.complex128) + c_tfc))
        case1_array_tf = tf.cast(scale_tf * complex_dot(p0p1_tf, q_cross_tf), dtype=tf.complex128) * (
                    f_p1_tf - f_p0_tf) / tf.cast(complex_dot(p0p1_tf, q_tf), dtype=tf.complex128)
        case2_array_tf = tf.cast(scale_tf * complex_dot(p0p1_tf, q_cross_tf),
                                 dtype=tf.complex128) * -j_tf * tf.math.exp(
            j_tf * tf.cast(complex_dot(p0_tf, q_tf) + c_tf, dtype=tf.complex128))
        res_array_tf = tf.where(tf.math.abs(complex_dot(p0p1_tf, q_tf)) >= self.epsilon_tf, case1_array_tf,
                                case2_array_tf)

        # p0p1_dist_tf = tf.abs(tf.reduce_sum(tf.abs(p0p1_tf), axis=-1))
        # epsilon2_array_tf = tf.broadcast_to(tf.square(self.epsilon_tf), tf.shape(p0p1_dist_tf))
        # condition_zero = tf.transpose(tf.broadcast_to(tf.math.less(p0p1_dist_tf, epsilon2_array_tf), tf.reverse(tf.shape(res_array_tf), axis=[0])))
        #
        # if not self.allow_variable_edges and tf.reduce_any(condition_zero):
        #     tf.print(condition_zero)
        #     tf.print("identical points detected")
        # res_array_tf = tf.where(condition_zero, tf.broadcast_to(tf.constant(0.0, dtype=tf.complex128), tf.shape(res_array_tf)), res_array_tf)
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
            case1_array = np.array(scale * np.dot(p0p1, q_cross), dtype=self.np_c_dtype) * (f_p1 - f_p0) / tf.cast(
                np.dot(p0p1, q), dtype=self.np_c_dtype)
            case2_array = np.array(scale * np.dot(p0p1, q_cross), dtype=self.np_c_dtype) * -1.0j * np.exp(
                1.0j * (tf.cast(np.dot(p0, q), dtype=self.np_c_dtype) + c))



            # print("complex dot in",p0p1_tf,q_cross_tf )
            # print("complex dot out", complex_dot(p0p1_tf, q_cross_tf))
            self.logger.debug("case1_array_np:\n{}\ncase1_array_tf:\n{}".format(case1_array, case1_array_tf.numpy()))
            # assert np.array_equal(case1_array, case1_array_tf.numpy())
            # assert np.array_equal(case2_array, case2_array_tf.numpy())
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


class Fcalculator:
    def __init__(self, points, epsilon=np.array(0.0001)):
        """points is list of tupel with x,y like [(x1,y1), (x2,y2), (x3,y3),...]
        numpy only scatter function (legacy version)"""
        self.epsilon = epsilon
        self.points = points

    def q_of_phi(self, phi):
        a_ = np.cos(phi, dtype=np.float128)
        b_ = np.sin(phi, dtype=np.float128) - 1.0
        q = np.array([a_, b_])
        logger.debug("q^2: {}".format(np.abs(q[0] ** 2 + q[1] ** 2)))
        return q

    def fc_of_qs(self, q, p0_, p1_, c=0.0):
        p0 = np.array(p0_)
        p1 = np.array(p1_)
        c = np.array(c)
        q_cross = np.array([-q[1], q[0]])
        p0p1 = p1 - p0
        scale = 1.0 / np.abs(np.abs(q[0] ** 2 + q[1] ** 2))

        if scale >= 1000.0 / self.epsilon:
            logger.debug("Scale == NONE")
            polygon = geometry.Polygon(self.points)
            area = np.array(polygon.area, dtype=np.complex)
            logger.debug("area: {}".format(area))
            s_value = area / len(self.points)
        elif np.abs(np.dot(p0p1, q)) >= 0.0001:
            f_p0 = -1.0 * np.exp(1.0j * (np.dot(p0, q) + c))
            f_p1 = -1.0 * np.exp(1.0j * (np.dot(p1, q) + c))
            s_value = scale * np.dot(p0p1, q_cross) * (f_p1 - f_p0) / np.dot(p0p1, q)
        else:
            logger.debug("np.dot(p0p1, q) > epsilon")
            s_value = scale * np.dot(p0p1, q_cross) * -1.0j * np.exp(1.0j * (np.dot(p0, q) + c))

        logger.debug("s_value: {:1.6f}".format(s_value))
        return s_value

    def fc_of_qs_arr(self, q, p0_, p1_, c=0.0):
        p0 = np.array(p0_)
        p1 = np.array(p1_)
        c = np.array(c)
        q_cross = np.array([-q[1], q[0]])
        p0p1 = p1 - p0
        # scale = 1.0 / np.abs(np.dot(q, q))
        scale = 1.0 / np.abs(q[0] ** 2 + q[1] ** 2)

        f_p0 = -1.0 * np.exp(1.0j * (np.dot(p0, q) + c))
        f_p1 = -1.0 * np.exp(1.0j * (np.dot(p1, q) + c))

        case1_array = scale * np.dot(p0p1, q_cross) * (f_p1 - f_p0) / np.dot(p0p1, q)
        case2_array = scale * np.dot(p0p1, q_cross) * -1.0j * np.exp(1.0j * (np.dot(p0, q) + c))
        # print("case1_array.shape", case1_array.shape)
        res_array = np.where(np.abs(np.dot(p0p1, q)) >= 0.0001, case1_array, case2_array)

        if np.max(scale) >= 1000.0 / self.epsilon:
            logger.debug("Scale == NONE")
            polygon = geometry.Polygon(self.points)
            area = np.array(polygon.area, dtype=np.complex)
            logger.debug("area: {}".format(area))
            s_value = area / len(self.points)
            case3_array = np.ones_like(q[0]) * s_value
            res_array = np.where(scale >= 1000.0 / self.epsilon, case3_array, res_array)

        return res_array

    def f_of_phi(self, phi, c=0.0):
        logger.debug("###########################################")
        logger.info("phi: {}".format(phi))
        sum_res = np.zeros_like(phi, dtype=np.complex256)
        q = self.q_of_phi(phi)

        for index in range(len(self.points)):
            logger.debug("index: {}".format(index))
            p0 = self.points[index - 1]
            p1 = self.points[index]
            logger.debug("p0: {}; p1: {}".format(p0, p1))
            sum_res += self.fc_of_qs_arr(q, p0, p1, c=c)
            logger.debug("sum_res {}".format(sum_res))

        final_res = sum_res
        logger.debug("sum_res.dtype: {}".format(sum_res.dtype))
        logger.info("final value: {}".format(final_res))
        return final_res


class Fcalculator5Case:
    def __init__(self, p1=np.array([0.0, 0.0]), p2=np.array([1.0, 0.0]), p3=np.array([0.0, 1.0]),
                 epsilon=np.array(0.0001), no_check=False, complex_phi=False):
        self._p1 = np.array(p1, dtype=np.float128)
        self._p2 = np.array(p2, dtype=np.float128)
        self._p3 = np.array(p3, dtype=np.float128)
        if not no_check:  # skip check if valid input is ensured for better performance
            assert np.sum(np.square(np.abs(self._p1 - self._p2))) > (10 * epsilon) ** 2
            assert np.sum(np.square(np.abs(self._p2 - self._p3))) > (10 * epsilon) ** 2
            assert np.sum(np.square(np.abs(self._p3 - self._p1))) > (10 * epsilon) ** 2

        self._jacobi_det = np.abs((self._p2[0] - self._p1[0]) * (self._p3[1] - self._p1[1]) -
                                  (self._p3[0] - self._p1[0]) * (self._p2[1] - self._p1[1]))
        self._epsilon = np.array(epsilon, dtype=np.float128)
        self._complex_phi = complex_phi

    @staticmethod
    def _case1(a, b):
        logging.info("case1, a!=b, a!=0, b!=0")
        with np.errstate(divide='ignore', invalid='ignore'):  # nan values not used by np.where-condition
            return 1.0 / (a * b * (b - a)) * (b * (np.exp(1.0j * a) - 1.0) - a * (np.exp(1.0j * b) - 1.0))

    @staticmethod
    def _case2(b):
        logging.info("case2, a!=b, a=0, b!=0")
        with np.errstate(divide='ignore', invalid='ignore'):  # nan values not used by np.where-condition
            return 1.0j / b - 1 / b ** 2 * (np.exp(1.0j * b) - 1.0)

    @staticmethod
    def _case3(a):
        logging.info("case3, a!=b, b=0, a!=0")
        with np.errstate(divide='ignore', invalid='ignore'):  # nan values not used by np.where-condition
            return 1.0j / a - 1 / a ** 2 * (np.exp(1.0j * a) - 1.0)

    @staticmethod
    def _case5(a):
        logging.info("case5, a=b, b!=0, a!=0")
        with np.errstate(divide='ignore', invalid='ignore'):  # nan values not used by np.where-condition
            return np.exp(1.0j * a) / (1.0j * a) + (np.exp(1.0j * a) - 1.0) / a ** 2

    def set_triangle(self, p1=np.array([0.0, 0.0], dtype=np.float128), p2=np.array([1.0, 0.0], dtype=np.float128),
                     p3=np.array([0.0, 1.0], dtype=np.float128), no_check=False):
        self.__init__(p1, p2, p3, self._epsilon, no_check)

    def call_on_array(self, phi_array):
        if not self._complex_phi:
            phi_array = np.array(phi_array, dtype=np.float128)
            a_ = np.cos(phi_array, dtype=np.float128)
            b_ = np.sin(phi_array, dtype=np.float128) - 1.0
        else:
            phi_array = np.array(phi_array, dtype=np.complex128)
            a_ = phi_array.real
            b_ = phi_array.imag
            # print("a_", a_)
            # print("b_", b_)

        a = a_ * (self._p2[0] - self._p1[0]) + b_ * (self._p2[1] - self._p1[1])
        b = a_ * (self._p3[0] - self._p1[0]) + b_ * (self._p3[1] - self._p1[1])
        c = a_ * self._p1[0] + b_ * self._p1[1]

        f_array = np.full_like(phi_array, np.nan, dtype=np.complex256)

        a_not_b = np.abs(a - b) > self._epsilon
        a_is_b = np.abs(a - b) <= self._epsilon
        a_not_0 = np.abs(a) - self._epsilon > 0
        b_not_0 = np.abs(b) - self._epsilon > 0
        a_is_0 = np.abs(a) <= self._epsilon
        b_is_0 = np.abs(b) <= self._epsilon

        cond1 = np.logical_and(np.logical_and(a_not_b, a_not_0), b_not_0)
        cond2 = np.logical_and(np.logical_and(a_not_b, a_is_0), b_not_0)
        cond3 = np.logical_and(np.logical_and(a_not_b, b_is_0), a_not_0)
        cond4 = np.logical_or(np.logical_or(np.logical_and(a_is_0, a_is_b), np.logical_and(b_is_0, a_is_b)),
                              np.logical_and(b_is_0, a_is_0))
        cond5 = np.logical_and(np.logical_and(a_is_b, b_not_0), a_not_0)
        assert (np.logical_xor(cond1, np.logical_xor(cond2, np.logical_xor(cond3, np.logical_xor(cond4,
                                                                                                 cond5))))).all() == True

        f_array = np.where(cond1, self._case1(a, b), f_array)
        f_array = np.where(cond2, self._case2(b), f_array)
        f_array = np.where(cond3, self._case3(a), f_array)
        f_array = np.where(cond4, 0.5, f_array)
        f_array = np.where(cond5, self._case5(a), f_array)

        assert np.isnan(f_array).any() == False
        return self._jacobi_det * np.exp(1.0j * c) * f_array


def scatter_phi(phi, p1=np.array([0.0, 0.0], dtype=np.float64), p2=np.array([1.0, 0.0], dtype=np.float64),
                p3=np.array([0.0, 1.0], dtype=np.float64), epsilon=0.001,
                no_check=False):
    """slow straight forward version of ScatterCalculator2D, for scalar values of phi only"""
    phi = np.array(phi, dtype=np.float64)
    if not no_check:  # skip check if valid input is ensured for better performance
        assert np.sum(np.square(np.abs(p1 - p2))) > (10 * epsilon) ** 2
        assert np.sum(np.square(np.abs(p2 - p3))) > (10 * epsilon) ** 2
        assert np.sum(np.square(np.abs(p3 - p1))) > (10 * epsilon) ** 2

        if phi < 0 or phi > np.pi:
            logging.error("input phi is out of range; phi: {}".format(phi))
            return np.nan

    jacobi_det = np.abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
    a_ = np.cos(phi)
    b_ = np.sin(phi) - 1.0
    a = a_ * (p2[0] - p1[0]) + b_ * (p2[1] - p1[1])
    b = a_ * (p3[0] - p1[0]) + b_ * (p3[1] - p1[1])
    c = a_ * p1[0] + b_ * p1[1]

    if np.abs(a - b) > epsilon and np.abs(a - epsilon) > 0 and np.abs(b - epsilon) > 0:
        logging.info("case1, a!=b, a!=0, b!=0")
        f_ = 1.0 / (a * b * (b - a)) * (b * (np.exp(1.0j * a) - 1) -
                                        a * (np.exp(1.0j * b) - 1.0))
    elif np.abs(a - b) > epsilon and np.abs(b - epsilon) > 0:
        logging.info("case2, a!=b, a=0, b!=0")
        f_ = 1.0j / b - 1 / b ** 2 * (np.exp(1.0j * b) - 1.0)
    elif np.abs(a - b) > epsilon and np.abs(a - epsilon) > 0:
        logging.info("case3, a!=b, b=0, a!=0")
        f_ = 1.0j / a - 1 / a ** 2 * (np.exp(1.0j * a) - 1.0)
    elif np.abs(a) <= epsilon and np.abs(b) - epsilon <= 0:
        assert np.abs(a - b) <= epsilon  # a and b have same monotonie for phi > pi
        logging.info("case4, a=b, a=0, b=0")
        f_ = 0.5
    elif np.abs(a - b) <= epsilon and np.abs(a - epsilon) > 0 and np.abs(b - epsilon):
        logging.info("case5, a=b, b!=0, a!=0")
        f_ = np.exp(1.0j * a) / (1.0j * a) + (np.exp(1.0j * a) - 1.0) / a ** 2
    else:
        logging.error("unexpected values for a and b!; a={}; b={}".format(a, b))
        return np.nan

    return jacobi_det * np.exp(1.0j * c) * f_


def complex_dot(a, b):
    return tf.einsum('i,i...->...', a, b)

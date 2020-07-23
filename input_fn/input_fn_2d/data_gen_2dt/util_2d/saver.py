import logging
import sys

import numpy as np
import tensorflow as tf

import input_fn.input_fn_2d.data_gen_2dt.util_2d.misc as misc
import input_fn.input_fn_2d.data_gen_2dt.util_2d.object_generator as object_generator
import model_fn.util_model_fn.custom_layers as c_layers

logger = logging.getLogger(__name__)
# logger.setLevel(level="DEBUG")


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class Triangle2dSaver(object):
    def __init__(self, epsilon, phi_arr, x_sorted, samples_per_file, dphi=0.001, complex_phi=False, centered=False,
                 min_aspect_ratio=0.0):
        self.epsilon = epsilon
        self.complex_phi = complex_phi  # True if a and b set via complex value of phi-python-variable
        self.centered = centered
        self.dphi = dphi
        self.phi_arr = phi_arr
        self.x_sorted = x_sorted
        self.min_aspect_ratio = min_aspect_ratio
        self.samples_per_file = samples_per_file
        print("  init t2d-saver with:")
        print("  epsilon: {}".format(self.epsilon))
        print("  len phi_arr: {}".format(len(self.phi_arr)))
        print("  phi_complex: {}".format(self.complex_phi))
        if self.complex_phi:
            print("  phi_arr:\n{}".format(phi_arr))
        print("  dphi: {}".format(phi_arr[1] - phi_arr[0]))
        print("  x_sorted: {}".format(self.x_sorted))
        print("  min_aspect_ratio: {}".format(self.min_aspect_ratio))
        print("  samples_per_file: {}".format(self.samples_per_file))

    @staticmethod
    def serialize_example_pyfunction(points, fc_arr):
        # Create a feature
        feature_ = {'points': _bytes_feature(tf.compat.as_bytes(points.tostring())),
                    'fc': _bytes_feature(tf.compat.as_bytes(fc_arr.tostring()))}
        # Create an example protocol buffer
        return tf.train.Example(features=tf.train.Features(feature=feature_)).SerializeToString()

    def save_file_tf(self, filename):
        # open the TFRecords file
        D_TYPE = tf.float32
        phi_tf = tf.expand_dims(tf.expand_dims(tf.constant(self.phi_arr, D_TYPE), axis=0), axis=0)

        fc_obj = c_layers.ScatterPolygon2D(phi_tf, epsilon=self.epsilon, dtype=D_TYPE, with_batch_dim=True)
        point_list = []
        for i in range(self.samples_per_file):
            points = object_generator.generate_target_triangle(x_sorted=self.x_sorted,
                                                               min_aspect_ratio=self.min_aspect_ratio)
            point_list.append(points)

        batch_points = np.stack(point_list)
        batch_points = misc.make_positiv_orientation(batch_points).numpy()
        fc_arr = fc_obj(batch_points)

        with tf.io.TFRecordWriter(filename) as writer:
            for i in range(self.samples_per_file):
                # fc_arr = tf.concat((phi_tf[0], fc_arr[i]), axis=0)
                serialized_sample = self.serialize_example_pyfunction(batch_points[i],
                                                                      fc_arr=tf.concat((phi_tf[0], fc_arr[i]),
                                                                                       axis=0).numpy())
                # Serialize to string and write on the file

                writer.write(serialized_sample)

        sys.stdout.flush()


class ArbitraryPolygon2dSaver(object):
    def __init__(self, epsilon, phi_arr, samples_per_file, min_edges=3, max_edges=6, max_size=50, dtype=tf.float32):
        self.epsilon = epsilon
        self.epsilon_tf = tf.constant(epsilon, dtype=dtype)
        self.phi_arr = phi_arr
        self.dphi = np.abs(phi_arr[1] - phi_arr[0])
        self.samples_per_file = samples_per_file
        self.max_edges = max_edges
        self.min_edges = min_edges
        self.max_size = max_size
        self._dtype = dtype
        print("  init polygon2d-saver with:")
        print("  epsilon: {}".format(self.epsilon))
        print("  max edges of polygon: {}".format(self.max_edges))
        print("  max size of polygon: {}".format(self.max_size))
        print("  len phi_arr: {}".format(len(self.phi_arr)))
        print("  dphi: {}".format(phi_arr[1] - phi_arr[0]))
        print("  samples_per_file: {}".format(self.samples_per_file))

    @staticmethod
    def serialize_example_pyfunction(edges, points, fc_arr):
        edges_array = np.array(edges, dtype=np.float32)
        points = np.array(points, dtype=np.float32)
        fc_arr = np.array(fc_arr, dtype=np.float32)
        feature_ = {'points': _bytes_feature(tf.compat.as_bytes(points.tostring())),
                    'fc': _bytes_feature(tf.compat.as_bytes(fc_arr.tostring())),
                    'edges': _bytes_feature(tf.compat.as_bytes(edges_array.tostring()))}
        # Create an example protocol buffer
        return tf.train.Example(features=tf.train.Features(feature=feature_)).SerializeToString()

    def save_file_tf(self, filename):
        point_list = []
        edges_list = []
        for i in range(self.samples_per_file):
            points, edges = object_generator.generate_target_polygon(max_edges=self.max_edges, min_edges=self.min_edges,
                                                                     max_size=self.max_size)

            # print(points)
            padded_points = np.pad(points, pad_width=[(0, self.max_edges - points.shape[0]), (0, 0)],
                                   mode='edge').astype(np.float32)
            # for k in padded_points:
            #     print(k)
            point_list.append(padded_points)

            def one_hot(a, num_classes):
                return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

            edges_list.append(one_hot(edges - 1, self.max_edges))

        batch_points = np.stack(point_list)
        batch_edges = np.stack(edges_list)

        phi_tf = tf.expand_dims(tf.expand_dims(tf.constant(self.phi_arr, self._dtype), axis=0), axis=0)
        bc_dims = [int(self.samples_per_file), 1, len(self.phi_arr)]
        phi_tf_batch = tf.broadcast_to(phi_tf, bc_dims)
        fc_obj = c_layers.ScatterPolygon2D(phi_tf, dtype=self._dtype, with_batch_dim=True, epsilon=self.epsilon_tf,
                                           allow_variable_edges=True)
        fc_arr = fc_obj(batch_points)

        with tf.io.TFRecordWriter(filename) as writer:
            for i in range(self.samples_per_file):
                serialized_sample = self.serialize_example_pyfunction(
                    fc_arr=tf.concat((phi_tf_batch[0], fc_arr[i]), axis=0).numpy(),
                    points=batch_points[i],
                    edges=batch_edges[i])
                # Serialize to string and write on the file
                writer.write(serialized_sample)

        sys.stdout.flush()


class RegularPolygon2dSaver(object):
    def __init__(self, epsilon, phi_arr, samples_per_file, centered=False, max_edges=8, min_edges=3, max_size=50,
                 dtype=tf.float32):
        assert samples_per_file > 0
        self.epsilon = epsilon
        self.epsilon_tf = tf.constant(epsilon, dtype=dtype)
        self._centered = centered
        self.phi_arr = phi_arr
        self.dphi = np.abs(phi_arr[1] - phi_arr[0])
        self.samples_per_file = samples_per_file
        self.max_edges = max_edges
        self.min_edges = min_edges
        self.max_size = max_size
        self._dtype = dtype
        print("  init regular polygon2d-saver with:")
        print("  epsilon: {}".format(self.epsilon))
        print("  max edges of polygon: {}".format(self.max_edges))
        print("  min edges of polygon: {}".format(self.min_edges))
        print("  max radius of polygon: {}".format(self.max_size))
        print("  len phi_arr: {}".format(len(self.phi_arr)))
        print("  dphi: {}".format(phi_arr[1] - phi_arr[0]))
        print("  samples_per_file: {}".format(self.samples_per_file))

    @staticmethod
    def serialize_example_pyfunction(fc_arr, points, radius, rotation, translation, edges):
        # assert type(edges) == int, "edges-type is {}, but shoud be int".format(type(edges))
        edges_array = np.array(edges, dtype=np.float32)
        radius = np.array(radius, dtype=np.float32)
        rotation = np.array(rotation, dtype=np.float32)
        translation = np.array(translation, dtype=np.float32)
        fc_arr = np.array(fc_arr, dtype=np.float32)
        points = np.array(points, dtype=np.float32)
        if logger.level <= 10:
            for array in [edges_array, radius, rotation, translation, fc_arr, points]:
                logger.debug(array.shape)
        # Create a feature
        # print(edges_array.shape, points.shape, fc_arr.shape)
        # print(edges_array)
        feature_ = {'fc': _bytes_feature(tf.compat.as_bytes(fc_arr.tostring())),
                    'radius': _bytes_feature(tf.compat.as_bytes(radius.tostring())),
                    'points': _bytes_feature(tf.compat.as_bytes(points.tostring())),
                    'rotation': _bytes_feature(tf.compat.as_bytes(rotation.tostring())),
                    'translation': _bytes_feature(tf.compat.as_bytes(translation.tostring())),
                    'edges': _bytes_feature(tf.compat.as_bytes(edges_array.tostring()))}
        # Create an example protocol buffer
        return tf.train.Example(features=tf.train.Features(feature=feature_)).SerializeToString()

    def save_file_tf(self, filename):
        np.set_printoptions(precision=2, suppress=True)
        rre_dict = dict()
        point_list = []
        radius_list = []
        rotation_list = []
        translation_list = []
        edges_list = []
        rre_dict_list = []
        for i in range(self.samples_per_file):
            points, rre_dict = object_generator.generate_target_regular_polygon(max_edges=self.max_edges,
                                                                                max_radius=self.max_size,
                                                                                min_edges=self.min_edges,
                                                                                translation=not self._centered)

            # print(points)
            padded_points = np.pad(points, pad_width=[(0, self.max_edges - points.shape[0]), (0, 0)],
                                   mode='edge').astype(np.float32)
            # for k in padded_points:
            #     print(k)
            point_list.append(padded_points)
            radius_list.append(rre_dict["radius"])
            rotation_list.append(rre_dict["rotation"])
            translation_list.append(rre_dict["translation"])

            def one_hot(a, num_classes):
                return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

            edges_list.append(one_hot(rre_dict["edges"] - 1, self.max_edges))
            # rre_dict_list.append(rre_dict)
        batch_points = np.stack(point_list)
        batch_radius = np.stack(radius_list)
        batch_rotation = np.stack(rotation_list)
        batch_translation = np.stack(translation_list)
        batch_edges = np.stack(edges_list)

        phi_tf = tf.expand_dims(tf.expand_dims(tf.constant(self.phi_arr, self._dtype), axis=0), axis=0)
        bc_dims = [int(self.samples_per_file), 1, len(self.phi_arr)]
        phi_tf_batch = tf.broadcast_to(phi_tf, bc_dims)
        fc_obj = c_layers.ScatterPolygon2D(phi_tf, dtype=self._dtype, with_batch_dim=True, epsilon=self.epsilon_tf,
                                           allow_variable_edges=True)
        fc_arr = fc_obj(batch_points)

        with tf.io.TFRecordWriter(filename) as writer:
            for i in range(self.samples_per_file):
                serialized_sample = self.serialize_example_pyfunction(
                    fc_arr=tf.concat((phi_tf_batch[0], fc_arr[i]), axis=0).numpy(),
                    points=batch_points[i],
                    radius=batch_radius[i],
                    rotation=batch_rotation[i],
                    translation=batch_translation[i],
                    edges=batch_edges[i])
                # Serialize to string and write on the file
                writer.write(serialized_sample)

        sys.stdout.flush()

# class StarPolygon2dSaver(object):
#     def __init__(self, epsilon, phi_arr, samples_per_file, edges=3, max_size=30):
#         self.epsilon = epsilon
#         self.phi_arr = phi_arr
#         self.dphi = np.abs(phi_arr[1] - phi_arr[0])
#         self.samples_per_file = samples_per_file
#         self.edges = edges
#         self.max_size = max_size
#         print("  init regular polygon2d-saver with:")
#         print("  epsilon: {}".format(self.epsilon))
#         print("  edges of polygon: {}".format(self.edges))
#         print("  max radius* of polygon: {}".format(self.max_size))
#         print("  len phi_arr: {}".format(len(self.phi_arr)))
#         print("  dphi: {}".format(phi_arr[1] - phi_arr[0]))
#         print("  samples_per_file: {}".format(self.samples_per_file))
#
#     @staticmethod
#     def serialize_example_pyfunction(fc_arr, points_array, edges):
#         assert type(edges) == int, "edges-type is {}, but shoud be int".format(type(edges))
#         edges_array = np.array([edges], dtype=np.int32)
#         translation = np.array(points_array, dtype=np.float32)
#         fc_arr = np.array(fc_arr, dtype=np.float32)
#         # Create a feature
#         # print(edges_array.shape, points.shape, fc_arr.shape)
#         # print(edges_array)
#         feature_ = {'fc': _bytes_feature(tf.compat.as_bytes(fc_arr.tostring())),
#                     'points': _bytes_feature(tf.compat.as_bytes(translation.tostring())),
#                     'edges': _bytes_feature(tf.compat.as_bytes(edges_array.tostring()))}
#         # Create an example protocol buffer
#         return tf.train.Example(features=tf.train.Features(feature=feature_)).SerializeToString()
#
#     def save_file(self, filename):
#         # open the TFRecords file
#         if filename.endswith("0000000.tfr"):
#             t1 = time.time()
#         writer = tf.io.TFRecordWriter(filename)
#
#         for i in range(self.samples_per_file):
#             points, rre_dict = polygon2d.generate_target_regular_polygon(max_edges=self.max_edges,
#                                                                          max_radius=self.max_size)
#             fc_arr = polygon2d.Fcalculator(points, epsilon=np.array(0.0001)).F_of_phi(phi=self.phi_arr).astype(
#                 dtype=np.complex64)
#             fc_arr = np.stack((self.phi_arr, fc_arr.real, fc_arr.imag), axis=0).astype(np.float32)
#             serialized_sample = self.serialize_example_pyfunction(fc_arr=fc_arr,
#                                                                   radius=rre_dict["radius"],
#                                                                   rotation=rre_dict["rotation"],
#                                                                   translation=rre_dict["translation"],
#                                                                   edges=rre_dict["edges"])
#             # Serialize to string and write on the file
#             writer.write(serialized_sample)

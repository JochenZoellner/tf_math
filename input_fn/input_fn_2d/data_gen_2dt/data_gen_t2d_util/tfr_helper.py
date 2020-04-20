import sys
import time
import logging

import numpy as np
import tensorflow as tf

import input_fn.input_fn_2d.data_gen_2dt.data_gen_t2d_util.polygone_2d_helper as polygon2d
import input_fn.input_fn_2d.data_gen_2dt.data_gen_t2d_util.tf_polygon_2d_helper as tf_p2d
import input_fn.input_fn_2d.data_gen_2dt.data_gen_t2d_util.triangle_2d_helper as t2d
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


def parse_t2d(example_proto):
    feature_description = {'points': tf.io.FixedLenFeature([], tf.string), 'fc': tf.io.FixedLenFeature([], tf.string)}
    # Parse the input tf.Example proto using the dictionary above.
    raw_dict = tf.io.parse_single_example(example_proto, feature_description)
    decoded_dict = ({"fc": tf.reshape(tf.compat.v1.decode_raw(raw_dict["fc"], out_type=tf.float32), (3, -1))},
                    {"points": tf.reshape(tf.compat.v1.decode_raw(raw_dict["points"], out_type=tf.float32), (3, -1))})

    return decoded_dict


def parse_t2d_phi_complex(example_proto):
    feature_description = {'points': tf.io.FixedLenFeature([], tf.string), 'fc': tf.io.FixedLenFeature([], tf.string)}
    # Parse the input tf.Example proto using the dictionary above.
    raw_dict = tf.io.parse_single_example(example_proto, feature_description)
    decoded_dict = ({"fc": tf.reshape(tf.compat.v1.decode_raw(raw_dict["fc"], out_type=tf.float32), (4, -1))},
                    {"points": tf.reshape(tf.compat.v1.decode_raw(raw_dict["points"], out_type=tf.float32), (3, -1))})

    return decoded_dict


def parse_polygon2d(example_proto):
    feature_description = {'fc': tf.io.FixedLenFeature([], tf.string),
                           'points': tf.io.FixedLenFeature([], tf.string),
                           'edges': tf.io.FixedLenFeature([], tf.string)}
    # Parse the input tf.Example proto using the dictionary above.
    raw_dict = tf.io.parse_single_example(example_proto, feature_description)
    # print(tf.compat.v1.decode_raw(raw_dict["edges"], out_type=tf.int32))
    decoded_dict = ({"fc": tf.reshape(tf.compat.v1.decode_raw(raw_dict["fc"], out_type=tf.float32), (3, -1))},
                    {"points": tf.reshape(tf.compat.v1.decode_raw(raw_dict["points"], out_type=tf.float32), (-1, 2)),
                     "edges": tf.reshape(tf.compat.v1.decode_raw(raw_dict["edges"], out_type=tf.int32), (1,))})

    return decoded_dict


def parse_regular_polygon2d(example_proto, batch_size=1000):
    """
    intput is:
        fc (fourier coefficients) [phi,real_part, imag_part] x [phi_0, ..., phi_n] shape: 3 x len(phi_array)
    target is:
        radius [1] float32
        rotation [1] float32
        translation [2] float32
        edges [1] int32
            """
    feature_description = {'fc': tf.io.FixedLenFeature([], tf.string),
                           'radius': tf.io.FixedLenFeature([], tf.string),
                           'rotation': tf.io.FixedLenFeature([], tf.string),
                           'translation': tf.io.FixedLenFeature([], tf.string),
                           'edges': tf.io.FixedLenFeature([], tf.string),
                           'points': tf.io.FixedLenFeature([], tf.string)}
    # Parse the input tf.Example proto using the dictionary above.
    raw_dict = tf.io.parse_single_example(example_proto, feature_description)
    # print(tf.compat.v1.decode_raw(raw_dict["edges"], out_type=tf.int32))
    decoded_dict = ({"fc": tf.reshape(tf.compat.v1.decode_raw(raw_dict["fc"], out_type=tf.float32), (batch_size, 3, -1))},
                    {"radius": tf.reshape(tf.compat.v1.decode_raw(raw_dict["radius"], out_type=tf.float32), (batch_size, 1,)),
                     "rotation": tf.reshape(tf.compat.v1.decode_raw(raw_dict["rotation"], out_type=tf.float32), (batch_size, 1,)),
                     "translation": tf.reshape(tf.compat.v1.decode_raw(raw_dict["translation"], out_type=tf.float32),(batch_size, 2,)),
                     "edges": tf.reshape(tf.compat.v1.decode_raw(raw_dict["edges"], out_type=tf.int32), (batch_size, 1,)),
                     "points": tf.reshape(tf.compat.v1.decode_raw(raw_dict["points"], out_type=tf.float32), (batch_size, -1, 2))})

    return decoded_dict


class Triangle2dSaver(object):
    def __init__(self, epsilon, phi_arr, x_sorted, samples_per_file, dphi=0.001, complex_phi=False, centered=False):
        self.epsilon = epsilon
        self.complex_phi = complex_phi  # True if a and b set via complex value of phi-pyhton-variable
        self.centered = centered
        self.dphi = dphi
        self.phi_arr = phi_arr
        self.x_sorted = x_sorted
        self.samples_per_file = samples_per_file
        print("  init t2d-saver with:")
        print("  epsilon: {}".format(self.epsilon))
        print("  len phi_arr: {}".format(len(self.phi_arr)))
        print("  phi_complex: {}".format(self.complex_phi))
        if self.complex_phi:
            print("  phi_arr:\n{}".format(phi_arr))
        print("  dphi: {}".format(phi_arr[1] - phi_arr[0]))
        print("  x_sorted: {}".format(self.x_sorted))
        print("  samples_per_file: {}".format(self.samples_per_file))

    @staticmethod
    def serialize_example_pyfunction(points, fc_arr):
        # Create a feature
        feature_ = {'points': _bytes_feature(tf.compat.as_bytes(points.tostring())),
                    'fc': _bytes_feature(tf.compat.as_bytes(fc_arr.tostring()))}
        # Create an example protocol buffer
        return tf.train.Example(features=tf.train.Features(feature=feature_)).SerializeToString()

    def save_file(self, filename):
        # open the TFRecords file
        writer = tf.io.TFRecordWriter(filename)

        for i in range(self.samples_per_file):
            points = t2d.generate_target(x_sorted=self.x_sorted, center_of_weight=self.centered)

            fc_arr = t2d.make_scatter_data(points, epsilon=self.epsilon, phi_arr=self.phi_arr, dphi=self.dphi,
                                           complex_phi=self.complex_phi)
            serialized_sample = self.serialize_example_pyfunction(points, fc_arr)
            # Serialize to string and write on the file
            writer.write(serialized_sample)

        writer.close()
        sys.stdout.flush()

    def save_file_tf(self, filename):
        # open the TFRecords file
        D_TYPE = tf.float32
        phi_tf = tf.expand_dims(tf.expand_dims(tf.constant(self.phi_arr, D_TYPE), axis=0), axis=0)

        fc_obj = c_layers.ScatterPolygonTF(phi_tf, dtype=D_TYPE, with_batch_dim=True)
        point_list = []
        for i in range(self.samples_per_file):
            points = t2d.generate_target(x_sorted=self.x_sorted)
            point_list.append(points)

        batch_points = np.stack(point_list)
        batch_points = tf_p2d.make_positiv_orientation(batch_points).numpy()
        fc_arr = fc_obj(batch_points)

        with tf.io.TFRecordWriter(filename) as writer:
            for i in range(self.samples_per_file):
                # fc_arr = tf.concat((phi_tf[0], fc_arr[i]), axis=0)
                serialized_sample = self.serialize_example_pyfunction(batch_points[i], fc_arr=tf.concat((phi_tf[0], fc_arr[i]), axis=0).numpy())
                # Serialize to string and write on the file

                writer.write(serialized_sample)

        sys.stdout.flush()


class Polygon2dSaver(object):
    def __init__(self, epsilon, phi_arr, samples_per_file, max_edges=6, max_size=50):
        self.epsilon = epsilon
        self.phi_arr = phi_arr
        self.dphi = np.abs(phi_arr[1] - phi_arr[0])
        self.samples_per_file = samples_per_file
        self.max_edges = max_edges
        self.max_size = max_size
        print("  init polygon2d-saver with:")
        print("  epsilon: {}".format(self.epsilon))
        print("  max edges of polygon: {}".format(self.max_edges))
        print("  max size of polygon: {}".format(self.max_size))
        print("  len phi_arr: {}".format(len(self.phi_arr)))
        print("  dphi: {}".format(phi_arr[1] - phi_arr[0]))
        print("  samples_per_file: {}".format(self.samples_per_file))

    @staticmethod
    def serialize_example_pyfunction(edges, points, fc_arr):
        assert type(edges) == int
        edges_array = np.array([edges], dtype=np.int32)
        points = np.array(points, dtype=np.float32)
        fc_arr = np.array(fc_arr, dtype=np.float32)
        feature_ = {'points': _bytes_feature(tf.compat.as_bytes(points.tostring())),
                    'fc': _bytes_feature(tf.compat.as_bytes(fc_arr.tostring())),
                    'edges': _bytes_feature(tf.compat.as_bytes(edges_array.tostring()))}
        # Create an example protocol buffer
        return tf.train.Example(features=tf.train.Features(feature=feature_)).SerializeToString()

    def save_file(self, filename):
        if filename.endswith("0000000.tfr"):
            t1 = time.time()
        writer = tf.io.TFRecordWriter(filename)

        for i in range(self.samples_per_file):
            points = polygon2d.generate_target_polygon(max_edge=self.max_edges, max_size=self.max_size)
            edges = points.shape[0]
            fc_arr = polygon2d.Fcalculator(points, epsilon=np.array(0.0001)).F_of_phi(phi=self.phi_arr).astype(
                dtype=np.complex64)
            fc_arr = np.stack((self.phi_arr, fc_arr.real, fc_arr.imag), axis=0).astype(np.float32)
            serialized_sample = self.serialize_example_pyfunction(edges=edges, points=points, fc_arr=fc_arr)
            # Serialize to string and write on the file
            writer.write(serialized_sample)

        writer.close()
        sys.stdout.flush()
        if filename.endswith("0000000.tfr"):
            print("time for one file@1thread: {}".format(time.time() - t1))


class RegularPolygon2dSaver(object):
    def __init__(self, epsilon, phi_arr, samples_per_file, centered=False, max_edges=8, max_size=50, dtype=tf.float32):
        assert samples_per_file > 0
        self.epsilon = epsilon
        self._centered = centered
        self.phi_arr = phi_arr
        self.dphi = np.abs(phi_arr[1] - phi_arr[0])
        self.samples_per_file = samples_per_file
        self.max_edges = max_edges
        self.max_size = max_size
        self._dtype = dtype
        print("  init regular polygon2d-saver with:")
        print("  epsilon: {}".format(self.epsilon))
        print("  max edges of polygon: {}".format(self.max_edges))
        print("  max radius of polygon: {}".format(self.max_size))
        print("  len phi_arr: {}".format(len(self.phi_arr)))
        print("  dphi: {}".format(phi_arr[1] - phi_arr[0]))
        print("  samples_per_file: {}".format(self.samples_per_file))

    @staticmethod
    def serialize_example_pyfunction(fc_arr, points, radius, rotation, translation, edges):
        # assert type(edges) == int, "edges-type is {}, but shoud be int".format(type(edges))
        edges_array = np.array(edges, dtype=np.int32)
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
                    'points':_bytes_feature(tf.compat.as_bytes(points.tostring())),
                    'rotation': _bytes_feature(tf.compat.as_bytes(rotation.tostring())),
                    'translation': _bytes_feature(tf.compat.as_bytes(translation.tostring())),
                    'edges': _bytes_feature(tf.compat.as_bytes(edges_array.tostring()))}
        # Create an example protocol buffer
        return tf.train.Example(features=tf.train.Features(feature=feature_)).SerializeToString()

    def save_file_tf(self, filename):

        rre_dict = dict()
        point_list = []
        radius_list = []
        rotation_list = []
        translation_list = []
        edges_list = []
        rre_dict_list = []
        for i in range(self.samples_per_file):
            points, rre_dict = polygon2d.generate_target_regular_polygon(max_edges=self.max_edges,
                                                                         max_radius=self.max_size,
                                                                         min_edges=self.max_edges,
                                                                         translation=not self._centered)
            point_list.append(points)
            radius_list.append(rre_dict["radius"])
            rotation_list.append(rre_dict["rotation"])
            translation_list.append(rre_dict["translation"])
            edges_list.append(rre_dict["edges"])
            # rre_dict_list.append(rre_dict)

        batch_points = np.stack(point_list)
        batch_radius = np.stack(radius_list)
        batch_rotation = np.stack(rotation_list)
        batch_translation = np.stack(translation_list)
        batch_edges = np.stack(edges_list)

        phi_tf = tf.expand_dims(tf.expand_dims(tf.constant(self.phi_arr, self._dtype), axis=0), axis=0)
        bc_dims = [int(self.samples_per_file), 1, len(self.phi_arr)]
        phi_tf_batch = tf.broadcast_to(phi_tf, bc_dims)
        fc_obj = c_layers.ScatterPolygonTF(phi_tf, dtype=self._dtype, with_batch_dim=True)
        fc_arr = fc_obj(batch_points)

        with tf.io.TFRecordWriter(filename) as writer:

            serialized_sample = self.serialize_example_pyfunction(fc_arr=tf.concat((phi_tf_batch, fc_arr), axis=1).numpy(),
                                                                      points=batch_points,
                                                                      radius=batch_radius,
                                                                      rotation=batch_rotation,
                                                                      translation=batch_translation,
                                                                      edges=batch_edges)
            # Serialize to string and write on the file
            writer.write(serialized_sample)

        sys.stdout.flush()


class StarPolygon2dSaver(object):
    def __init__(self, epsilon, phi_arr, samples_per_file, edges=3, max_size=30):
        self.epsilon = epsilon
        self.phi_arr = phi_arr
        self.dphi = np.abs(phi_arr[1] - phi_arr[0])
        self.samples_per_file = samples_per_file
        self.edges = edges
        self.max_size = max_size
        print("  init regular polygon2d-saver with:")
        print("  epsilon: {}".format(self.epsilon))
        print("  edges of polygon: {}".format(self.edges))
        print("  max radius* of polygon: {}".format(self.max_size))
        print("  len phi_arr: {}".format(len(self.phi_arr)))
        print("  dphi: {}".format(phi_arr[1] - phi_arr[0]))
        print("  samples_per_file: {}".format(self.samples_per_file))

    @staticmethod
    def serialize_example_pyfunction(fc_arr, points_array, edges):
        assert type(edges) == int, "edges-type is {}, but shoud be int".format(type(edges))
        edges_array = np.array([edges], dtype=np.int32)
        translation = np.array(points_array, dtype=np.float32)
        fc_arr = np.array(fc_arr, dtype=np.float32)
        # Create a feature
        # print(edges_array.shape, points.shape, fc_arr.shape)
        # print(edges_array)
        feature_ = {'fc': _bytes_feature(tf.compat.as_bytes(fc_arr.tostring())),
                    'points': _bytes_feature(tf.compat.as_bytes(translation.tostring())),
                    'edges': _bytes_feature(tf.compat.as_bytes(edges_array.tostring()))}
        # Create an example protocol buffer
        return tf.train.Example(features=tf.train.Features(feature=feature_)).SerializeToString()

    def save_file(self, filename):
        # open the TFRecords file
        if filename.endswith("0000000.tfr"):
            t1 = time.time()
        writer = tf.io.TFRecordWriter(filename)

        for i in range(self.samples_per_file):
            points, rre_dict = polygon2d.generate_target_regular_polygon(max_edges=self.max_edges,
                                                                         max_radius=self.max_size)
            fc_arr = polygon2d.Fcalculator(points, epsilon=np.array(0.0001)).F_of_phi(phi=self.phi_arr).astype(
                dtype=np.complex64)
            fc_arr = np.stack((self.phi_arr, fc_arr.real, fc_arr.imag), axis=0).astype(np.float32)
            serialized_sample = self.serialize_example_pyfunction(fc_arr=fc_arr,
                                                                  radius=rre_dict["radius"],
                                                                  rotation=rre_dict["rotation"],
                                                                  translation=rre_dict["translation"],
                                                                  edges=rre_dict["edges"])
            # Serialize to string and write on the file
            writer.write(serialized_sample)

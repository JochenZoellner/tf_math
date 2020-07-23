import tensorflow as tf


class InterfaceBase2D(object):
    def __init__(self, **kwargs):
        super(InterfaceRegularPolygon2D).__init__()
        self._shape_tuple = None
        self._type_tuple = None
        self._padding_tuple = None

    def get_shape_tuple(self):
        return self._shape_tuple

    def get_padding_tuple(self):
        return self._padding_tuple

    # def parse_proto(self, example_proto):
    #     assert self._shape_tuple
    #     feature_description = {}
    #     for idx, shape_dict in enumerate(self._shape_tuple):
    #         for key in shape_dict:
    #             feature_description[key] = tf.io.FixedLenFeature([], tf.string)
    #     # Parse the input tf.Example proto using the dictionary above.
    #     raw_dict = tf.io.parse_single_example(example_proto, feature_description)
    #     decoded_dict_tuple = ({}, {})
    #     for idx, decoded_dict in enumerate(self._shape_tuple):
    #         for key in decoded_dict:
    #             decoded_object = tf.compat.v1.decode_raw(raw_dict[key], out_type=self._type_tuple[idx][key])
    #             shape_tuple = [x if x != None else -1 for x in self._shape_tuple[idx][key]]
    #             decoded_dict_tuple[idx][key] = tf.reshape(decoded_object, (3, -1))
    #
    #     decoded_dict = ({"fc": tf.reshape(tf.compat.v1.decode_raw(raw_dict["fc"], out_type=tf.float32), (3, -1))},
    #                     {"radius": tf.reshape(tf.compat.v1.decode_raw(raw_dict["radius"], out_type=tf.float32), (1,)),
    #                      "rotation": tf.reshape(tf.compat.v1.decode_raw(raw_dict["rotation"], out_type=tf.float32), (1,)),
    #                      "translation": tf.reshape(tf.compat.v1.decode_raw(raw_dict["translation"], out_type=tf.float32), (2,)),
    #                      "edges": tf.reshape(tf.compat.v1.decode_raw(raw_dict["edges"], out_type=tf.float32), (self.max_edges,)),
    #                      "points": tf.reshape(tf.compat.v1.decode_raw(raw_dict["points"], out_type=tf.float32), (-1, 2))})
    #
    #     return decoded_dict_tuple


class InterfaceTriangle2D(InterfaceBase2D):
    def __init__(self, **kwargs):
        super(InterfaceTriangle2D).__init__()
        self._shape_tuple = ({'fc': [3, None]},
                             {'points': [3, 2]})
        self._padding_tuple = ({'fc': tf.constant(0, dtype=tf.float32)},
                               {'points': tf.constant(0, dtype=tf.float32)})

    def parse_proto(self, example_proto):
        """
        intput is:
            fc (fourier coefficients) [phi,real_part, imag_part] x [phi_0, ..., phi_n] shape: 3 x len(phi_array)
        target is:
            radius [1] float32
            rotation [1] float32
            translation [2] float32
            edges [max_edges] float32
            points [max_edges, 2] float32
                """
        feature_description = {'fc': tf.io.FixedLenFeature([], tf.string),
                               'points': tf.io.FixedLenFeature([], tf.string)}
        # Parse the input tf.Example proto using the dictionary above.
        raw_dict = tf.io.parse_single_example(example_proto, feature_description)
        # print(tf.compat.v1.decode_raw(raw_dict["edges"], out_type=tf.int32))
        decoded_dict = ({"fc": tf.reshape(tf.compat.v1.decode_raw(raw_dict["fc"], out_type=tf.float32), (3, -1))},
                        {"points": tf.reshape(tf.compat.v1.decode_raw(raw_dict["points"], out_type=tf.float32),
                                              (-1, 2))})

        return decoded_dict


class InterfaceRegularPolygon2D(InterfaceBase2D):
    def __init__(self, max_edges):
        super(InterfaceRegularPolygon2D).__init__()
        self.max_edges = max_edges
        self._shape_tuple = ({'fc': [3, None]},
                             {'radius': [1],
                              'rotation': [1],
                              'translation': [2],
                              'edges': [self.max_edges],
                              'points': [self.max_edges, 2]})
        self._padding_tuple = ({'fc': tf.constant(0, dtype=tf.float32)},
                               {'radius': tf.constant(0, dtype=tf.float32),
                                'rotation': tf.constant(0, dtype=tf.float32),
                                'translation': tf.constant(0, dtype=tf.float32),
                                'edges': tf.constant(0, dtype=tf.float32),
                                'points': tf.constant(0, dtype=tf.float32)})

    def parse_proto(self, example_proto):
        """
        intput is:
            fc (fourier coefficients) [phi,real_part, imag_part] x [phi_0, ..., phi_n] shape: 3 x len(phi_array)
        target is:
            radius [1] float32
            rotation [1] float32
            translation [2] float32
            edges [max_edges] float32
            points [max_edges, 2] float32
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
        decoded_dict = ({"fc": tf.reshape(tf.compat.v1.decode_raw(raw_dict["fc"], out_type=tf.float32), (3, -1))},
                        {"radius": tf.reshape(tf.compat.v1.decode_raw(raw_dict["radius"], out_type=tf.float32), (1,)),
                         "rotation": tf.reshape(tf.compat.v1.decode_raw(raw_dict["rotation"], out_type=tf.float32),
                                                (1,)),
                         "translation": tf.reshape(
                             tf.compat.v1.decode_raw(raw_dict["translation"], out_type=tf.float32), (2,)),
                         "edges": tf.reshape(tf.compat.v1.decode_raw(raw_dict["edges"], out_type=tf.float32),
                                             (self.max_edges,)),
                         "points": tf.reshape(tf.compat.v1.decode_raw(raw_dict["points"], out_type=tf.float32),
                                              (-1, 2))})

        return decoded_dict


class InterfaceArbitraryPolygon2D(InterfaceBase2D):
    def __init__(self, max_edges):
        super(InterfaceArbitraryPolygon2D).__init__()
        self.max_edges = max_edges
        self._shape_tuple = ({'fc': [3, None]},
                             {'edges': [self.max_edges],
                              'points': [self.max_edges, 2]})
        self._padding_tuple = ({'fc': tf.constant(0, dtype=tf.float32)},
                               {'edges': tf.constant(0, dtype=tf.float32),
                                'points': tf.constant(0, dtype=tf.float32)})

    def parse_proto(self, example_proto):
        """
        intput is:
            fc (fourier coefficients) [phi,real_part, imag_part] x [phi_0, ..., phi_n] shape: 3 x len(phi_array)
        target is:
            edges [max_edges] float32
            points [max_edges, 2] float32
                """
        feature_description = {'fc': tf.io.FixedLenFeature([], tf.string),
                               'edges': tf.io.FixedLenFeature([], tf.string),
                               'points': tf.io.FixedLenFeature([], tf.string)}
        # Parse the input tf.Example proto using the dictionary above.
        raw_dict = tf.io.parse_single_example(example_proto, feature_description)
        # print(tf.compat.v1.decode_raw(raw_dict["edges"], out_type=tf.int32))
        decoded_dict = ({"fc": tf.reshape(tf.compat.v1.decode_raw(raw_dict["fc"], out_type=tf.float32), (3, -1))},
                        {"edges": tf.reshape(tf.compat.v1.decode_raw(raw_dict["edges"], out_type=tf.float32),
                                             (self.max_edges,)),
                         "points": tf.reshape(tf.compat.v1.decode_raw(raw_dict["points"], out_type=tf.float32),
                                              (-1, 2))})

        return decoded_dict

# def parse_t2d(example_proto):
#     feature_description = {'points': tf.io.FixedLenFeature([], tf.string), 'fc': tf.io.FixedLenFeature([], tf.string)}
#     # Parse the input tf.Example proto using the dictionary above.
#     raw_dict = tf.io.parse_single_example(example_proto, feature_description)
#     decoded_dict = ({"fc": tf.reshape(tf.compat.v1.decode_raw(raw_dict["fc"], out_type=tf.float32), (3, -1))},
#                     {"points": tf.reshape(tf.compat.v1.decode_raw(raw_dict["points"], out_type=tf.float32), (3, -1))})
#
#     return decoded_dict
#
#
# def parse_t2d_phi_complex(example_proto):
#     feature_description = {'points': tf.io.FixedLenFeature([], tf.string), 'fc': tf.io.FixedLenFeature([], tf.string)}
#     # Parse the input tf.Example proto using the dictionary above.
#     raw_dict = tf.io.parse_single_example(example_proto, feature_description)
#     decoded_dict = ({"fc": tf.reshape(tf.compat.v1.decode_raw(raw_dict["fc"], out_type=tf.float32), (4, -1))},
#                     {"points": tf.reshape(tf.compat.v1.decode_raw(raw_dict["points"], out_type=tf.float32), (3, -1))})
#
#     return decoded_dict
#
#
# def parse_polygon2d(example_proto):
#     feature_description = {'fc': tf.io.FixedLenFeature([], tf.string),
#                            'points': tf.io.FixedLenFeature([], tf.string),
#                            'edges': tf.io.FixedLenFeature([], tf.string)}
#     # Parse the input tf.Example proto using the dictionary above.
#     raw_dict = tf.io.parse_single_example(example_proto, feature_description)
#     # print(tf.compat.v1.decode_raw(raw_dict["edges"], out_type=tf.int32))
#     decoded_dict = ({"fc": tf.reshape(tf.compat.v1.decode_raw(raw_dict["fc"], out_type=tf.float32), (3, -1))},
#                     {"points": tf.reshape(tf.compat.v1.decode_raw(raw_dict["points"], out_type=tf.float32), (-1, 2)),
#                      "edges": tf.reshape(tf.compat.v1.decode_raw(raw_dict["edges"], out_type=tf.int32), (1,))})
#
#     return decoded_dict

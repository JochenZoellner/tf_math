import numpy as np
import logging
import multiprocessing
import tensorflow as tf
from input_fn.input_fn_generator_base import InputFnBase
from input_fn.input_fn_2d.data_gen_2dt.util_2d import interface, object_generator, misc_tf
import model_fn.util_model_fn.custom_layers as c_layers
from input_fn.input_fn_2d.input_fn_2d_util import phi_array_open_symetric_no90
from util.flags import update_params

logger = logging.getLogger(__name__)


class InputFn2D(InputFnBase):
    """Input Function Generator for regular polygon 2d problems, dataset returns a dict..."""

    def __init__(self, flags_):
        super(InputFn2D, self).__init__(flags_)
        self.iterator = None
        self._next_batch = None
        self.dataset = None
        self._interface_obj = None
        self._input_params["min_fov"] = 0.0
        self._input_params["max_fov"] = 180.0

    def cut_phi_batch(self, batch, min_fov=None, max_fov=None):
        """
        :param batch: array [batch, (phi, real, imag), phi_vec_len]
        :param phi_vec: 1D-array
        :param min_fov: full field of view cut around pi/2
        :param max_fov: full field of view keept (min_fov < max_fov < 180) for half-room
        :return:
        """
        if not min_fov:
            min_fov = float(self._input_params["min_fov"])

        if not max_fov:
            max_fov = float(self._input_params["max_fov"])
        phi_vec = batch["fc"][0, 0, :]
        max_fov = max_fov / 180.0 * np.pi  # max_angle_of_view_cut_rad
        min_fov = min_fov / 180.0 * np.pi  # hole
        outer_cut = (np.pi - max_fov) / 2.0
        inner_cut = min_fov / 2.0
        lower_block = tf.logical_and(phi_vec >= outer_cut, phi_vec < (np.pi / 2.0 - inner_cut))
        upper_block = tf.logical_and(phi_vec >= np.pi / 2.0 + inner_cut, phi_vec < (np.pi - outer_cut))
        both_blocks = tf.logical_or(lower_block, upper_block)
        batch["fc"] = tf.where(both_blocks, batch["fc"], tf.zeros_like(batch["fc"]))
        # batch["fc"] = tf.concat((batch["fc"][:, :1], mask_batch[:, 1:]), axis=1)

        return batch

    def tf_cut_phi_batch(self, feature_dict, target_dict):
        return self.cut_phi_batch(feature_dict), target_dict

    def map_and_batch(self, raw_dataset, batch_size):
        parsed_dataset = raw_dataset.map(self._interface_obj.parse_proto)
        parsed_dataset_batched = parsed_dataset.batch(batch_size)
        parsed_dataset_batched = parsed_dataset_batched.map(self.tf_cut_phi_batch)
        return parsed_dataset_batched

    def get_input_fn_train(self):
        # One instance of train dataset to produce infinite many samples
        assert len(self._flags.train_lists) == 1, "exact one train list is needed for this scenario"
        with open(self._flags.train_lists[0], "r") as tr_fobj:
            train_filepath_list = [x.strip("\n") for x in tr_fobj.readlines()]
        raw_dataset = tf.data.TFRecordDataset(train_filepath_list)
        parsed_dataset_batched = self.map_and_batch(raw_dataset, self._flags.train_batch_size)
        parsed_dataset_batched = parsed_dataset_batched.shuffle(buffer_size=self._flags.train_batch_size * 10)

        self.dataset = parsed_dataset_batched.repeat()
        return self.dataset.prefetch(5)

    def get_input_fn_val(self):
        with open(self._flags.val_list, "r") as tr_fobj:
            train_filepath_list = [x.strip("\n") for x in tr_fobj.readlines()]
        raw_dataset = tf.data.TFRecordDataset(train_filepath_list)

        parsed_dataset_batched = self.map_and_batch(raw_dataset, self._flags.val_batch_size)

        return parsed_dataset_batched

    def get_input_fn_file(self, filepath, batch_size=1):
        """input of just one file for use in predict mode to guarantee order"""
        assert type(filepath) is str
        raw_dataset = tf.data.TFRecordDataset(filepath)
        parsed_dataset_batched = self.map_and_batch(raw_dataset, self._flags.val_batch_size)

        return parsed_dataset_batched


class InputFnRegularPolygon2D(InputFn2D):
    """Input Function Generator for regular polygon 2d problems, dataset returns a dict..."""

    def __init__(self, flags_):
        super(InputFnRegularPolygon2D, self).__init__(flags_)
        self._interface_obj = interface.InterfaceRegularPolygon2D(max_edges=self._flags.max_edges)


class InputFnArbitraryPolygon2D(InputFn2D):
    """Input Function Generator for polygon 2d problems, dataset returns a dict..."""
    def __init__(self, flags_):
        super(InputFnArbitraryPolygon2D, self).__init__(flags_)
        self._interface_obj = interface.InterfaceArbitraryPolygon2D(max_edges=self._flags.max_edges)
        

class TriangleWorker(object):
    def __init__(self, x_sorted=True, center=False):
        self._x_sorted = x_sorted
        self._center = center

    def work(self, i):
        points = object_generator.generate_target_triangle(x_sorted=self._x_sorted, center=self._center)
        return points


class InputFnTriangle2D(InputFn2D):
    """Input Function Generator for 2d triangle problems,  dataset returns a dict..."""

    def __init__(self, flags_):
        super(InputFnTriangle2D, self).__init__(flags_)
        self._dphi = 0.01
        self._val_infinity = False
        self._batch_size = None

        logger.info("Set input_fn_params")
        self._input_params["centered"] = False

        # Updating of the default params if provided via flags as a dict
        self._input_params = update_params(self._input_params, self._flags.input_params, "input")

        self._interface_obj = interface.InterfaceTriangle2D()
        self._fc_obj = None
        self._phi_batch = None

    def make_batch(self):
        amount = self._batch_size * 10

        triangle_worker = TriangleWorker()
        with multiprocessing.Pool(min(4, multiprocessing.cpu_count() // 2)) as pool:
        # with multiprocessing.Pool(1) as pool:
            point_list_ = pool.map(triangle_worker.work, range(amount))

        batch_points_ = np.stack(point_list_)
        batch_points_ = misc_tf.make_spin_positive(batch_points_).numpy()
        fc_arr = self._fc_obj(batch_points_)
        fc_batch_ = tf.concat((self._phi_batch, fc_arr), axis=1)
        return fc_batch_, batch_points_

    def batch_generator(self):
        _phi_arr = phi_array_open_symetric_no90(delta_phi=self._dphi)
        D_TYPE = tf.float32
        phi_tf = tf.expand_dims(tf.expand_dims(tf.constant(_phi_arr, D_TYPE), axis=0), axis=0)
        if self._fc_obj is None:
            self._fc_obj = c_layers.ScatterPolygon2D(phi_tf, dtype=D_TYPE, with_batch_dim=True)
        if self._phi_batch is None:
            self._phi_batch = np.broadcast_to(np.expand_dims(_phi_arr, axis=0),
                                              (self._batch_size * 10, 1, _phi_arr.shape[0]))

        fc_batch = None
        batch_points = None
        index = self._batch_size * 10 - 2
        while True:
            index += 1
            if index >= self._batch_size * 10 - 1:
                fc_batch, batch_points = self.make_batch()
                index = 0
            yield {"fc": tf.cast(fc_batch[index], dtype=tf.float32)}, \
                  {"points": tf.cast(batch_points[index], dtype=tf.float32)}

    def get_input_fn_train(self):
        # One instance of train dataset to produce infinite many samples
        if self._val_infinity:
            self._batch_size = self._flags.val_batch_size
        else:
            self._batch_size = self._flags.train_batch_size
        if self._val_infinity or "infinity" in self._flags.train_lists[0]:
            parsed_dataset = tf.data.Dataset.from_generator(self.batch_generator,
                                                                    output_types=self._interface_obj.get_type_tuple(),
                                                                    output_shapes=self._interface_obj.get_shape_tuple())
            parsed_dataset_batched = parsed_dataset.batch(self._batch_size)
            parsed_dataset_batched = parsed_dataset_batched.map(self.tf_cut_phi_batch, num_parallel_calls=4)
            return parsed_dataset_batched.prefetch(2)
        else:
            return super(InputFnTriangle2D, self).get_input_fn_train()


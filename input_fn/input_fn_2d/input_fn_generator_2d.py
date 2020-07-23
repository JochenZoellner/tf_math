import tensorflow as tf
import numpy as np
import input_fn.input_fn_2d.data_gen_2dt.util_2d.interface as interface
from input_fn.input_fn_generator_base import InputFnBase


class InputFn2D(InputFnBase):
    """Input Function Generator for regular polygon 2d problems, dataset returns a dict..."""

    def __init__(self, flags_):
        super(InputFn2D, self).__init__(flags_)
        self.iterator = None
        self._next_batch = None
        self.dataset = None
        self._interface_obj = None

    def cut_phi_batch(self, batch, min_fov=None, max_fov=None):
        """
        :param batch: array [batch, (phi, real, imag), phi_vec_len]
        :param phi_vec: 1D-array
        :param min_fov: full field of view cut around pi/2
        :param max_fov: full field of view keept (min_fov < max_fov < 180) for half-room
        :return:
        """
        if not min_fov:
            min_fov = self._input_params["min_fov"]

        if not max_fov:
            max_fov = self._input_params["max_fov"]
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

    # @tf.function(input_signature=[tf.TensorSpec([None, None, None], tf.float32)])
    def tf_cut_phi_batch(self, feature_dict, target_dict):
        return self.cut_phi_batch(feature_dict), target_dict

    def map_and_batch(self, raw_dataset, batch_size):
        parsed_dataset = raw_dataset.map(self._interface_obj.parse_proto)
        parsed_dataset_batched = parsed_dataset.batch(batch_size)
        return parsed_dataset_batched

    def get_input_fn_train(self):
        # One instance of train dataset to produce infinite many samples
        assert len(self._flags.train_lists) == 1, "exact one train list is needed for this scenario"
        with open(self._flags.train_lists[0], "r") as tr_fobj:
            train_filepath_list = [x.strip("\n") for x in tr_fobj.readlines()]
        raw_dataset = tf.data.TFRecordDataset(train_filepath_list)
        parsed_dataset_batched = self.map_and_batch(raw_dataset, self._flags.train_batch_size)
        # parsed_dataset = parsed_dataset.shuffle(buffer_size=1000)

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

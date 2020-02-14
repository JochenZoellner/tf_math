import logging
import tensorflow as tf
import numpy as np

import input_fn.input_fn_2d.data_gen_2dt.data_gen_t2d_util.tfr_helper as tfr_helper
from input_fn.input_fn_generator_base import InputFnBase

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

class InputFn2DT(InputFnBase):
    """Input Function Generator for 2d triangle problems,  dataset returns a dict..."""

    def __init__(self, flags):
        super(InputFn2DT, self).__init__()
        self._flags = flags
        self.iterator = None
        self._next_batch = None
        self.dataset = None
        self._input_params = None
        if self._flags.hasKey("input_fn_params"):
            logger.info("Set input_fn_params")
            self._input_params = self._flags.input_fn_params
            self._min_fov = self._input_params["min_fov"]
            self._max_fov = self._input_params["max_fov"]

    def cut_phi_batch(self, batch):
        """

        :param batch: array [batch, (phi, real, imag), phi_vec_len]
        :param phi_vec: 1D-array
        :param min_fov: full field of view cut around pi/2
        :param max_fov: full field of view keept (min_fov < max_fov < 180) for half-room
        :return:
        """
        phi_vec = batch["fc"][0, 0, :]
        max_fov = self._max_fov / 180.0 * np.pi  # max_angle_of_view_cut_rad
        min_fov = self._min_fov / 180.0 * np.pi  # hole
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

    def get_input_fn_train(self):
        # One instance of train dataset to produce infinite many samples
        assert len(self._flags.train_lists) == 1, "exact one train list is needed for this scenario"

        with open(self._flags.train_lists[0], "r") as tr_fobj:
            train_filepath_list = [x.strip("\n") for x in tr_fobj.readlines()]

        raw_dataset = tf.data.TFRecordDataset(train_filepath_list)
        # print("complex phi in generator", self._flags.complex_phi)
        if not self._flags.complex_phi:
            parsed_dataset = raw_dataset.map(tfr_helper.parse_t2d, num_parallel_calls=10)
        else:
            parsed_dataset = raw_dataset.map(tfr_helper.parse_t2d_phi_complex, num_parallel_calls=10)


        # parsed_dataset = parsed_dataset.shuffle(buffer_size=1000)
        parsed_dataset_batched = parsed_dataset.batch(self._flags.train_batch_size)
        if self._input_params:
            parsed_dataset_batched = parsed_dataset_batched.map(self.tf_cut_phi_batch)

        self.dataset = parsed_dataset_batched.repeat()

        return self.dataset.prefetch(100)

    def get_input_fn_val(self):

        with open(self._flags.val_list, "r") as tr_fobj:
            train_filepath_list = [x.strip("\n") for x in tr_fobj.readlines()]

        raw_dataset = tf.data.TFRecordDataset(train_filepath_list)
        if not self._flags.complex_phi:
            parsed_dataset = raw_dataset.map(tfr_helper.parse_t2d, num_parallel_calls=10)
        else:
            parsed_dataset = raw_dataset.map(tfr_helper.parse_t2d_phi_complex, num_parallel_calls=10)
        self.dataset = parsed_dataset.batch(self._flags.val_batch_size)
        if self._input_params:
            self.dataset = self.dataset.map(self.tf_cut_phi_batch)

        return self.dataset.prefetch(2)

    def get_input_fn_file(self, filepath, batch_size=1):
        assert type(filepath) is str
        raw_dataset = tf.data.TFRecordDataset(filepath)
        if not self._flags.complex_phi:
            parsed_dataset = raw_dataset.map(tfr_helper.parse_t2d)
        else:
            parsed_dataset = raw_dataset.map(tfr_helper.parse_t2d_phi_complex)
        self.dataset = parsed_dataset.batch(batch_size)
        return self.dataset.prefetch(100)


if __name__ == "__main__":
    import util.flags as flags
    import trainer.trainer_base  # do not remove, needed for flag imports

    print("run input_fn_generator_2dtriangle debugging...")

    # gen = Generator2dt(flags.FLAGS)
    # for i in range(10):
    #     in_data, tgt = gen.get_data().__next__()
    #     print("output", type(tgt["points"]), in_data["fc"].shape)
    #     # print(tgt["points"])
    #
    # print(os.getcwd())
    # flags.print_flags()
    #
    # input_fn = InputFn2DT(flags.FLAGS)
    # train_dataset = input_fn.get_input_fn_train()
    # counter = 0
    # for i in train_dataset:
    #     counter += 1
    #     if counter >= 10:
    #         break
    #     in_data, tgt = i
    #     print("output", type(tgt["points"]), in_data["fc"].shape)
    #     # print(tgt["points"])
    #
    # print("Done.")

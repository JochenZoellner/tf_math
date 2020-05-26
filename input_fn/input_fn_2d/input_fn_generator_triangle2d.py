import logging
import time

import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    import util.flags as flags

import input_fn.input_fn_2d.data_gen_2dt.data_gen_t2d_util.tfr_helper as tfr_helper
from input_fn.input_fn_generator_base import InputFnBase
from util.flags import update_params
import model_fn.util_model_fn.custom_layers as c_layers
import input_fn.input_fn_2d.data_gen_2dt.data_gen_t2d_util.tf_polygon_2d_helper as tf_p2d
import input_fn.input_fn_2d.data_gen_2dt.data_gen_t2d_util.triangle_2d_helper as t2d
from input_fn.input_fn_2d.input_fn_2d_util import phi_array_open_symetric_no90
logger = logging.getLogger(__name__)


# logger.setLevel("DEBUG")


class InputFn2DT(InputFnBase):
    """Input Function Generator for 2d triangle problems,  dataset returns a dict..."""

    def __init__(self, flags_):
        super(InputFn2DT, self).__init__(flags_)
        self.iterator = None
        self._next_batch = None
        self.dataset = None
        self._dphi = 0.01
        self._val_infinity = False
        self._batch_size = None

        logger.info("Set input_fn_params")
        self._input_params["min_fov"] = 0.0
        self._input_params["max_fov"] = 180.0
        self._input_params["centered"] = False

        # Updating of the default params if provided via flags as a dict
        self._input_params = update_params(self._input_params, self._flags.input_params, "input")

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

    def batch_generator(self):
        _phi_arr = phi_array_open_symetric_no90(delta_phi=self._dphi)
        print(_phi_arr.shape)
        D_TYPE = tf.float32
        phi_tf = tf.expand_dims(tf.expand_dims(tf.constant(_phi_arr, D_TYPE), axis=0), axis=0)

        fc_obj = c_layers.ScatterPolygon2D(phi_tf, dtype=D_TYPE, with_batch_dim=True)
        phi_batch = np.broadcast_to(np.expand_dims(_phi_arr, axis=0),
                                    (self._batch_size, 1, _phi_arr.shape[0]))
        while True:
            point_list = []
            for i in range(self._batch_size):
                points = t2d.generate_target(x_sorted=True, center_of_weight=self._input_params["centered"])
                point_list.append(points)

            batch_points = np.stack(point_list)
            batch_points = tf_p2d.make_positiv_orientation(batch_points).numpy()
            fc_arr = fc_obj(batch_points)
            fc_batch = tf.concat((phi_batch, fc_arr), axis=1)
            yield {"fc": tf.cast(fc_batch, dtype=tf.float32)}, \
                  {"points": tf.cast(batch_points, dtype=tf.float32)}

    def get_input_fn_train(self):
        # One instance of train dataset to produce infinite many samples
        if self._val_infinity:
            self._batch_size = self._flags.val_batch_size
        else:
            self._batch_size = self._flags.train_batch_size
        if self._val_infinity or "infinity" in self._flags.train_lists[0]:
            parsed_dataset_batched = tf.data.Dataset.from_generator(self.batch_generator,
                                                                    output_types=(
                                                                    {"fc": tf.float32}, {"points": tf.float32}),
                                                                    output_shapes=
                                                                    ({"fc": (self._batch_size, 3, None)},
                                                                     {"points": (self._batch_size, 3, None)}))
            # parsed_dataset_batched = parsed_dataset_batched.map(lambda y, x: (y, x), num_parallel_calls=8)

            parsed_dataset_batched = parsed_dataset_batched.map(self.tf_cut_phi_batch, num_parallel_calls=4)
            return parsed_dataset_batched.prefetch(100)
        else:
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
            parsed_dataset_batched = parsed_dataset.batch(self._batch_size)
            if self._input_params:
                parsed_dataset_batched = parsed_dataset_batched.map(self.tf_cut_phi_batch)

            self.dataset = parsed_dataset_batched.repeat()

            return self.dataset.prefetch(100)

    def get_input_fn_val(self):

        if "infinity" in self._flags.val_list:
            self._val_infinity = True
            return self.get_input_fn_train()
        with open(self._val_list, "r") as tr_fobj:
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


def test_generator():
    input_fn = InputFn2DT(flags.FLAGS)
    dataset_train = input_fn.get_input_fn_train()
    max_batches = 1
    start_t = time.time()
    for i, batch in enumerate(dataset_train):
        if i >= max_batches:
            break
        print(i, batch[0]["fc"].shape, batch[1]["points"].shape)
        print(batch[0]["fc"][0, 0, :])
        # print(batch)
    t = time.time() - start_t
    samples = flags.FLAGS.train_batch_size * max_batches
    print("Time: {}, Samples: {}, S/S: {:0.0f}".format(t, samples, float(samples) / t))


if __name__ == "__main__":
    test_generator()

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

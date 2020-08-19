import logging
import multiprocessing
import time

import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    import util.flags as flags

from util.flags import update_params
from input_fn.input_fn_2d.data_gen_2dt.util_2d import interface, object_generator, misc_tf
# import input_fn.input_fn_2d.data_gen_2dt.util_2d.saver as saver
from input_fn.input_fn_2d.input_fn_generator_2d import InputFn2D
# from input_fn.input_fn_2d import InputFn2D
import model_fn.util_model_fn.custom_layers as c_layers
# import input_fn.input_fn_2d.data_gen_2dt.util_2d.tf_polygon_2d_helper as tf_p2d
from input_fn.input_fn_2d.input_fn_2d_util import phi_array_open_symetric_no90
logger = logging.getLogger(__name__)


# logger.setLevel("DEBUG")


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

    # def triangle_worker(self, i):
    #     points = object_generator.generate_target_triangle(x_sorted=True, center=False)
    #     return points

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

    #
    #         with open(self._flags.train_lists[0], "r") as tr_fobj:
    #             train_filepath_list = [x.strip("\n") for x in tr_fobj.readlines()]
    #
    #         raw_dataset = tf.data.TFRecordDataset(train_filepath_list)
    #         # print("complex phi in generator", self._flags.complex_phi)
    #         if not self._flags.complex_phi:
    #             parsed_dataset = raw_dataset.map(saver.parse_t2d, num_parallel_calls=10)
    #         else:
    #             parsed_dataset = raw_dataset.map(saver.parse_t2d_phi_complex, num_parallel_calls=10)
    #
    #         # parsed_dataset = parsed_dataset.shuffle(buffer_size=1000)
    #         parsed_dataset_batched = parsed_dataset.batch(self._batch_size)
    #         if self._input_params:
    #             parsed_dataset_batched = parsed_dataset_batched.map(self.tf_cut_phi_batch)
    #
    #         self.dataset = parsed_dataset_batched.repeat()
    #
    #         return self.dataset.prefetch(100)
    #
    # def get_input_fn_val(self):
    #
    #     if "infinity" in self._flags.val_list:
    #         self._val_infinity = True
    #         return self.get_input_fn_train()
    #     with open(self._val_list, "r") as tr_fobj:
    #         train_filepath_list = [x.strip("\n") for x in tr_fobj.readlines()]
    #
    #     raw_dataset = tf.data.TFRecordDataset(train_filepath_list)
    #     if not self._flags.complex_phi:
    #         parsed_dataset = raw_dataset.map(saver.parse_t2d, num_parallel_calls=10)
    #     else:
    #         parsed_dataset = raw_dataset.map(saver.parse_t2d_phi_complex, num_parallel_calls=10)
    #     self.dataset = parsed_dataset.batch(self._flags.val_batch_size)
    #     if self._input_params:
    #         self.dataset = self.dataset.map(self.tf_cut_phi_batch)
    #
    #     return self.dataset.prefetch(2)
    #
    # def get_input_fn_file(self, filepath, batch_size=1):
    #     assert type(filepath) is str
    #     raw_dataset = tf.data.TFRecordDataset(filepath)
    #     if not self._flags.complex_phi:
    #         parsed_dataset = raw_dataset.map(saver.parse_t2d)
    #     else:
    #         parsed_dataset = raw_dataset.map(saver.parse_t2d_phi_complex)
    #     self.dataset = parsed_dataset.batch(batch_size)
    #     return self.dataset.prefetch(100)


class TriangleWorker(object):
    def __init__(self, x_sorted=True, center=False):
        self._x_sorted = x_sorted
        self._center = center

    def work(self, i):
        points = object_generator.generate_target_triangle(x_sorted=self._x_sorted, center=self._center)
        return points




def test_generator():
    input_fn = InputFnTriangle2D(flags.FLAGS)
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
    # input_fn = InputFnTriangle2D(flags.FLAGS)
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

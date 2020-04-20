import datetime
import os
import time
import uuid

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hide all gpu's until needed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'} set tensorflow logleve 2=warning
import tensorflow as tf

import util.flags as flags
from util.misc import get_commit_id, Tee
import input_fn.input_fn_2d.input_fn_2d_util as phi_fn

# ========
flags.define_string("data_id", "", "select a name unique name for the dataset")
flags.define_string("output_dir", "data/synthetic_data", "select a name unique name for the dataset")
flags.define_string('print_to', 'console', 'write prints to "console, "file", "both"')
flags.define_boolean("centered", False, "use values for a and b not depending from phi")
flags.define_string("mode", "debug", "select 'val' or 'train' or 'debug'")
flags.define_list('files_train_val', int, "[int(train_files), int(val_files)]",
                  'files to generate for train data/val data', default_value=[1000, 10])
flags.define_integer("samples_per_file", 1000, "set number of samples saved in each file")
flags.define_integer("jobs", -1, "set number of samples saved in each file")
flags.define_float("delta_phi", 0.01, "set distance (in degree) between measurement points")
flags.define_float("epsilon", 0.001, "set epsilon for nummerical critical points")
flags.define_boolean("debug", False, "start debug plotting after data generation")
flags.define_dict('target_shape_params', {}, "key=value pairs defining the configuration of the input function."
                  "configure the target shape generation")


D_TYPE = tf.float32


class DataGeneratorBase(object):
    def __init__(self):
        self._flags = flags.FLAGS
        assert os.path.isdir(self._flags.output_dir)
        assert self._flags.data_id is not "", "--data_id is required and please check the working dir?"
        self._main_data_out = os.path.join(self._flags.output_dir, self._flags.data_id)
        tee_path = os.path.join(self._main_data_out, "log_{}_{}.txt".format(flags.FLAGS.data_id, flags.FLAGS.mode))
        if not os.path.isdir(os.path.dirname(tee_path)):
            os.makedirs(os.path.dirname(tee_path))
        if flags.FLAGS.print_to == "file":
            print("redirect messages to: {}".format(tee_path))
            self.tee = Tee(tee_path, console=False, delete_existing=True)
        elif flags.FLAGS.print_to == "both":
            self.tee = Tee(tee_path, console=True, delete_existing=True)
        else:
            self.tee = None

        commit_id, repos_path = get_commit_id(os.path.realpath(__file__))
        print("{} commit-id: {}".format(repos_path, commit_id))
        print("tf-version: {}".format(tf.__version__))
        flags.print_flags()

        self._shape_description = None
        self._shape_description_short = None
        self._phi_arr = phi_fn.phi_array_open_symetric_no90(self._flags.delta_phi)
        self._dtype = D_TYPE
        self.saver_obj = None
        self.parse_fn = None
        if flags.FLAGS.mode == "val":
            self._number_of_files = flags.FLAGS.files_train_val[1]
        elif flags.FLAGS.mode == "train":
            self._number_of_files = flags.FLAGS.files_train_val[0]
        elif flags.FLAGS.mode == "debug":
            self._number_of_files = 1
        else:
            raise ValueError("Unexpected mode: {}".format(flags.FLAGS.mode))
        self._data_folder = os.path.join(self._main_data_out, flags.FLAGS.mode)
        if not os.path.isdir(self._data_folder):
            os.makedirs(self._data_folder)
        self._filename_list = list(
            [os.path.join(self._data_folder, "data_{:07d}.tfr".format(x)) for x in range(self._number_of_files)])
        self._debug_batch = None

    def run(self):
        self.init_run()
        self.generate()
        self.load_test()
        print("Finished.")
        return 0

    def init_run(self):
        data_points_per_sample = self._phi_arr.shape[0]
        bytes_to_write = 4 * 3 * data_points_per_sample * self._number_of_files * flags.FLAGS.samples_per_file / 1024 ** 2

        print("\nRun '{}' data generator in {}-mode".format(self._shape_description, self._flags.mode))
        print("  Number of files: {}".format(self._number_of_files))
        print("  Data points per sample: {}".format(data_points_per_sample))
        print("  Estimated space in MB: {:0.1f}".format(bytes_to_write))
        print("  {} samples to generate.".format(self._number_of_files * flags.FLAGS.samples_per_file))
        self.saver_obj = self.get_saver_obj()
        self.parse_fn = self.get_parse_fn()

    def generate(self):
        print("Generating samples...")
        timer1 = time.time()
        for i in self._filename_list:
            self.saver_obj.save_file_tf(i)
        print("  Time for data generation: {:0.1f}".format(time.time() - timer1))
        print("  Done.")

        print("Write list...")
        out_list_name = "lists/{}_{}.lst".format(flags.FLAGS.data_id, flags.FLAGS.mode)
        with open(out_list_name, "w") as file_object:
            file_object.writelines([str(x) + "\n" for x in self._filename_list])
        print("  date+id: {}".format(datetime.datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid.uuid4())))
        print("  Done.")

    def load_test(self):
        print("Load & batch-test...")
        timer1 = time.time()
        raw_dataset = tf.data.TFRecordDataset(self._filename_list)
        print("Raw dataset:\n  ", raw_dataset)

        parsed_dataset = raw_dataset.map(self.parse_fn)
        print("Parsed dataset:\n  ", parsed_dataset)

        parsed_dataset_unbatched = parsed_dataset.unbatch()
        print("unBatched dataset:\n  ", parsed_dataset_unbatched)
        parsed_dataset_batched = parsed_dataset_unbatched.batch(self._flags.samples_per_file)
        print("Batched dataset:\n  ", parsed_dataset_batched)

        counter = 0
        for sample in parsed_dataset_batched:
            # tf.decode_raw(sample["fc"], out_type=tf.float32)
            inputs = sample[0]
            targets = sample[1]
            if counter == 0:
                print("Inputs:")
                for key in [*inputs]:
                    print("Shape of '{}': {}".format(key, inputs[key].shape))
                print("Target:")
                for key in [*targets]:
                    print("Shape of '{}': {}".format(key, targets[key].shape))

            counter += 1
        print("  Time for load test: {:0.1f}".format(time.time() - timer1))
        if self._flags.mode == 'debug':
            self._debug_batch = sample
        print("  Done.")

    def get_saver_obj(self):
        raise NotImplementedError

    def get_parse_fn(self):
        raise NotImplementedError

    def __del__(self):
        # reset print streams
        del self.tee

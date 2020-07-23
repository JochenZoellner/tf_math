import datetime
import multiprocessing
import os
import sys
import time
import uuid

import numpy as np
import deprecation


os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hide all gpu's until needed
import tensorflow as tf

import util.flags as flags
import input_fn.input_fn_2d.data_gen_2dt.util_2d.saver as tfr_helper
import model_fn.util_model_fn.custom_layers as c_layer

from util.misc import get_commit_id, Tee

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'} set tensorflow logleve 2=warning


# ========
flags.define_string("data_id", "magic_synthetic_dataset", "select a name unique name for the dataset")
flags.define_string('print_to', 'console', 'write prints to "console, "file", "both"')
flags.define_boolean("complex_phi", False, "use values for a and b not depending from phi")
flags.define_boolean("centered", False, "use values for a and b not depending from phi")
flags.define_boolean("sorted", False, "use values for a and b not depending from phi")
flags.define_float("epsilon", 0.0001, "small value to handle numerical critical points")
flags.define_float("dphi", 0.01, "distance between points in the phi array")
flags.define_string("mode", "val", "select 'val' or 'train'")
flags.define_list('files_train_val', int, "[int(train_files), int(val_files)]",
                  'files to generate for train data/val data', default_value=[1000, 10])
flags.define_integer("samples_per_file", 1000, "set number of samples saved in each file")
flags.define_integer("jobs", -1, "set number of samples saved in each file")


@deprecation.deprecated(details="Use data_generator_t2d instead")
def main():
    main_data_out = "data/synthetic_data/{}".format(flags.FLAGS.data_id)
    tee_path = os.path.join(main_data_out, "log_{}_{}.txt".format(flags.FLAGS.data_id, flags.FLAGS.mode))
    if not os.path.isdir(os.path.dirname(tee_path)):
        os.makedirs(os.path.dirname(tee_path))
    if flags.FLAGS.print_to == "file":
        print("redirect messages to: {}".format(tee_path))
        tee = Tee(tee_path, console=False, delete_existing=True)
    elif flags.FLAGS.print_to == "both":
        tee = Tee(tee_path, console=True, delete_existing=True)
    else:
        tee = None

    print("run IS2d_triangle")
    commit_id, repos_path = get_commit_id(os.path.realpath(__file__))
    print("{} commit-id: {}".format(repos_path, commit_id))
    print("tf-version: {}".format(tf.__version__))
    if flags.FLAGS.mode == "val":
        number_of_files = flags.FLAGS.files_train_val[1]
    else:
        number_of_files = flags.FLAGS.files_train_val[0]
    print("number of files: {}".format(number_of_files))

    flags.print_flags()
    timer1 = time.time()
    dphi = flags.FLAGS.dphi
    complex_phi = flags.FLAGS.complex_phi
    epsilon = flags.FLAGS.epsilon
    # complex_phi = False

    if not complex_phi:
        phi_arr = np.arange(dphi, np.pi, dphi)
    else:
        range_arr = (np.arange(10, dtype=np.float32) + 1.0) / 100.0
        zeros_arr = np.zeros_like(range_arr, dtype=np.float32)
        a = np.concatenate((range_arr, range_arr, zeros_arr), axis=0)
        b = np.concatenate((zeros_arr, range_arr, range_arr), axis=0)
        phi_arr = a + 1.0j * b

    samples_per_file = flags.FLAGS.samples_per_file
    data_folder = os.path.join(main_data_out, flags.FLAGS.mode)
    data_points_per_sample = phi_arr.shape[0]
    bytes_to_write = 4 * 3 * data_points_per_sample * number_of_files * flags.FLAGS.samples_per_file / 1024 ** 2
    D_TYPE = tf.float32
    print("dphi: {}; epsilon: {}, saved dtype: {}", dphi, epsilon, D_TYPE)
    print("data points per sample: {}".format(data_points_per_sample))
    print("estimated space in MB: {:0.1f}".format(bytes_to_write))
    print("{} samples to generate.".format(number_of_files * flags.FLAGS.samples_per_file))
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    filename_list = list([os.path.join(data_folder, "data_{:07d}.tfr".format(x)) for x in range(number_of_files)])

    print("generating samples...")

    t2d_saver_obj = tfr_helper.Triangle2dSaver(epsilon=epsilon, phi_arr=tf.constant(phi_arr, D_TYPE),
                                               x_sorted=flags.FLAGS.sorted,
                                               samples_per_file=samples_per_file, complex_phi=complex_phi,
                                               centered=flags.FLAGS.centered)

    for i in filename_list:
        t2d_saver_obj.save_file_tf(i)

    print("  Time for data generation: {:0.1f}".format(time.time() - timer1))
    print("  Done.")

    print("load&batch-test...")
    timer1 = time.time()
    raw_dataset = tf.data.TFRecordDataset(filename_list)
    print(raw_dataset)
    if not complex_phi:
        parsed_dataset = raw_dataset.map(tfr_helper.parse_t2d)
    else:
        parsed_dataset = raw_dataset.map(tfr_helper.parse_t2d_phi_complex)
    parsed_dataset_batched = parsed_dataset.batch(1000)
    # parsed_dataset_batched = parsed_dataset_batched.repeat(10)
    print(parsed_dataset)
    counter = 0
    for sample in parsed_dataset_batched:
        # tf.decode_raw(sample["fc"], out_type=tf.float32)
        a = sample[0]["fc"]
        if counter == 1:
            print(a.shape)
        counter += 1

    # print(counter)
    print("  Time for load test: {:0.1f}".format(time.time() - timer1))
    print("  Done.")

    print("write list...")
    out_list_name = "lists/{}_{}.lst".format(flags.FLAGS.data_id, flags.FLAGS.mode)
    with open(out_list_name, "w") as file_object:
        file_object.writelines([str(x) + "\n" for x in filename_list])
    print("date+id: {}".format(datetime.datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid.uuid4())))
    print("  Done.")

    print("Finished.")

    # data_path = filename_list
    # with tf.compat.v1.Session() as sess:
    #     feature = {str(prefix) + '/points': tf.FixedLenFeature([], tf.string),
    #                str(prefix) + '/fc': tf.FixedLenFeature([], tf.string)}
    #     # Create a list of filenames and pass it to a queue
    #     filename_queue = tf.train.string_input_producer(data_path, num_epochs=1)
    #     # Define a reader and read the next record
    #     reader = tf.TFRecordReader()
    #     _, serialized_example = reader.read(filename_queue)
    #     # Decode the record read by the reader
    #     features = tf.parse_single_example(serialized_example, features=feature)
    #     # Convert the image data from string back to the numbers
    #     image = tf.decode_raw(features['train/image'], tf.float32)
    #
    #     # Cast label data into int32
    #     label = tf.cast(features['train/label'], tf.int32)
    #     # Reshape image data into the original shape
    #     image = tf.reshape(image, [224, 224, 3])
    #
    #     # Any preprocessing here ...
    #
    #     # Creates batches by randomly shuffling tensors
    #     images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1,
    #                                             min_after_dequeue=10)

if __name__ == "__main__":
    main()
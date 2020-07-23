import os

import matplotlib.pyplot as plt
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'} set tensorflow logleve 2=warning

import tensorflow as tf
import input_fn.input_fn_2d.data_gen_2dt.util_2d.saver as tfr_helper


if __name__ == "__main__":
    print("run IS2d_triangle")
    # prefix = "val"
    input_list_name = "lists/t2d_tf_316_1M_val.lst"
    with open(input_list_name) as fobj:
        filename_list = [x.strip("\n") for x in fobj.readlines()]

    print("input list hast {} files".format(len(filename_list)))

    print("load&batch-test...")
    raw_dataset = tf.data.TFRecordDataset(filename_list)
    print(raw_dataset)
    parsed_dataset = raw_dataset.map(tfr_helper.parse_t2d)
    batch_size = 10
    max_batches = 10

    parsed_dataset_batched = parsed_dataset.batch(batch_size)
    # parsed_dataset_batched = parsed_dataset_batched.repeat(10)
    print(parsed_dataset)
    counter = 0
    number_of_batches = 0


    make_scatter_plot = False
    if make_scatter_plot:
        plt.figure()
        min_area = 10
        ## make scatter plot of areas
        for batch_idx, sample in enumerate(parsed_dataset_batched):
            if batch_idx >= max_batches:
                break
            number_of_batches = batch_idx + 1

            points = sample[1]["points"]
            # points[batch_sample, point, component]
            a_x = points[:, 0, 0]
            a_y = points[:, 0, 1]
            b_x = points[:, 1, 0]
            b_y = points[:, 1, 1]
            c_x = points[:, 2, 0]
            c_y = points[:, 2, 1]

            ab = np.sqrt((a_x - b_x) ** 2 + (a_y - b_y) ** 2)
            bc = np.sqrt((b_x - c_x) ** 2 + (b_y - c_y) ** 2)
            ca = np.sqrt((c_x - a_x) ** 2 + (c_y - a_y) ** 2)

            areas = np.abs((a_x * (b_y - c_y) + b_x * (c_y - a_y) + c_x * (a_y - b_y)) / 2.0)
            inner_circle = 2 * areas / (ab + bc + ca)
            outer_circle = ab * bc * ca / (4.0 * areas)
            min_area = np.minimum(min_area, np.min(areas))
            print(areas)
            print(inner_circle)
            print(outer_circle)
            plt.scatter(areas, inner_circle / outer_circle)

            # print(a, a.shape)

        print("min_area", min_area)

    for batch_idx, sample in enumerate(parsed_dataset_batched):
        if batch_idx >= max_batches:
            break
        print(sample)
        fc = sample[0]["fc"]
        points = sample[1]["points"]
        plt.figure("Sample")

        plt.plot(fc[0, 0], fc[0, 1], label="real")
        plt.plot(fc[0, 0], fc[0, 2], label="imag")
        plt.legend()
        plt.show()


    plt.show()
    print("{} samples in list: {}".format(number_of_batches * batch_size, input_list_name))

    print("  Done.")

    print("Finished.")

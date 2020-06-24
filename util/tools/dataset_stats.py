import matplotlib.pyplot as plt
import numpy as np
import logging
import os
import glob
import pathlib
from util import flags
import tensorflow as tf
import json
import input_fn.input_fn_2d.input_fn_generator_triangle2d as InputFn
import input_fn.input_fn_2d.data_gen_2dt.data_gen_t2d_util.tf_polygon_2d_helper as tf_p2dh


logger = logging.getLogger("dataset_stats.py")
# logger.setLevel("DEBUG")
# logger.setLevel("INFO")

flags.define_string('val_list', None, '.lst-file specifying the dataset used for validation')
flags.define_integer('data_len', 314, 'F(phi) amount of values saved in one line')
flags.define_dict('input_params', {}, "key=value pairs defining the configuration of the input function."
                  "input Pipeline parametrization, see input_fn.input_fn_<your-project> for usage.")
flags.define_integer("batch_limiter", -1, "set to positiv value to stop validation after this number of batches")

flags.define_boolean('complex_phi', False, "if set: a=phi.real, b=phi.imag, instead of a=cos(phi) b=sin(phi)-1")
flags.define_integer('val_batch_size', 100, 'number of elements in a val_batch between training '
                                            'epochs(default: %(default)s). '
                                            'has no effect if status is not "train"')

def find_result_files(series_dir, pattern="best_loss.json", not_pattern="fritz"):
    """returns a list with PosixPaths matching 'pattern' recursively in 'series_dir'"""
    logger.info("run 'find_result_files' on: {}".format(series_dir))
    assert os.path.isdir(series_dir), "{} is no valid directory!".format(series_dir)
    names = pathlib.Path(series_dir).rglob(pattern=pattern)
    file_paths = [x for x in names]
    logger.info("Found {} files: {}".format(len(file_paths), file_paths))
    logger.info("'find_result_files'...Done.")
    file_paths.sort()
    return file_paths

def series_to_array(file_paths):
    target_fp_list = find_result_files(series_dir=file_paths)
    res_dict = {}
    for file_path in target_fp_list:
        with open(str(file_path), "r") as fp:
            res_dict[file_path.parts[-2]] = json.load(fp)
    print(len(res_dict))
    xy_array = np.empty((2, len(res_dict)))
    for idx_, key in enumerate(res_dict.keys()):
        print(key, res_dict[key])
        x = key.split("_")[-1]
        y = json.loads(res_dict[key].replace("\'", "\""))
        y_asnumber = [x[0] for x in y.values()][0]
        xy_array[0, idx_] = x
        xy_array[1, idx_] = y_asnumber
        print("x: {:10}\t y: {}".format(x, y_asnumber))
    xy_array_sorted = xy_array[:, xy_array[0, :].argsort()]
    for i in range(xy_array_sorted.shape[1]):
        print("x: {:10}, y {}".format(xy_array_sorted[0, i], xy_array_sorted[1, i]))

    return xy_array_sorted


def signchange(array):
    """"""
    asign = np.sign(array)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(bool)
    signchange[0] = 0
    signchange[-1] = 0
    return signchange


def plot_area_distribution(triangle_area_arr):
    plt.figure(figsize=(8, 4))
    sorted_area = np.sort(triangle_area_arr)
    # print(triangle_area_arr[:500])
    # print(sorted_area[:500])
    plt.hist(triangle_area_arr, bins=100)
    plt.xlim((0, 3000))
    plt.xlabel("A")
    labels = [item.get_text() for item in plt.gca().get_yticklabels()]
    empty_string_labels = [""]*len(labels)
    plt.gca().set_yticklabels(empty_string_labels)
    plt.ylabel("occurence")
    plt.grid()
    plt.draw()
    plt.savefig("area_distribution.pdf")
    # plt.ylabel("relativ error [%]")
    # plt.xlabel("min_fov [°]")
    # plt.savefig("plot_series_json.pdf")


def plot_zero_crossing(first_zero_crossing):
    fzc_arr = np.array(first_zero_crossing)
    plt.figure(figsize=(8, 4))
    plt.hist(fzc_arr, bins=1000)
    plt.xlim((0, 90))
    plt.xlabel("A")
    labels = [item.get_text() for item in plt.gca().get_yticklabels()]
    empty_string_labels = [""] * len(labels)
    # plt.gca().set_yticklabels(empty_string_labels)
    plt.ylabel("occurence")
    plt.grid()
    # plt.ylabel("relativ error [%]")
    # plt.xlabel("min_fov [°]")
    # plt.savefig("plot_series_json.pdf")
    plt.draw()
    plt.savefig("zero_crossing.pdf")
    plt.cla()
    plt.close()


if __name__ == "__main__":
    # logging.basicConfig()
    logging.basicConfig(level="INFO")
    np.set_printoptions(precision=6, suppress=True)
    # logging.basicConfig(level="INFO")
    logger.debug("CWD: {}".format(os.getcwd()))
    assert flags.FLAGS.val_list, "--val_list must be set!"
    target_fp_list = []
    array_list = []
    input_fn_generator = InputFn.InputFn2DT(flags.FLAGS)
    dataset_val = input_fn_generator.get_input_fn_val()
    test_batch = next(iter(dataset_val))

    dphi = test_batch[0]["fc"][0, 0]
    # print("test_batch shape: {}".format(test_batch[0]["fc"].numpy().shape))
    phi_array = test_batch[0]["fc"][0, 0]

    len_phi_array = phi_array.shape[0]
    first, second = phi_array[:len_phi_array//2], phi_array[len_phi_array//2:]
    firsth_batch = test_batch[0]["fc"][:,:len_phi_array//2]
    masked_firsth_batch = np.where(signchange(firsth_batch[:]))
    # print("first, second: {}; {}".format(first.shape[0], second.shape[0]))
    # print("len_phi: {}; ".format(len_phi_array))
    #
    # print("dphi: {}".format(dphi))
    # first_zero_crossing = []
    # for idx, sample in enumerate(test_batch[0]["fc"][-15:-4]):
    #     mask_phi = np.ma.masked_where(np.invert(signchange(sample[1, :])), sample[0, :].numpy())
    #     # print(sample[1, len_phi_array//2-4:len_phi_array//2+4].numpy())
    #     print(-mask_phi[:len_phi_array//2].max() * 180.0 / np.pi + 90)
    #     print(mask_phi[len_phi_array//2:].min() * 180.0 / np.pi - 90)
    #     first_zero_crossing.append(-mask_phi[:len_phi_array//2].max() * 180.0 / np.pi + 90)
    #     first_zero_crossing.append(mask_phi[len_phi_array//2:].min() * 180.0 / np.pi - 90)

        # print(mask_phi)


    N = 1000000
    first_zero_crossing = []
    triangle_area_arr = np.empty(N)
    for (batch, (input_features, targets)) in enumerate(input_fn_generator.get_input_fn_val()):
        if flags.FLAGS.val_batch_size*(batch) >= N:
            break
        if flags.FLAGS.batch_limiter != -1 and flags.FLAGS.batch_limiter <= batch:
            print(
                "stop stats after {} batches with {} samples each.".format(batch, flags.FLAGS.val_batch_size))
            break

        for idx, sample in enumerate(input_features["fc"]):
            mask_phi = np.ma.masked_where(np.invert(signchange(sample[1, :])), sample[0, :].numpy())
            # print(sample[1, len_phi_array//2-4:len_phi_array//2+4].numpy())
            # print(-mask_phi[:len_phi_array // 2].max() * 180.0 / np.pi + 90)
            # print(mask_phi[len_phi_array // 2:].min() * 180.0 / np.pi - 90)
            first_zero_crossing.append(-mask_phi[:len_phi_array // 2].max() * 180.0 / np.pi + 90)
            first_zero_crossing.append(mask_phi[len_phi_array // 2:].min() * 180.0 / np.pi - 90)
        triangle_area_arr[batch*flags.FLAGS.val_batch_size:(batch+1)*flags.FLAGS.val_batch_size] = tf_p2dh.get_area_of_triangle(targets['points'])


    plot_area_distribution(triangle_area_arr)
    plot_zero_crossing(first_zero_crossing)

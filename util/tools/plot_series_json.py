import matplotlib.pyplot as plt
import numpy as np
import logging
import os
import glob
import pathlib
from util import flags
import json

logger = logging.getLogger("tf_polygone_2d_helper")
# logger.setLevel("DEBUG")
# logger.setLevel("INFO")

flags.define_list('series_dirs', str, 'space seperated list of training sample lists',
                  "names of the training sample lists to use. You can provide a single list as well. ",
                  [])
flags.define_string('parameter_name', "min_fov", "set variable of series, like 'min_fov', 'max_fov")
flags.define_string('parameter_unit', "°", "set variable of series, like 'min_fov', 'max_fov")


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


if __name__ == "__main__":
    # logging.basicConfig()
    logging.basicConfig(level="INFO")
    np.set_printoptions(precision=6, suppress=True)
    # logging.basicConfig(level="INFO")
    logger.debug("CWD: {}".format(os.getcwd()))
    assert flags.FLAGS.series_dirs, "--series_dirs must be set!"
    target_fp_list = []
    array_list = []
    for idx, series_dir_ in enumerate(flags.FLAGS.series_dirs):
        array_list.append(series_to_array(series_dir_))

    mean_array = np.zeros(array_list[0].shape)
    for arr_ in array_list:
        mean_array[1, :] += arr_[1, :]
    mean_array /= len(array_list)
    mean_array[0, :] = array_list[0][0, :]
    plt.figure(figsize=(8, 4))
    plt.plot(mean_array[0], mean_array[1])
    for run in array_list:
        plt.plot(run[0], run[1], "r.", markersize=2)
    out_folder = os.path.join(*os.path.split(flags.FLAGS.series_dirs[0])[:-1])
    out_folder = os.path.join(out_folder, "series_res")
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    logger.info("output folder: {}".format(out_folder))
    plt.grid()
    plt.ylabel("relativ error [%]")
    plt.xlabel("{} [{}]".format(flags.FLAGS.parameter_name,flags.FLAGS.parameter_unit))
    plt.savefig(os.path.join(out_folder, "plot_error_by_{}}.pdf".format(flags.FLAGS.parameter_name)))




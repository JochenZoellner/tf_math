import os
import glob

import tensorflow as tf
import numpy as np
from tensorflow.python.summary.summary_iterator import summary_iterator
import util.flags as flags

flags.define_string('checkpoint_dir', '', 'Checkpoint to save model information in.')
flags.define_boolean("find_max", False, "if the best value is max like in accuracy")
flags.define_string("metric", "loss", "metric name")

def get_metric_from_event_file(file_path, metric="loss", dtype=np.float32):
    metric_value_lst = []
    for e in summary_iterator(file_path):
        for v in e.summary.value:
            if v.tag == metric:
                metric_value_lst.append(np.array([e.step, np.fromstring(v.tensor.tensor_content, dtype=dtype)[0]]))
    metric_value_array = np.asarray(metric_value_lst, dtype=dtype)
    return metric_value_array


if __name__ == "__main__":
    print("best of metric v1")

    print("CWD: {}".format(os.getcwd()))

    log_dir = os.path.join(flags.FLAGS.checkpoint_dir, "logs")
    event_dirs = [name for name in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, name))]
    for event_dir in event_dirs:
        event_dir_path = os.path.join(log_dir, event_dir)
        event_files = [name for name in os.listdir(event_dir_path) if str(name).startswith("events.out.tfevents.")]
        array_list = []
        for event_file in event_files:
            array_list.append(get_metric_from_event_file(os.path.join(event_dir_path, event_file), metric=flags.FLAGS.metric))
        array = np.concatenate(array_list, axis=0)
        print("Event dir: {}".format(event_dir))
        if array.size != 0:
            if flags.FLAGS.find_max:
                print("max {}: {}".format(flags.FLAGS.metric, np.max(array[:, 1])))
            else:
                print("min {}: {}".format(flags.FLAGS.metric, np.min(array[:, 1])))
        else:
            print("no entry for {}".format(flags.FLAGS.metric))


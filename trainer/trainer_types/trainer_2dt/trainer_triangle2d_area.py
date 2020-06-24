import glob
import os
import logging
from trainer.trainer_base import TrainerBase
import tensorflow as tf
import model_fn.model_fn_2d.model_fn_triangle2d_area as models
import util.flags as flags
from input_fn.input_fn_2d.input_fn_generator_triangle2d import InputFn2DT
from util.misc import get_commit_id, Tee, get_from_keyword
# Model parameter
# ===============
flags.define_string('model_type', 'ModelTriangleArea', 'Model Type to use choose from: ModelTriangle')
flags.define_string('graph', 'GraphConv1MultiFF', 'class name of graph architecture')
flags.define_list('loss_mode', str, ['mse'], 'switch loss calculation, see model_fn_2dtriangle.py')
flags.define_integer('data_len', 314, 'F(phi) amount of values saved in one line')
flags.define_boolean('complex_phi', False, "if set: a=phi.real, b=phi.imag, instead of a=cos(phi) b=sin(phi)-1")
flags.define_string('mode', None, 'switch to plot-mode ["plot"]')
flags.FLAGS.parse_flags()


class Trainer2DTriangle(TrainerBase):
    def __init__(self):
        super(Trainer2DTriangle, self).__init__()
        self._input_fn_generator = InputFn2DT(self._flags)
        self._model_fn_class = getattr(models, self._flags.model_type)
        # self._graph.info()

    def plot_architecure(self):
        from tensorflow.keras.utils import plot_model
        commit_id, repos_path = get_commit_id(os.path.realpath(__file__))
        print("source code path:{}\ncommit-id: {}".format(repos_path, commit_id))
        print("tf-version: {}".format(tf.__version__))

        if not self._model:
            self._model = self._model_fn_class(self._params)
        if not self._model.graph_train:
            self._model.graph_train = self._model.get_graph()
            self._model.set_optimizer()
            self._model.set_interface(self._input_fn_generator.get_input_fn_val())
            self._model.graph_train.print_params()
            self._model.graph_train.summary()
        plot_model(self._model.graph_train, show_shapes=True, show_layer_names=True, expand_nested=True, to_file="network.png")

    def get_date_id(self):
        with open(flags.FLAGS.train_lists[0], "r") as f_obj:
            train_filepath = f_obj.readline().strip("\n")
        assert os.path.isfile(train_filepath)
        print(train_filepath)
        dataset_path = os.path.dirname(os.path.dirname(train_filepath))
        logging.info("dataset_train dir: {}".format(dataset_path))
        dataset_train_log_filename = glob.glob1(dataset_path, pattern="log_*_train.txt")[0]
        dataset_train_log_filepath = os.path.join(dataset_path, dataset_train_log_filename)
        logging.info("dataset_train log filepath: {}".format(dataset_train_log_filepath))
        date_id = get_from_keyword(dataset_train_log_filepath, keyword="date+id")
        return date_id


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    trainer = Trainer2DTriangle()
    date_id = trainer.get_date_id()
    print("date+id: {}".format(date_id))
    if flags.FLAGS.mode == "plot":
        trainer.plot_architecure()
    else:
        trainer.train()


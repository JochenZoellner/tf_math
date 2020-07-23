import os
import logging
from trainer.trainer_base import TrainerBase
import tensorflow as tf
import model_fn.model_fn_2d.model_fn_2dtriangle as models
import util.flags as flags
from input_fn.input_fn_2d.input_fn_generator_t2d import InputFnTriangle2D
from util.misc import get_commit_id, Tee
# Model parameter
# ===============
flags.define_string('model_type', 'ModelTriangle', 'Model Type to use choose from: ModelTriangle')
flags.define_string('graph', 'GraphMultiFF', 'class name of graph architecture')
flags.define_list('loss_mode', str, ['point_diff'], 'switch loss calculation, see model_fn_2dtriangle.py')
flags.define_integer('data_len', 3142, 'F(phi) amount of values saved in one line')
flags.define_boolean('complex_phi', False, "if set: a=phi.real, b=phi.imag, instead of a=cos(phi) b=sin(phi)-1")
flags.define_string('mode', None, 'switch to plot-mode ["plot"]')
flags.FLAGS.parse_flags()


class Trainer2DTriangle(TrainerBase):
    def __init__(self):
        super(Trainer2DTriangle, self).__init__()
        self._input_fn_generator = InputFnTriangle2D(self._flags)
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

if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    trainer = Trainer2DTriangle()
    if flags.FLAGS.mode == "plot":
        trainer.plot_architecure()
    else:
        trainer.train()

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf

tf.config.experimental_run_functions_eagerly(True)
import util.flags as flags
from trainer.trainer_base import TrainerBase

import input_fn.input_fn_2d.input_fn_generator_2d as input_fns
import model_fn.model_fn_2d.model_fn_polygon2d_classifier as models

# Model parameter
# ===============
flags.define_string('model_type', 'ModelPolygonClassifier', 'Model Type to use choose from: ModelTriangle')
flags.define_string('input_type', 'InputFnArbitraryPolygon2D', 'Model Type to use choose from: '
                                                                'InputFnArbirtraryPolygon2D, InputFnRegpularPolygon2D')

flags.define_string('loss_mode', "softmax_crossentropy", "'abs_diff', 'softmax_crossentropy")
flags.define_string('graph', 'GraphConv1MultiFF', 'class name of graph architecture')
flags.define_boolean('complex_phi', False, "if set: a=phi.real, b=phi.imag, instead of a=cos(phi) b=sin(phi)-1")
flags.define_integer('data_len', 3142, 'F(phi) amount of values saved in one line')
flags.define_integer('max_edges', 6, "Max number of edges must be known (depends on dataset), "
                                     "if unknown pick one which is definitv higher than edges in dataset")
flags.FLAGS.parse_flags()

class TrainerPolygon2DClassifier(TrainerBase):
    def __init__(self):
        super(TrainerPolygon2DClassifier, self).__init__()
        # self._input_fn_generator = InputFnRegularPolygon2D(self._flags)
        self._input_fn_generator = getattr(input_fns, self._flags.input_type)(self._flags)
        self._model_fn_class = getattr(models, self._flags.model_type)



if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    trainer = TrainerPolygon2DClassifier()
    trainer.train()

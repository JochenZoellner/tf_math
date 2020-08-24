import os
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import util.flags as flags
from trainer.trainer_base import TrainerBase
from input_fn.input_fn_2d.input_fn_generator_2d import InputFnRegularPolygon2D
import model_fn.model_fn_2d.model_fn_rp2d as models
from util.misc import get_date_id


# Model parameter
# ===============
flags.define_string('model_type', 'ModelArbitraryPolygon', 'Model Type to use choose from: ModelTriangle')
flags.define_string('graph', 'GraphConv2MultiFF', 'class name of graph architecture')
flags.define_list('loss_mode', str, ['mse'], 'switch loss calculation, see model_fn_rp2d.py')
flags.define_integer('data_len', 314, 'F(phi) amount of values saved in one line')
flags.define_boolean('complex_phi', False, "if set: a=phi.real, b=phi.imag, instead of a=cos(phi) b=sin(phi)-1")
flags.define_integer('max_edges', 12, "Max number of edges must be known (depends on dataset), "
                                     "if unknown pick one which is definitv higher than edges in dataset")
flags.FLAGS.parse_flags()


class TrainerRegularPolygon2D(TrainerBase):
    def __init__(self):
        super(TrainerRegularPolygon2D, self).__init__()
        self._input_fn_generator = InputFnRegularPolygon2D(self._flags)
        self._model_fn_class = getattr(models, self._flags.model_type)
        # self._model_fn.info()


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    trainer = TrainerRegularPolygon2D()
    # date_id = get_date_id(flags.FLAGS.train_lists[0])
    # print("date+id: {}".format(date_id))
    trainer.train()

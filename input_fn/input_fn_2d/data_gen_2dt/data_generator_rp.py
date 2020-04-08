import tensorflow as tf
from input_fn.data_generator_base import DataGeneratorBase
import input_fn.input_fn_2d.data_gen_2dt.data_gen_t2d_util.tfr_helper as tfr_helper
import util.flags as flags

flags.define_boolean("sorted", False, "sort target point by x coordinate")


class DataGeneratorRP(DataGeneratorBase):
    def __init__(self):
        super(DataGeneratorRP, self).__init__()
        self._shape_description = "Regular Polygon 2D"
        self._shape_description_short = "rp"

    def get_parse_fn(self):
        return tfr_helper.parse_regular_polygon2d

    def get_saver_obj(self):
        return tfr_helper.Triangle2dSaver(epsilon=flags.FLAGS.epsilon, phi_arr=tf.constant(self._phi_arr, self._dtype),
                                          x_sorted=flags.FLAGS.sorted, samples_per_file=flags.FLAGS.samples_per_file,
                                          centered=flags.FLAGS.centered)


if __name__ == "__main__":
    data_generator = DataGeneratorRP()
    data_generator.run()

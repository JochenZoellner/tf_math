import tensorflow as tf
from input_fn.data_generator_base import DataGeneratorBase
import input_fn.input_fn_2d.data_gen_2dt.util_2d.saver as saver
import input_fn.input_fn_2d.data_gen_2dt.util_2d.interface as interface
import util.flags as flags

flags.define_integer("min_edges", 3, "set number minimal edges, >=3")
flags.define_integer("max_edges", 6, "set number minimal edges, >=3")
flags.define_float("min_angle", 20.0, "set minimum angle (to 0 or 180 degree)")
flags.define_float("min_distance", 2.0, "set minimum distance between all points and point-side distances")


class DataGeneratorAP2D(DataGeneratorBase):
    def __init__(self):
        super(DataGeneratorAP2D, self).__init__()
        self._shape_description = "Arbitrary Polygon 2D"
        self._shape_description_short = "ap"

    def get_parse_fn(self):
        return interface.InterfaceArbitraryPolygon2D(flags.FLAGS.max_edges).parse_proto

    def get_saver_obj(self):
        return saver.ArbitraryPolygon2dSaver(epsilon=flags.FLAGS.epsilon,
                                             phi_arr=tf.constant(self._phi_arr, self._dtype),
                                             samples_per_file=flags.FLAGS.samples_per_file,
                                             min_edges=flags.FLAGS.min_edges,
                                             max_edges=flags.FLAGS.max_edges,
                                             min_angle=flags.FLAGS.min_angle,
                                             min_distance=flags.FLAGS.min_distance)

    def debug(self):
        print("Debuggin in data_generator_rp")
        import matplotlib.pyplot as plt
        from shapely import geometry
        inputs, targets = self._debug_batch

        for idx in range(flags.FLAGS.samples_per_file):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 14))
            fig.suptitle("{}: edges: {}".format(idx, targets["edges"][idx]))
            ax1.fill(targets["points"][idx][:, 0], targets["points"][idx][:, 1], "b", alpha=0.5)
            ax1.set_aspect(1.0)
            ax1.set_xlim(-50, 50)
            ax1.set_ylim(-50, 50)

            ax2.plot(inputs["fc"][idx][0], inputs["fc"][idx][1], label="fc_real")
            ax2.plot(inputs["fc"][idx][0], inputs["fc"][idx][2], label="fc_imag")


            plt.show()

if __name__ == "__main__":
    data_generator = DataGeneratorAP2D()
    data_generator.run()
    if flags.FLAGS.mode == "debug":
        data_generator.debug()

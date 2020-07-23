import input_fn.input_fn_2d.data_gen_2dt.util_2d.interface as interface
from input_fn.input_fn_2d.input_fn_generator_2d import InputFn2D


class InputFnArbirtraryPolygon2D(InputFn2D):
    """Input Function Generator for polygon 2d problems, dataset returns a dict..."""
    def __init__(self, flags_):
        super(InputFnArbirtraryPolygon2D, self).__init__(flags_)
        self._interface_obj = interface.InterfaceArbitraryPolygon2D(max_edges=self._flags.max_edges)


if __name__ == "__main__":
    import util.flags as flags

    import trainer.trainer_base  # do not remove, needed for flag imports

    print("run input_fn_generator_2dtriangle debugging...")

    # gen = Generator2dt(flags.FLAGS)
    # for i in range(10):
    #     in_data, tgt = gen.get_data().__next__()
    #     print("output", type(tgt["points"]), in_data["fc"].shape)
    #     # print(tgt["points"])
    #
    # print(os.getcwd())
    # flags.print_flags()
    #
    # input_fn = InputFnTriangle2D(flags.FLAGS)
    # train_dataset = input_fn.get_input_fn_train()
    # counter = 0
    # for i in train_dataset:
    #     counter += 1
    #     if counter >= 10:
    #         break
    #     in_data, tgt = i
    #     print("output", type(tgt["points"]), in_data["fc"].shape)
    #     # print(tgt["points"])
    #
    # print("Done.")

import logging

logger = logging.getLogger(__name__)


class InputFnBase(object):
    def __init__(self, flags):
        self._flags = flags
        self._input_params = dict()

        if self._flags.hasKey("val_lists") and self._flags.val_lists:
            self._val_list = self._flags.val_lists[0]
            if self._flags.val_list:
                logger.critical("--val_list with value: {} is IGNORED if --val_lists (with S) is not empty")
        else:
            self._val_list = self._flags.val_list

    def get_input_fn_train(self):
        pass

    def get_input_fn_val(self):
        pass

    def print_params(self):
        print("##### {}:".format("INPUT"))
        sorted_dict = sorted(self._input_params.items(), key=lambda kv: kv[0])
        for a in sorted_dict:
            print("  {}: {}".format(a[0], a[1]))

    def set_val_list(self, value):
        self._val_list = self._flags.val_lists[value]
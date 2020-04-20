import glob
import logging
import os
import time

import tensorflow as tf
from tensorflow.python.eager import profiler

import util.flags as flags
from util.misc import get_commit_id, Tee
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Training
# ========
flags.define_integer('epochs', 200, 'Epochs to train. If checkpoint already has these epochs, '
                                    'a evaluation and export is done')
flags.define_integer('samples_per_epoch', 100000, 'Samples shown to the net per epoch.')
flags.define_boolean('calc_ema', False, 'Choose whether you want to use EMA (Exponential Moving Average) '
                                        'weights or not,')
# flags.define_float('clip_grad', 0.0, 'gradient clipping value: for positive values GLOBAL norm clipping is performed,'
#                                      ' for negative values LOCAL norm clipping is performed (default: %(default)s)')
flags.define_string('optimizer', 'FinalDecayOptimizer', 'the optimizer used to compute and apply gradients.')
flags.define_dict('optimizer_params', {}, "key=value pairs defining the configuration of the optimizer.")
flags.define_dict('input_params', {}, "key=value pairs defining the configuration of the input function."
                  "input Pipeline parametrization, see input_fn.input_fn_<your-project> for usage.")
flags.define_dict('model_params', {}, "key=value pairs defining the configuration of the model function."
                  "model specific parametrization, see model_fn.model_fn_<your-project> for usage.")
flags.define_dict('graph_params', {}, "key=value pairs defining the configuration of the input function."
                  "graph specific parametrization, see model_fn.model_fn_<your-project>. ... graphs for usage.")

# flags.define_string('train_scopes', '', 'Change only variables in this scope during training')
flags.define_integer('eval_every_n', 1, "Evaluate/Validate every 'n' epochs")  # Todo: to be implemented
flags.define_string('checkpoint_dir', '', 'Checkpoint to save model information in.')
# flags.define_string('warmstart_dir', '', 'load pretrained model (ignored if checkpoint_dir already exists, '
#                                          'then this one is used).')
flags.define_boolean('reset_global_step', False, 'resets global_step, this restarts the learning_rate decay,'
                                                 'only works with load from warmstart_dir')  # Todo: to be implemented
flags.define_list('train_lists', str, 'space seperated list of training sample lists',
                  "names of the training sample lists to use. You can provide a single list as well. ",
                  ["lists/stazh_train.lst"])
flags.define_list('train_list_ratios', float, 'space seperated list of training sample list ratios',
                  "List has to have the same length as the train_list ", [1.0])
flags.define_integer('train_batch_size', 100, 'number of elements in a training batch, '
                                              'samples between optimizer steps.')
# flags.define_integer('train_accum_steps', 1,
#                      'Reduce on device batchSize by gradient accumulation (default: %(default)s).'
#                      'Train_batch_size is divided by this factor BUT the gradient is accumulated'
#                      'this many times, until an optimization step is performed. This allows HIGH'
#                      'batchSizes even with limited memory and huge models.')
flags.define_string('val_list', None, '.lst-file specifying the dataset used for validation')
flags.define_list('val_lists', str, 'space separated list of val-list-paths',
                  'names of val-lists, if set ''--val_list is ignored!', default_value=None)
flags.define_integer('val_batch_size', 100, 'number of elements in a val_batch between training '
                                            'epochs(default: %(default)s). '
                                            'has no effect if status is not "train"')
flags.define_boolean('profile', False, 'produce profile file each epoch')
# flags.define_boolean('predict_mode', False, 'If and only if true the prediction will be accomplished, '
#                                             'predict means no targets provided')
# flags.define_string('predict_list', '',
#                     '.lst-file specifying the dataset used for prediction. Only used in predict_mode')
# flags.define_string('predict_dir', '', 'path/to/file where to write the prediction')


# Hardware
# ========
# flags.define_boolean('xla', False, 'Disable in case of XLA related errors or performance issues (default: %(default)s)')
flags.define_list('gpu_devices', int, 'space seperated list of GPU indices to use. ', " ", [])
# flags.define_string('dist_strategy', 'mirror', 'DistributionStrategy in MultiGPU scenario. '
#                                                'mirror - MirroredStrategy, ps - ParameterServerStrategy')
# flags.define_boolean('gpu_auto_tune', False, 'GPU auto tune (default: %(default)s)')

flags.define_float('gpu_memory_limit', -1, 'set gpu memory in MB allocated on each (allowed) gpu '
                                           'KEEP IN MIND: there are around 350 MB overhead per process,'
                                           'NOTE: this disables memory growth!')
flags.define_float('gpu_memory_growth', True, 'allocate only the needed memory within memory-limit')
flags.define_string('print_to', 'console', 'write prints to "console, "file", "both"')
flags.define_boolean("tensorboard", True, "if True: write tensorboard logs")
flags.define_boolean('force_eager', False, 'ignore tf.function decorator, run every thing eagerly for debugging')
flags.FLAGS.parse_flags()


def set_run_config():
    # hide gpu's befor gpu initializing by using tf functions
    gpu_list = ','.join(str(x) for x in flags.FLAGS.gpu_devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    print("VISIBLE GPU-DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        if flags.FLAGS.gpu_memory_growth:
            tf.config.experimental.set_memory_growth(gpu, True)
        if flags.FLAGS.gpu_memory_limit > 0:
            tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=flags.FLAGS.gpu_memory_limit)])

    if flags.FLAGS.force_eager:
        tf.config.experimental_run_functions_eagerly(run_eagerly=True)


set_run_config()


class TrainerBase(object):
    def __init__(self):
        self._flags = flags.FLAGS
        tee_path = os.path.join(os.path.dirname(self._flags.checkpoint_dir),
                                "log_" + os.path.basename(self._flags.checkpoint_dir) + ".txt")
        if self._flags.print_to == "file":
            self.tee = Tee(tee_path, console=False, delete_existing=False)
        elif self._flags.print_to == "both":
            self.tee = Tee(tee_path, console=True, delete_existing=False)
        else:
            self.tee = None

        flags.print_flags()
        self._input_fn_generator = None
        self._model_fn_class = None
        self._model = None
        self._tape = None
        self._checkpoint_obj_val = None
        self._optimizer_fn = None
        self._optimizer = None
        self._train_dataset = None
        self._params = None
        self._current_epoch = 0
        self.epoch_loss = 0.0
        self._train_collection = None
        self._params = {'steps_per_epoch': int(self._flags.samples_per_epoch / self._flags.train_batch_size),
                        'num_gpus': len(self._flags.gpu_devices)}

    def __del__(self):
        # reset print streams
        del self.tee

    def train(self):
        if self._flags.profile:
            profiler.start_profiler_server(6009)
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

        if self._flags.calc_ema:
            ema = tf.train.ExponentialMovingAverage(decay=0.999)

        checkpoint_obj = tf.train.Checkpoint(step=self._model.graph_train.global_step, optimizer=self._model.optimizer,
                                             model=self._model.graph_train)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint_obj, directory=self._flags.checkpoint_dir,
                                                        max_to_keep=1)

        if tf.train.get_checkpoint_state(self._flags.checkpoint_dir):
            print("restore from checkpoint: {}".format(self._flags.checkpoint_dir))
            checkpoint_obj.restore(tf.train.latest_checkpoint(self._flags.checkpoint_dir))
        if self._model.graph_train.global_epoch.numpy() >= self._flags.epochs:
            print('Loaded model already in epoch {}. Evaluation...'.format(
                self._model.graph_train.global_epoch.numpy()))
            self.eval()  # run eval() if epochs reach on first attempt
            self.export()
            return 0
        else:
            print('starting in epoch ' + str(self._model.graph_train.global_epoch.numpy()))

        if not self._train_dataset:
            self._train_dataset = self._input_fn_generator.get_input_fn_train()

        @tf.function(input_signature=self._model.graph_signature)
        def _train_step_intern(input_features_, targets_):
            with tf.GradientTape() as self.tape:
                self._model.graph_train._graph_out = self._model.graph_train(input_features_, training=True)
                loss = self._model.loss(predictions=self._model.graph_train._graph_out, targets=targets_)
                gradients = self.tape.gradient(loss, self._model.graph_train.trainable_variables)
                self._model.optimizer.apply_gradients(zip(gradients, self._model.graph_train.trainable_variables))
                self._model.graph_train.global_step.assign(self._model.optimizer.iterations)
                self._model.graph_train._graph_out["loss"] = tf.reduce_mean(loss)
            return self._model.graph_train._graph_out

        while True:
            if self._model.graph_train.global_epoch.numpy() >= self._flags.epochs:
                break
            self.epoch_loss = 0.0
            t1 = time.time()
            self._model.set_mode("train")
            train_batch_number = 0

            for (batch, (input_features, targets)) in enumerate(self._input_fn_generator.get_input_fn_train()):

                # do the _train_step as tf.function to improve performance
                train_out_dict = _train_step_intern(input_features, targets)
                if self._flags.calc_ema:
                    ema.apply(self._model.graph_train.trainable_variables)
                self._model.to_tensorboard(train_out_dict, targets, input_features)
                self.epoch_loss += train_out_dict["loss"]
                train_batch_number = batch
                if batch + 1 >= int(self._flags.samples_per_epoch / self._flags.train_batch_size):
                    # stop endless '.repeat()' dataset with break
                    break

            self.epoch_loss /= float(train_batch_number + 1.0)
            self._model.graph_train.global_epoch.assign_add(1)
            print("\nEPOCH:   {:10.0f}, optimizer steps: {:9}".format(self._model.graph_train.global_epoch.numpy(),
                                                                      self._model.graph_train.global_step.numpy()))
            print("train-loss:{:8.3f}, samples/seconde:{:8.1f}, time:{:6.1f}"
                  .format(self.epoch_loss, self._flags.samples_per_epoch / (time.time() - t1), time.time() - t1))
            # Save checkpoint each epoch
            checkpoint_manager.save()
            self._model.write_tensorboard()

            # Evaluation on this checkpoint
            self._model.set_mode("eval")
            self.eval()


        self.export()

    def eval(self):
        if not self._model:
            self._model = self._model_class(self._params)
        if not self._model.graph_eval:
            self._model.graph_eval = self._model.get_graph()
        if not self._checkpoint_obj_val:
            self._checkpoint_obj_val = tf.train.Checkpoint(model=self._model.graph_eval)

        self._checkpoint_obj_val.restore(tf.train.latest_checkpoint(self._flags.checkpoint_dir))
        val_loss = 0.0
        t_val = time.time()

        @tf.function(input_signature=self._model.graph_signature)
        def call_graph(input_features_, targets_):
            self._model.graph_eval._graph_out = self._model.graph_eval(input_features_, training=False)
            loss_ = self._model.loss(predictions=self._model.graph_eval._graph_out, targets=targets_)
            self._model.graph_eval._graph_out["loss"] = tf.reduce_mean(loss_)
            return self._model.graph_eval._graph_out

        val_batch_number = 0

        if self._flags.val_lists:
            val_lists = self._flags.val_lists
        else:
            val_lists = [self._flags.val_list]
        print('\nEvaluation results after epoch {}:'.format(self._model.graph_eval.global_epoch.numpy()))
        for idx, val_list in enumerate(val_lists):
            if self._flags.val_lists:
                self._input_fn_generator.set_val_list(idx)
                eval_name = os.path.basename(val_list)[:-4]
            else:
                eval_name = None
            for (batch, (input_features, targets)) in enumerate(self._input_fn_generator.get_input_fn_val()):
                self.open_maybe_save_graph(maybe=batch == 1 and self._model.graph_train.global_epoch.numpy() == 1)
                eval_out_dict = call_graph(input_features, targets)
                self.close_maybe_save_graph(maybe=batch == 1 and self._model.graph_train.global_epoch.numpy() == 1)
                self._model.to_tensorboard(eval_out_dict, targets, input_features)
                val_loss += eval_out_dict["loss"]
                val_batch_number = batch
            val_loss /= float(val_batch_number + 1.0)
            if self._flags.val_lists:
                print("Results for list: {}".format(val_list))
            print("val-loss:{:10.3f}, samples/second:{:8.1f}, time:{:6.1f}"
                  .format(val_loss, (val_batch_number + 1) * self._flags.val_batch_size /
                          (time.time() - t_val), time.time() - t_val))
            self._model.write_tensorboard(summary_writer_name=os.path.basename(val_list[:-4]))

    def export(self):
        # Export as saved model
        print("Export saved_model to: {}".format(os.path.join(self._flags.checkpoint_dir, "export")))
        self._model.graph_train.save(os.path.join(self._flags.checkpoint_dir, "export"))

    @staticmethod
    def open_maybe_save_graph(maybe=False):
        if maybe:
            tf.summary.trace_on(graph=True, profiler=True)

    def close_maybe_save_graph(self, maybe=False):
        if maybe:
            with self._model.summary_writer["eval"].as_default():
                tf.summary.trace_export(
                    name="my_func_trace",
                    step=0,
                    profiler_outdir=os.path.join(self._flags.checkpoint_dir, "logs"))
            self._model.summary_writer["eval"].flush()
            # tf.summary.trace_on(graph=False, profiler=False)





import logging
import os
import shutil

import model_fn.model_fn_2d.util_2d.graphs_2d as graphs
import tensorflow as tf
from model_fn.model_fn_base import ModelBase


class ModelPolygonClassifier(ModelBase):
    def __init__(self, params):
        super(ModelPolygonClassifier, self).__init__(params)
        self.mydtype = tf.float32
        self._targets = None
        self._point_dist = None
        self.mape = None
        self._summary_object = {"tgt_points": [], "pre_points": [], "ordered_best": [], "unordered_best": []}
        self._graph = self.get_graph()
        self._scatter_calculator = None
        self.scatter_polygon_tf = None
        self._loss = tf.Variable(0.0, dtype=self.mydtype, trainable=False)
        self._loss_counter = tf.Variable(0, dtype=tf.int64, trainable=False)
        self._pdf_pages = None
        # log different log types to tensorboard:
        self.metrics["train"]["loss_cross_entropy_edges"] = tf.keras.metrics.Mean("loss_cross_entropy_edges",
                                                                                  self.mydtype)
        self.metrics["eval"]["loss_cross_entropy_edges"] = tf.keras.metrics.Mean("loss_cross_entropy_edges",
                                                                                 self.mydtype)
        self.metrics["train"]["accuracy_edges"] = tf.keras.metrics.Mean("accuracy_edges", self.mydtype)
        self.metrics["eval"]["accuracy_edges"] = tf.keras.metrics.Mean("accuracy_edges", self.mydtype)

        self.metrics["train"]["abs_dist_loss"] = tf.keras.metrics.Mean("abs_dist_loss", self.mydtype)
        self.metrics["eval"]["abs_dist_loss"] = tf.keras.metrics.Mean("abs_dist_loss", self.mydtype)

    def set_interface(self, val_dataset):
        build_inputs, build_out = super(ModelPolygonClassifier, self).set_interface(val_dataset)
        # if self._flags.loss_mode == "input_diff":

        return build_inputs, build_out

    def get_graph(self):
        return getattr(graphs, self._params['flags'].graph)(self._params)

    def get_predictions(self):
        return self._graph_out

    def info(self):
        self.get_graph().print_params()

    def loss(self, predictions, targets):
        self._loss = tf.constant(0.0, dtype=self.mydtype)

        if 'softmax_cross_entropy' in self._flags.loss_mode:
            loss_edge = tf.reduce_mean(tf.compat.v1.losses.softmax_cross_entropy(targets['edges'],
                                                                                 predictions['e_pred']))
            self._loss += loss_edge
            self.metrics[self._mode]["loss_cross_entropy_edges"](loss_edge)
        else:
            logging.error("no valid loss-mode in loss_params")
            raise AttributeError

        equal_tensor = tf.equal(tf.argmax(predictions['e_pred']), tf.argmax(targets['edges']))
        match_edges_tensor = tf.where(equal_tensor, tf.ones(1, dtype=self.mydtype), tf.zeros(1, dtype=self.mydtype))
        accuracy_edges = tf.reduce_mean(match_edges_tensor)

        abs_dist_tensor = tf.losses.mean_absolute_error(tf.argmax(predictions['e_pred']), tf.argmax(targets['edges']))

        self.metrics[self._mode]["accuracy_edges"](accuracy_edges)
        self.metrics[self._mode]["abs_dist_loss"](abs_dist_tensor)

        return self._loss

    def export_helper(self):
        for train_list in self._params['flags'].train_lists:
            data_id = os.path.basename(train_list).replace("_train.lst", "").replace("_val.lst", "")
            shutil.copy(os.path.join("data/synthetic_data", data_id, "log_{}_train.txt".format(data_id)),
                        os.path.join(self._params['flags'].checkpoint_dir, "export"))
        data_id = os.path.basename(self._params['flags'].val_list).replace("_train.lst", "").replace("_val.lst", "")
        shutil.copy(os.path.join("data/synthetic_data", data_id, "log_{}_val.txt".format(data_id)),
                    os.path.join(self._params['flags'].checkpoint_dir, "export"))

    # @property
    # def graph_signature(self):
    #     gs = [{'fc': tf.TensorSpec(shape=[self._current_batch_size, 3, self._flags.data_len], dtype=tf.float32)},
    #           {'points': tf.TensorSpec(shape=[self._current_batch_size, 3, 2], dtype=tf.float32)}]
    #     logging.debug("Graph signature: {gs}".format(**locals()))
    #     return gs

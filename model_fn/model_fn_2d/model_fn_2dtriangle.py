import logging
import os
import shutil

import tensorflow as tf

import model_fn.model_fn_2d.util_2d.graphs_2d as graphs
import model_fn.util_model_fn.custom_layers as c_layer
import model_fn.util_model_fn.losses as losses
from input_fn.input_fn_2d.data_gen_2dt.util_2d import misc_tf, misc
from input_fn.input_fn_2d.input_fn_2d_util import phi2s_tf
from model_fn.model_fn_base import ModelBase


class ModelTriangle(ModelBase):
    def __init__(self, params):
        super(ModelTriangle, self).__init__(params)
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
        self.metrics["train"]["loss_input_diff"] = tf.keras.metrics.Mean("loss_input_diff", self.mydtype)
        self.metrics["eval"]["loss_input_diff"] = tf.keras.metrics.Mean("loss_input_diff", self.mydtype)
        self.metrics["train"]["loss_input_diff_normed"] = tf.keras.metrics.Mean("loss_input_diff_normed", self.mydtype)
        self.metrics["eval"]["loss_input_diff_normed"] = tf.keras.metrics.Mean("loss_input_diff_normed", self.mydtype)
        self.metrics["train"]["loss_point_diff"] = tf.keras.metrics.Mean("loss_point_diff", self.mydtype)
        self.metrics["eval"]["loss_point_diff"] = tf.keras.metrics.Mean("loss_point_diff", self.mydtype)
        self.metrics["train"]["loss_point_diff_s_norm"] = tf.keras.metrics.Mean("loss_point_diff_s_norm", self.mydtype)
        self.metrics["eval"]["loss_point_diff_s_norm"] = tf.keras.metrics.Mean("loss_point_diff_s_norm", self.mydtype)
        self.metrics["train"]["loss_best_point_diff"] = tf.keras.metrics.Mean("loss_best_point_diff", self.mydtype)
        self.metrics["eval"]["loss_best_point_diff"] = tf.keras.metrics.Mean("loss_best_point_diff", self.mydtype)

        self.metrics["train"]["loss_rmse_area"] = tf.keras.metrics.Mean("loss_rmse_area", self.mydtype)
        self.metrics["eval"]["loss_rmse_area"] = tf.keras.metrics.Mean("loss_rmse_area", self.mydtype)
        self.metrics["train"]["loss_mape_area"] = tf.keras.metrics.Mean("loss_mape_area", self.mydtype)
        self.metrics["eval"]["loss_mape_area"] = tf.keras.metrics.Mean("loss_mape_area", self.mydtype)

    def set_interface(self, val_dataset):
        build_inputs, build_out = super(ModelTriangle, self).set_interface(val_dataset)
        # if self._flags.loss_mode == "input_diff":
        self.scatter_polygon_tf = c_layer.ScatterPolygon2D(fc_tensor=tf.cast(build_inputs[0]["fc"],
                                                                             dtype=self.mydtype),
                                                           points_tf=tf.cast(build_inputs[1]["points"],
                                                                             dtype=self.mydtype),
                                                           with_batch_dim=True, dtype=tf.float64)
        return build_inputs, build_out

    def get_graph(self):
        return getattr(graphs, self._params['flags'].graph)(self._params)

    def get_placeholder(self):
        if not self._flags.complex_phi:
            return {"fc": tf.compat.v1.placeholder(tf.float32, [None, 3, None], name="infc")}
        else:
            return {"fc": tf.compat.v1.placeholder(tf.float32, [None, 4, None], name="infc")}

    def get_output_nodes(self, has_graph=True):
        if has_graph:
            tf.identity(self._graph_out['pre_points'], name="pre_points")  # name to grab from java
        return "pre_points"  # return names as comma separated string without spaces

    def get_target_keys(self):
        return 'points'

    def get_predictions(self):
        return self._graph_out['pre_points']

    def info(self):
        self.get_graph().print_params()

    def loss(self, predictions, targets):
        self._loss = tf.constant(0.0, dtype=self.mydtype)
        if "pre_points" in predictions:
            if not self.scatter_polygon_tf:
                self.scatter_polygon_tf = c_layer.ScatterPolygon2D(
                    fc_tensor=tf.cast(predictions["fc"], dtype=self.mydtype),
                    points_tf=tf.cast(predictions["pre_points"],
                                      dtype=self.mydtype),
                    with_batch_dim=True, dtype=tf.float64)
            fc = tf.cast(predictions['fc'], dtype=self.mydtype)
            pre_points = tf.cast(tf.reshape(predictions['pre_points'], [-1, 3, 2]), dtype=self.mydtype)
            pre_points = misc_tf.make_spin_positive(pre_points, dtype=self.mydtype)
            pre_in = tf.cast(self.scatter_polygon_tf(points_tf=pre_points), dtype=self.mydtype)
            tgt_in = fc[:, 1:, :]
            s_norm = phi2s_tf(fc[:, 1, :])
            s_norm_stack = tf.stack((s_norm, s_norm), axis=1)
            loss_input_diff = tf.reduce_mean(tf.keras.losses.mean_absolute_error(pre_in, tgt_in))
            loss_input_diff_s_norm = tf.reduce_mean(
                tf.keras.losses.mean_absolute_error(s_norm_stack * pre_in, s_norm_stack * tgt_in))
            # tf.print(loss_input_diff)

            # input_loss_normed
            fpre_in = tf.keras.backend.flatten(pre_in)
            ftgt_in = tf.keras.backend.flatten(tgt_in)
            subtract = tf.abs(tf.subtract(fpre_in, ftgt_in))
            max_of_both = tf.maximum(tf.abs(fpre_in), tf.abs(ftgt_in))
            add = tf.maximum(2 * max_of_both, 5.0 * tf.ones(fpre_in.shape))
            loss_input_diff_normed = 10.0 * tf.reduce_mean(tf.divide(subtract, add))

            targets_oriented = misc_tf.make_spin_positive(targets["points"], dtype=self.mydtype)
            loss_point_diff = tf.cast(tf.reduce_mean(tf.keras.losses.mean_squared_error(pre_points, targets_oriented)),
                                      self.mydtype)
            if tf.math.mod(self._loss_counter, tf.constant(50, dtype=tf.int64)) == tf.constant(0, dtype=tf.int64):
                if "best_point_diff" in self._flags.loss_mode or "show_best_point_diff" in self._flags.loss_mode:
                    loss_best_point_diff = tf.cast(losses.batch_point3_loss(targets["points"],
                                                                            predictions["pre_points"],
                                                                            batch_size=self._current_batch_size),
                                                   self.mydtype)
                else:
                    loss_best_point_diff = tf.constant(0.0, self.mydtype)
                self.metrics[self._mode]["loss_best_point_diff"](loss_best_point_diff)
            # tf.print("input_diff-loss", loss_input_diff)
            if "input_diff" in self._flags.loss_mode:
                self._loss += loss_input_diff
            if "input_diff_normed" in self._flags.loss_mode:
                self._loss += loss_input_diff_normed
            if "input_diff_s_norm" in self._flags.loss_mode:
                self._loss += loss_input_diff_s_norm
            if "point_diff" in self._flags.loss_mode:
                self._loss += loss_point_diff
            # if "best_point_diff" in self._flags.loss_mode:
            #     self._loss += loss_best_point_diff

            self.metrics[self._mode]["loss_input_diff"](loss_input_diff)
            self.metrics[self._mode]["loss_input_diff_normed"](loss_input_diff_normed)
            self.metrics[self._mode]["loss_point_diff"](loss_point_diff)
            self.metrics[self._mode]["loss_point_diff_s_norm"](loss_input_diff_s_norm)

        # relative_loss = tf.constant(0, self.mydtype)
        # mse_area = tf.constant(0, self.mydtype)
        if "pre_area" in predictions:
            areas_tgt = tf.expand_dims(misc.get_area_of_triangle(targets['points']), axis=-1)
            # tf.print(tf.shape(areas_tgt), tf.shape(predictions['pre_area']))
            if not self.mape:
                self.mape = tf.keras.losses.MeanAbsolutePercentageError()
            relative_loss = tf.reduce_mean(self.mape(areas_tgt, predictions['pre_area']))
            mse_area = tf.reduce_mean(tf.keras.losses.mean_squared_error(areas_tgt, predictions['pre_area']))

            if "mape_area" in self._flags.loss_mode:
                self._loss += relative_loss

            if "mse_area" in self._flags.loss_mode:
                self._loss += mse_area

            self.metrics[self._mode]["loss_rmse_area"](mse_area)
            self.metrics[self._mode]["loss_mape_area"](relative_loss)

        # plt.figure()
        # mask_fc = np.ma.masked_where(fc[0, 0, :] == 0.0, fc[0, 0, :])
        # plt.plot(mask_fc, fc[0, 1, :], "-r", label="input_real")
        # plt.plot(mask_fc, fc[0, 2, :], "-b", label="input_imag")
        # import input_fn.input_fn_2d.data_gen_2dt.util_2d.triangle_2d_helper as t2dh
        # scatter_ploygon = t2dh.ScatterCalculator2D(p1=targets_oriented[0][0], p2=targets_oriented[0][1], p3=targets_oriented[0][2])
        # res_np = scatter_ploygon.call_on_array(fc[0, 0, :])
        # mask_res_np = np.ma.masked_where(fc[0, 0, :] == 0.0, res_np)
        # plt.plot(mask_fc, mask_res_np.real, "+r", label="tgt_np_real")
        # plt.plot(mask_fc, mask_res_np.imag, "+b", label="tgt_np_imag")
        #
        #
        # plt.plot(mask_fc, res_scatter[0, 0, :], label="pred_rec_real")
        # plt.plot(mask_fc, res_scatter[0, 1, :], label="pred_rec_imag")
        # # res_scatter = self.scatter_polygon_tf(points_tf=targets_oriented)
        # # plt.plot(mask_fc, res_scatter[0, 0, :], label="input_rec_real")
        # # plt.plot(mask_fc, res_scatter[0, 1, :], label="input_rec_imag")
        # print("phi:", fc[0, 0, :])
        # print("tgt shape", tf.shape(targets_oriented))
        # print("scatter res shape", tf.shape(res_scatter))
        # plt.legend()
        # plt.show()
        # if self._flags.loss_mode == "point3":
        #     loss0 = tf.cast(batch_point3_loss(self._targets['points'], self._graph_out['pre_points'],
        #                                   self._params["flags"].train_batch_size), dtype=tf.float32)
        # elif self._flags.loss_mode == "no_match":
        #     loss0 = tf.cast(ordered_point3_loss(self._targets['points'], self._graph_out['pre_points'],
        #                                   self._params["flags"].train_batch_size, keras_graph=self), dtype=tf.float32)
        # elif self._flags.loss_mode == "fast_no_match":
        #     loss0 = tf.cast(no_match_loss(self._targets['points'], self._graph_out['pre_points'],
        #                                   self._params["flags"].train_batch_size), dtype=tf.float32)
        # else:
        #     raise KeyError("Loss-mode: {} do not exist!".format(self._flags.loss_mode))

        # ### plot
        # fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)
        # fig.suptitle("compare loss")
        # ax1.set_title("real")
        # ax1.plot(fc[0, 0, :], tgt_in[0, 0, :], label="normed_input")
        # ax1.plot(fc[0, 0, :], pre_in[0, 0, :], label="normed_reconstruction")
        # ax1.legend()
        # ax2.set_title("imag")
        # ax2.plot(fc[0, 0, :], tgt_in[0, 1, :], label="normed_input")
        # ax2.plot(fc[0, 0, :], pre_in[0, 1, :], label="normed_reconstruction")
        # ax2.legend()
        #
        # pre_points = pre_points[0]
        # tgt_points = targets["points"][0]
        # # print(pre_points)
        # # print(tgt_points)
        # pre_polygon = geometry.Polygon([pre_points[0], pre_points[1], pre_points[2]])
        # tgt_polygon = geometry.Polygon([tgt_points[0], tgt_points[1], tgt_points[2]])
        # # print(pre_points, tgt_points)
        # # print(i)
        # intersetion_area = pre_polygon.intersection(tgt_polygon).area
        # union_area = pre_polygon.union(tgt_polygon).area
        # # iou_arr = intersetion_area / union_area
        # # tgt_area_arr = tgt_polygon.area
        # # pre_area_arr = pre_polygon.area
        # ax3.fill(tgt_points.numpy().transpose()[0], tgt_points.numpy().transpose()[1], "b", pre_points.numpy().transpose()[0],
        #          pre_points.numpy().transpose()[1], "r", alpha=0.5)
        # ax3.set_aspect(1.0)
        # ax3.set_xlim(-50, 50)
        # ax3.set_ylim(-50, 50)
        # plt.show()
        ### end plot

        self._loss_counter.assign_add(1)
        return self._loss

    def export_helper(self):
        for train_list in self._params['flags'].train_lists:
            data_id = os.path.basename(train_list).replace("_train.lst", "").replace("_val.lst", "")
            shutil.copy(os.path.join("data/synthetic_data", data_id, "log_{}_train.txt".format(data_id)),
                        os.path.join(self._params['flags'].checkpoint_dir, "export"))
        data_id = os.path.basename(self._params['flags'].val_list).replace("_train.lst", "").replace("_val.lst", "")
        shutil.copy(os.path.join("data/synthetic_data", data_id, "log_{}_val.txt".format(data_id)),
                    os.path.join(self._params['flags'].checkpoint_dir, "export"))

    def print_evaluate(self, output_dict, target_dict):
        with tf.compat.v1.Session().as_default():
            tgt_area_sum = 0
            area_diff_sum = 0
            # loss_ordered = ordered_point3_loss(output_dict["pre_points"], target_dict["points"], self._params['flags'].val_batch_size)
            # loss_best = batch_point3_loss(output_dict["pre_points"], target_dict["points"], self._params['flags'].val_batch_size)

            self._summary_object["tgt_points"].extend(
                [target_dict["points"][x] for x in range(self._params['flags'].val_batch_size)])
            self._summary_object["pre_points"].extend(
                [output_dict["pre_points"][x] for x in range(self._params['flags'].val_batch_size)])
            if "fc" in output_dict and "fc" not in self._summary_object:
                self._summary_object["fc"] = output_dict["fc"]
            ob_buffer_list = []
            ub_buffer_list = []
            # for i in range(output_dict["pre_points"].shape[0]):
            # print("## {:4d} Sample ##".format(i))
            # # print(loss_ordered.eval())
            # print("loss: {:3.2f}(ordered)| {:3.2f} (best)".format(loss_ordered.eval()[i], loss_best.eval()[i]))
            # if np.abs(loss_ordered.eval()[i] - loss_best.eval()[i]) > 0.01:
            #     # print("WARNING: losses are not equal")
            #     ob_buffer_list.append(np.nan)
            #     ub_buffer_list.append(loss_best.eval()[i])
            # else:
            #     ob_buffer_list.append(loss_best.eval()[i])
            #     ub_buffer_list.append(np.nan)

            self._summary_object["ordered_best"].extend(ob_buffer_list)
            self._summary_object["unordered_best"].extend(ub_buffer_list)

            # print("predicted points")
            # print(output_dict["pre_points"][i])
            # print("target points")
            # print(target_dict["points"][i])
            # pred_area = np.abs(np.dot((output_dict["pre_points"][i][0] - output_dict["pre_points"][i][1]), (output_dict["pre_points"][i][1] - output_dict["pre_points"][i][2])) / 2.0)
            # tgt_area = np.abs(np.dot((target_dict["points"][i][0] - target_dict["points"][i][1]), (target_dict["points"][i][1] - target_dict["points"][i][2])) / 2.0)
            # area_diff_sum += np.max(pred_area - tgt_area)
            # tgt_area_sum += tgt_area
            # print("area diff: {:0.3f}".format(np.abs(pred_area - tgt_area) / tgt_area))
            # print("target area: {:0.3f}".format(np.abs(tgt_area)))

        return area_diff_sum, tgt_area_sum

    def print_evaluate_summary(self):
        import model_fn.model_fn_2d.util_2d.plot_t2d as plt_fn
        spt_obj = plt_fn.SummaryPlotterTriangle(summary_object=self._summary_object, flags=self._flags)
        spt_obj.process()

    @property
    def graph_signature(self):
        gs = [{'fc': tf.TensorSpec(shape=[self._current_batch_size, 3, self._flags.data_len], dtype=tf.float32)},
              {'points': tf.TensorSpec(shape=[self._current_batch_size, 3, 2], dtype=tf.float32)}]
        logging.debug("Graph signature: {gs}".format(**locals()))
        return gs

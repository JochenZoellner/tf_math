import os

import matplotlib.pyplot as plt
import model_fn.util_model_fn.custom_layers as c_layer
import numpy as np
import numpy.ma as npm
import tensorflow as tf
from input_fn.input_fn_2d.data_gen_2dt.util_2d import misc_tf, misc
from input_fn.input_fn_2d.input_fn_2d_util import phi_array_open_symetric_no90
from input_fn.input_fn_2d.input_fn_generator_2d import InputFnTriangle2D
from matplotlib.backends.backend_pdf import PdfPages
from shapely import geometry

np.set_printoptions(precision=6, suppress=True)


class SummaryPlotterTriangle(object):
    def __init__(self, summary_object, flags):
        super(SummaryPlotterTriangle).__init__()
        self.mydtype = tf.float32
        self._summary_object = summary_object
        self._summary_lenght = len(self._summary_object["tgt_points"])
        self._flags = flags
        self.fig_list = []
        self._pdf_pages = None

        if "fc" in self._summary_object:
            self._phi_array = self._summary_object["fc"][0][0:1]
        else:
            self._phi_array = tf.expand_dims(phi_array_open_symetric_no90(delta_phi=0.01), axis=0)

        self._fc_obj = c_layer.ScatterPolygon2D(self._phi_array, dtype=self.mydtype, with_batch_dim=False)
        self._input_gen = InputFnTriangle2D(self._flags)

    def cut_res(self, input_points):
        input_points = tf.squeeze(misc_tf.make_spin_positive(tf.expand_dims(input_points, axis=0)),
                                  axis=0)
        fc_res = self._fc_obj(input_points)
        phi_batch = np.broadcast_to(np.expand_dims(self._phi_array, axis=0),
                                    (1, 1, self._phi_array.shape[-1]))
        d_input_points = {"fc": tf.concat((phi_batch, tf.expand_dims(fc_res, axis=0)), axis=1)}
        fc_res_cut = tf.squeeze(self._input_gen.cut_phi_batch(d_input_points)["fc"], axis=0).numpy()
        return fc_res_cut

    def uncut_res(self, input_points):
        input_points = tf.squeeze(misc_tf.make_spin_positive(tf.expand_dims(input_points, axis=0)),
                                  axis=0)
        fc_res = self._fc_obj(input_points)
        phi_batch = self._phi_array
        return tf.concat((phi_batch, fc_res), axis=0).numpy()

    def process(self):
        print("Processing summary plot's with length: {}...".format(self._summary_lenght))
        tgt_area_arr = np.zeros(self._summary_lenght)
        pre_area_arr = np.zeros(self._summary_lenght)
        pre_area_arr = np.zeros(self._summary_lenght)
        min_aspect_ratio_arr = np.zeros(self._summary_lenght)
        iou_arr = np.zeros(self._summary_lenght)
        co_loss_arr = np.ones(self._summary_lenght) * np.nan
        wo_loss_arr = np.ones(self._summary_lenght) * np.nan
        doa_real_arr = np.zeros(self._summary_lenght)
        doa_imag_arr = np.zeros(self._summary_lenght)
        doa_real_arr_cut = np.zeros(self._summary_lenght)
        doa_imag_arr_cut = np.zeros(self._summary_lenght)

        if not self._pdf_pages:
            self._pdf_pages = PdfPages(os.path.join(self._flags.model_dir, "plot_summary.pdf"))

        select_counter = 0
        for i in range(self._summary_lenght):
            if "select_counter" in self._flags.plot_params and self._flags.plot_params[
                "select_counter"] <= select_counter:
                break
            pre_points = np.reshape(self._summary_object["pre_points"][i], (3, 2))
            tgt_points = np.reshape(self._summary_object["tgt_points"][i], (3, 2))

            # print(pre_points)
            # print(tgt_points)
            pre_polygon = geometry.Polygon([pre_points[0], pre_points[1], pre_points[2]])
            tgt_polygon = geometry.Polygon([tgt_points[0], tgt_points[1], tgt_points[2]])
            # print(pre_points, tgt_points)
            # print(i)
            intersetion_area = pre_polygon.intersection(tgt_polygon).area
            union_area = pre_polygon.union(tgt_polygon).area
            iou_arr[i] = intersetion_area / union_area
            tgt_area_arr[i] = tgt_polygon.area
            pre_area_arr[i] = pre_polygon.area
            min_aspect_ratio_arr[i] = misc.get_min_aspect_ratio(tgt_points)

            # co_loss_arr[i] = self._summary_object["ordered_best"][i]
            # wo_loss_arr[i] = self._summary_object["unordered_best"][i]
            # PLOT = "stripes"

            # fc_arr_tgt = scatter.make_scatter_data(tgt_points, epsilon=0.002, phi_arr=phi_arr_full)
            # fc_arr_pre = scatter.make_scatter_data(pre_points, epsilon=0.002, phi_arr=phi_arr_full)
            # phi_batch = np.broadcast_to(np.expand_dims(phi_arr, axis=0),
            #                             (1, 1, phi_arr.shape[0]))
            # fc_arr_tgt = fc_obj(tgt_points)
            # fc_arr_tgt = tf.concat((phi_batch, tf.expand_dims(fc_arr_tgt, axis=0)), axis=1)
            # fc_arr_pre = fc_obj(pre_points)

            def calc_doa_x(fc_tgt, fc_pre):
                return np.sum(np.abs(fc_tgt - fc_pre)) / np.sum(
                    np.abs(fc_tgt) + np.abs(fc_pre))

            def calc_doa_x_normed(fc_tgt, fc_pre):
                fc_tgt = np.array(fc_tgt)
                fc_pre = np.array(fc_pre)
                non_zero = np.nonzero(np.abs(fc_tgt) + np.abs(fc_pre))[0].shape[0]
                assert non_zero > 0.0, "tgt and pre vector is zero"
                # print("tgt + pre")
                # print(fc_tgt + fc_pre)
                doa_normed1 = np.abs(fc_tgt - fc_pre)
                doa_normed2 = np.maximum(np.abs(fc_tgt) + np.abs(fc_pre), np.broadcast_to(1.0, fc_tgt.shape))
                # print("doa nomed2 \n", doa_normed2)
                doa_normed = doa_normed1 / doa_normed2
                # print("normed x")
                # print(doa_normed)
                return np.sum(doa_normed) / non_zero

            fc_arr_tgt_cut = self.cut_res(tgt_points)
            fc_arr_pre_cut = self.cut_res(pre_points)

            fc_arr_tgt = self.uncut_res(tgt_points)
            fc_arr_pre = self.uncut_res(pre_points)
            # normal
            doa_real_arr[i] = calc_doa_x(fc_arr_tgt[1], fc_arr_pre[1])
            doa_imag_arr[i] = calc_doa_x(fc_arr_tgt[2], fc_arr_pre[2])

            doa_real_arr_cut[i] = calc_doa_x(fc_arr_tgt_cut[1], fc_arr_pre_cut[1])
            doa_imag_arr_cut[i] = calc_doa_x(fc_arr_tgt_cut[2], fc_arr_pre_cut[2])
            # normed ever delta phi has influenc between 0 and 1
            # doa_real_arr[i] = calc_doa_x_normed(fc_arr_tgt[1], fc_arr_pre[1])
            # doa_imag_arr[i] = calc_doa_x_normed(fc_arr_tgt[2], fc_arr_pre[2])
            # doa_real_arr_cut[i] = calc_doa_x_normed(fc_arr_tgt_cut[1], fc_arr_pre_cut[1])
            # doa_imag_arr_cut[i] = calc_doa_x_normed(fc_arr_tgt_cut[2], fc_arr_pre_cut[2])
            # print("target over prediction")
            # print(fc_arr_tgt_cut[1])
            # print(fc_arr_pre_cut[1])
            if "select" in self._flags.plot_params and self._flags.plot_params["select"] == "1":
                select = iou_arr[i] < 0.50 and doa_imag_arr_cut[i] < 0.03 and doa_real_arr_cut[i] < 0.03
            elif "select" in self._flags.plot_params and self._flags.plot_params["select"] == "2":
                select = iou_arr[i] < 0.70 and doa_imag_arr_cut[i] < 0.08 and doa_real_arr_cut[i] < 0.08
            elif "select" in self._flags.plot_params and self._flags.plot_params["select"] == "all":
                select = True
            else:
                select = False

            if self._flags.plot and select:
                if not self._pdf_pages:
                    self._pdf_pages = PdfPages(os.path.join(self._params['flags'].model_dir, self._flags.plot_params["filename"]))
                select_counter += 1

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 14))

                ax1.fill(tgt_points.transpose()[0], tgt_points.transpose()[1], "b", pre_points.transpose()[0],
                         pre_points.transpose()[1], "r", alpha=0.5)
                ax1.set_aspect(1.0)
                ax1.set_xlim(-50, 50)
                ax1.set_ylim(-50, 50)

                ax2.set_title("F(phi)")

                if "plot_cut" in self._flags.plot_params and self._flags.plot_params["plot_cut"]:
                    ax2.plot(fc_arr_tgt_cut[0], npm.masked_where(0 == fc_arr_tgt_cut[1], fc_arr_tgt_cut[1]), 'b-',
                             label="real_tgt", linewidth=2)
                    ax2.plot(fc_arr_tgt_cut[0], npm.masked_where(0 == fc_arr_tgt_cut[2], fc_arr_tgt_cut[2]), 'y-',
                             label="imag_tgt", linewidth=2)
                    #  prediction
                    ax2.plot(fc_arr_pre_cut[0], npm.masked_where(0 == fc_arr_pre_cut[1], fc_arr_pre_cut[1]), "g-",
                             label="real_pre_cut", linewidth=2)
                    ax2.plot(fc_arr_pre_cut[0], npm.masked_where(0 == fc_arr_pre_cut[2], fc_arr_pre_cut[2]), "r-",
                             label="imag_pre_cut", linewidth=2)
                    ax2.legend(loc=4)
                else:
                    ax2.plot(fc_arr_tgt[0], fc_arr_tgt[1], label="real_tgt")
                    ax2.plot(fc_arr_tgt[0], fc_arr_tgt[2], label="imag_tgt")
                    #  prediction
                    ax2.plot(fc_arr_pre[0], fc_arr_pre[1], label="real_pre")
                    ax2.plot(fc_arr_pre[0], fc_arr_pre[2], label="imag_pre")

                    ax2.set_xlim(0, np.pi)
                    # target scatter
                    # phi_3dim = np.abs(phi_arr - np.pi / 2)
                    # fc_arr_tgt = t2d.make_scatter_data(tgt_points, phi_arr=phi_arr, epsilon=0.002)
                    # ## prediction
                    # fc_arr_pre = t2d.make_scatter_data(pre_points, phi_arr=phi_arr, epsilon=0.002)
                    # for idx in range(fc_arr_tgt[2].shape[0]):
                    #     ax2.plot((fc_arr_tgt[1][idx], fc_arr_pre[1][idx]), (fc_arr_tgt[2][idx], fc_arr_pre[2][idx]),phi_3dim[idx],  label="diffs")
                    # #     # ax2.plot(fc_arr_tgt[1], fc_arr_tgt[2], phi_3dim,  'b', label="diffs", linewidth=0.2)
                    # #     # ax2.plot(fc_arr_pre[1], fc_arr_pre[2], phi_3dim,  'r', label="diffs", linewidth=0.2)
                    # #     ax2.plot(fc_arr_tgt[1], fc_arr_tgt[2], 'b', label="diffs", linewidth=0.2)
                    # #     ax2.plot(fc_arr_pre[1], fc_arr_pre[2], 'r', label="diffs", linewidth=0.2)
                    #     ax2.set_xlabel("real")
                    #     ax2.set_ylabel("imag")
                    # #     # ax2.set_zlabel("|phi-pi/2|")

                # # complexphi
                # target
                # if PLOT == "stripes":
                #     range_arr = (np.arange(10, dtype=np.float) + 1.0) / 10.0
                #     zeros_arr = np.zeros_like(range_arr, dtype=np.float)
                #     a = np.concatenate((range_arr, range_arr, zeros_arr), axis=0)
                #     b = np.concatenate((zeros_arr, range_arr, range_arr), axis=0)
                #     phi_arr = a + 1.0j * b
                #     fc_arr_tgt = scatter.make_scatter_data(tgt_points, phi_arr=phi_arr, epsilon=0.002, dphi=0.001,
                #                                            complex_phi=True)
                #     fc_arr_pre = scatter.make_scatter_data(pre_points, phi_arr=phi_arr, epsilon=0.002, dphi=0.001,
                #                                            complex_phi=True)
                #     # ax2.scatter(fc_arr_tgt[0], fc_arr_tgt[1], label="real_tgt")
                #     for idx in range(fc_arr_tgt[2].shape[0]):
                #         ax2.plot((fc_arr_tgt[2][idx], fc_arr_pre[2][idx]), (fc_arr_pre[3][idx], fc_arr_tgt[3][idx]),
                #                  label="diffs")

                # ax2.legend(loc=4)

                # # for beamer neiss-kick of 2019/10/30
                # from mpl_toolkits.axes_grid1 import Divider, Size
                # h = [Size.Fixed(0.0), Size.Scaled(0.), Size.Fixed(.0)]
                # v = [Size.Fixed(0.0), Size.Scaled(0.), Size.Fixed(.0)]
                # fig = plt.figure(figsize=(10, 4))
                # divider = Divider(fig, (0.06, 0.13, 0.35, 0.80), h, v, aspect=False)
                # divider2 = Divider(fig, (0.55, 0.13, 0.4, 0.80), h, v, aspect=False)
                # ax1 = plt.Axes(fig, divider.get_position())
                # ax2 = plt.Axes(fig, divider2.get_position())
                # ax1.annotate('', xy=(1.32, 0.3), xycoords='axes fraction', xytext=(1.05, 0.3),
                #                 arrowprops=dict(headlength=12, headwidth=12, color='b', lw=8))
                # ax1.annotate('', xy=(1.05, 0.6), xycoords='axes fraction', xytext=(1.326, 0.6),
                #                 arrowprops=dict(headlength=12, headwidth=12, color='r', lw=8))
                # ax1.annotate('', xy=(1.32, 0.1), xycoords='axes fraction', xytext=(1.05, 0.1),
                #                 arrowprops=dict(headlength=12, headwidth=12, color='g', lw=8))
                # fig.text(0.4, 0.42, "Berechnung",)
                # fig.text(0.4, 0.67, "Neuronales Netz",)
                # fig.text(0.4, 0.25, "Vergleich",)
                # fig.add_axes(ax1)
                # fig.add_axes(ax2)
                # plt.rc('text', usetex=True,)
                # plt.rc('font', family='serif', size=14)
                # ax1.fill(tgt_points.transpose()[0], tgt_points.transpose()[1], "b", label="Ziel", alpha=0.5)
                # ax1.fill(pre_points.transpose()[0], pre_points.transpose()[1], "r", label="Netz", alpha=0.5)
                # fig.text(0.34, 0.78, "Ziel",bbox=dict(facecolor='blue', alpha=0.5))
                # fig.text(0.34, 0.86, "Netz",bbox=dict(facecolor='red', alpha=0.5))
                # ax1.set_aspect(1.0)
                # ax1.set_xticks([-10, 0, 10])
                # ax1.set_yticks([-10, 0, 10])
                # ax1.set_xlim(-17, 17)
                # ax1.set_ylim(-17, 17)
                # ax1.set_ylabel(r'y', usetex=True)
                # ax1.set_xlabel(r'x', usetex=True)
                # ax2.set_ylabel(r'F', usetex=True)
                # ax2.set_xlabel(r'$\varphi$', usetex=True)
                # ## target
                # fc_arr_tgt = t2d.make_scatter_data(tgt_points, epsilon=0.002, dphi=0.001)
                # fc_arr_pre = t2d.make_scatter_data(pre_points, epsilon=0.002, dphi=0.001)
                # norm = np.max(np.concatenate([fc_arr_tgt[1], fc_arr_tgt[2], fc_arr_pre[1], fc_arr_pre[2] ]))
                # ax2.plot(fc_arr_tgt[0], fc_arr_tgt[1]/norm, label="Re(F$_\mathrm{Ziel}$)")
                # ax2.plot(fc_arr_tgt[0], fc_arr_tgt[2]/norm, label="Im(F$_\mathrm{Ziel}$)")
                # ## prediction
                # ax2.plot(fc_arr_pre[0], fc_arr_pre[1]/norm, label="Re(F$_\mathrm{Netz}$)")
                # ax2.plot(fc_arr_pre[0], fc_arr_pre[2]/norm, label="Im(F$_\mathrm{Netz}$)")
                # ax2.set_yticks([0, 1])
                # ax2.set_xticks([0, np.pi/2, np.pi])
                # ax2.set_xticklabels(["0", "$\pi/2$", "$\pi$"])
                # ax2.legend(loc=2)

                ax1.set_title("(red) pre: P1={:3.2f},{:3.2f}|P2={:3.2f},{:3.2f}|P3={:3.2f},{:3.2f}\n"
                              "(blue)tgt: P1={:3.2f},{:3.2f}|P2={:3.2f},{:3.2f}|P3={:3.2f},{:3.2f}\n"
                              "IoU: {:1.2f}; MAR {:0.2f}\n"
                              "DoA (real)    {:1.2f}; DoA (imag)    {:1.2f}\n"
                              "DoA_cut(real) {:1.2f}; DoA_cut(imag) {:1.2f}".format(
                                pre_points[0][0], pre_points[0][1], pre_points[1][0],
                                pre_points[1][1], pre_points[2][0], pre_points[2][1],
                                tgt_points[0][0], tgt_points[0][1], tgt_points[1][0],
                                tgt_points[1][1], tgt_points[2][0], tgt_points[2][1],
                                intersetion_area / union_area, min_aspect_ratio_arr[i],
                                doa_real_arr[i], doa_imag_arr[i],
                                doa_real_arr_cut[i], doa_imag_arr_cut[i]))
                plt.grid()

                self._pdf_pages.savefig(fig)
                plt.clf()
                plt.close()

        self._pdf_pages.close()
        print("selected: {}".format(select_counter))
        print("mean iou: {}".format(np.mean(iou_arr)))
        print("sum tgt area: {}; sum pre area: {}; p/t-area: {}".format(np.mean(tgt_area_arr), np.mean(pre_area_arr),
                                                                        np.sum(pre_area_arr) / np.sum(tgt_area_arr)))


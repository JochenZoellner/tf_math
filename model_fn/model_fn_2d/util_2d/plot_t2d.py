import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as npm
import tensorflow as tf
from matplotlib.backends.backend_pdf import PdfPages
from shapely import geometry

import model_fn.util_model_fn.custom_layers as c_layer
from input_fn.input_fn_2d.data_gen_2dt.util_2d import misc_tf, misc
from input_fn.input_fn_2d.data_gen_2dt.util_2d.misc_tf import center_triangle, flip_along_axis
from input_fn.input_fn_2d.input_fn_2d_util import phi_array_open_symetric_no90, phi2s
from input_fn.input_fn_2d.input_fn_generator_2d import InputFnTriangle2D
from util.flags import update_params

np.set_printoptions(precision=6, suppress=True)


class SummaryPlotterTriangle(object):
    def __init__(self, summary_object, flags):
        super(SummaryPlotterTriangle).__init__()
        self.mydtype = tf.float64
        self._summary_object = summary_object
        self._summary_lenght = len(self._summary_object["tgt_points"])
        self._flags = flags
        self.plot_params = {}
        self.plot_params["select"] = "all"
        self.plot_params["select_counter"] = 200
        self.plot_params["filename"] = "plot_summary.pdf"
        self.plot_params["plot_cr"] = False  # plot the triangles centered and with best 180deg rotation

        self.plot_params = update_params(self.plot_params, self._flags.plot_params, "plot")
        self.fig_list = []
        self._pdf_pages = None

        if "fc" in self._summary_object:
            print("fc is in summary object")
            self._phi_array = self._summary_object["fc"][0][0:1]
            print(self._phi_array)
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
        phi_batch = tf.cast(self._phi_array, dtype=self.mydtype)
        return tf.concat((phi_batch, fc_res), axis=0).numpy()

    def process(self):
        print("Processing summary plot's with length: {}...".format(self._summary_lenght))
        tgt_area_arr = np.zeros(self._summary_lenght)
        pre_area_arr = np.zeros(self._summary_lenght)
        pre_area_arr = np.zeros(self._summary_lenght)
        min_aspect_ratio_arr = np.zeros(self._summary_lenght)
        iou_arr = np.zeros(self._summary_lenght)
        iou_arr_cr = np.zeros(self._summary_lenght)
        co_loss_arr = np.ones(self._summary_lenght) * np.nan
        wo_loss_arr = np.ones(self._summary_lenght) * np.nan
        doa_real_arr = np.zeros(self._summary_lenght)
        doa_imag_arr = np.zeros(self._summary_lenght)
        doa_abs_arr = np.zeros(self._summary_lenght)
        doa_real_arr_cut = np.zeros(self._summary_lenght)
        doa_imag_arr_cut = np.zeros(self._summary_lenght)

        if not self._pdf_pages:
            self._pdf_pages = PdfPages(os.path.join(self._flags.model_dir, self.plot_params["filename"]))

        csv_str_list = []
        select_counter = 0
        for i in range(self._summary_lenght):

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

            # best Intersection if centered and may rotated
            pre_points_c = center_triangle(pre_points)
            pre_points_cr = flip_along_axis(pre_points_c, axis="xy")
            tgt_points_c = center_triangle(tgt_points)

            pre_polygon_c = geometry.Polygon([pre_points_c[0], pre_points_c[1], pre_points_c[2]])
            pre_polygon_cr = geometry.Polygon([pre_points_cr[0], pre_points_cr[1], pre_points_cr[2]])
            tgt_polygon_c = geometry.Polygon([tgt_points_c[0], tgt_points_c[1], tgt_points_c[2]])
            intersetion_area_c = pre_polygon_c.intersection(tgt_polygon_c).area
            union_area_c = pre_polygon_c.union(tgt_polygon_c).area
            intersetion_area_cr = pre_polygon_cr.intersection(tgt_polygon_c).area
            union_area_cr = pre_polygon_cr.union(tgt_polygon_c).area
            iou_c = intersetion_area_c / union_area_c
            iou_cf = intersetion_area_cr / union_area_cr
            iou_arr_cr[i] = max(intersetion_area_c / union_area_c, iou_cf)

            if iou_c > iou_cf:
                iou_arr_cr[i] = iou_c
                pre_points_best_match = pre_points_c
            else:
                iou_arr_cr[i] = iou_cf
                pre_points_best_match = pre_points_cr


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
            fc_arr_pre_cr = self.uncut_res(pre_points_cr)
            fc_arr_pre_c = self.uncut_res(pre_points_c)
            fc_arr_pre = self.uncut_res(pre_points)
            # normal

            def abs_array(array):
                res = np.sqrt(np.sum(np.multiply(array, array), axis=0))
                return res

            assert np.allclose(
                abs_array(fc_arr_pre_cr[1:]), abs_array(fc_arr_pre[1:]), atol=1e-4, rtol=1e-4), f'cr-assertion faild on sample {i}: {pre_points_c} and {pre_points_cr} with distance {np.max(np.abs(abs_array(fc_arr_pre[1:]-fc_arr_pre_cr[1:])))}.'

                # plt.figure()
                # plt.plot(fc_arr_pre_cr[0], abs_array(fc_arr_pre_cr[1:]) - abs_array(fc_arr_pre[1:]))
                # plt.plot(fc_arr_pre_cr[0], abs_array(fc_arr_pre_cr[1:]))
                # plt.show()


            doa_real_arr[i] = calc_doa_x(fc_arr_tgt[1], fc_arr_pre[1])
            doa_imag_arr[i] = calc_doa_x(fc_arr_tgt[2], fc_arr_pre[2])
            doa_abs_arr[i] = calc_doa_x(abs_array(fc_arr_tgt[1:]), abs_array(fc_arr_pre[1:]))

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
            if "select_counter" in self.plot_params and self.plot_params["select_counter"] <= select_counter:
                continue
            if "select" in self.plot_params and self.plot_params["select"] == "one":
                select = iou_arr[i] < 0.50 and doa_imag_arr_cut[i] < 0.03 and doa_real_arr_cut[i] < 0.03
            elif "select" in self.plot_params and self.plot_params["select"] == "two":
                select = iou_arr[i] < 0.70 and doa_imag_arr_cut[i] < 0.08 and doa_real_arr_cut[i] < 0.08
            elif "select" in self.plot_params and self.plot_params["select"] == "one_abs":
                select = iou_arr_cr[i] < 0.70 and doa_abs_arr[i] < 0.10
            elif "select" in self.plot_params and self.plot_params["select"] == "two_abs":
                select = iou_arr_cr[i] < 0.70 and doa_abs_arr[i] < 0.05
            elif "select" in self.plot_params and self.plot_params["select"] == "all":
                select = True
            else:
                select = False

            if self._flags.plot and select:
                if not self._pdf_pages:
                    self._pdf_pages = PdfPages(os.path.join(self._flags.model_dir, self.plot_params["filename"]))
                select_counter += 1

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 14))
                if "plot_cr" in self.plot_params and self.plot_params["plot_cr"]:
                    tgt_points = tgt_points_c.numpy()
                    pre_points = pre_points_best_match.numpy()
                ax1.fill(tgt_points.transpose()[0], tgt_points.transpose()[1], "b", pre_points.transpose()[0],
                         pre_points.transpose()[1], "r", alpha=0.5)
                ax1.set_aspect(1.0)
                ax1.set_xlim(-50, 50)
                ax1.set_ylim(-50, 50)

                ax2.set_title("F(phi)")

                if "plot_cut" in self.plot_params and self.plot_params["plot_cut"]:
                    ax2.plot(fc_arr_tgt_cut[0], npm.masked_where(0 == fc_arr_tgt_cut[1], fc_arr_tgt_cut[1]), 'b-',
                             label="real_tgt", linewidth=2)
                    ax2.plot(fc_arr_tgt_cut[0], npm.masked_where(0 == fc_arr_tgt_cut[2], fc_arr_tgt_cut[2]), 'y-',
                             label="imag_tgt", linewidth=2)
                    #  prediction
                    ax2.plot(fc_arr_pre_cut[0], npm.masked_where(0 == fc_arr_pre_cut[1], fc_arr_pre_cut[1]), "g-",
                             label="real_pre_cut", linewidth=2)
                    ax2.plot(fc_arr_pre_cut[0], npm.masked_where(0 == fc_arr_pre_cut[2], fc_arr_pre_cut[2]), "r-",
                             label="imag_pre_cut", linewidth=2)

                elif "plot_cr" in self.plot_params and self.plot_params["plot_cr"]:
                    if "plot_s_norm" in self.plot_params and self.plot_params["plot_s_norm"]:
                        norm = phi2s(fc_arr_tgt[0])
                    else:
                        norm = 1.0
                    ax2.plot(fc_arr_tgt[0], abs_array(fc_arr_tgt[1:]) * norm, label="abs_tgt")
                    #  prediction
                    ax2.plot(fc_arr_pre[0], abs_array(fc_arr_pre[1:]) * norm, label="abs_pre")

                    ax2.set_xlim(fc_arr_pre[0, 0], fc_arr_pre[0, -1])
                    # print(fc_arr_pre[0, :])

                elif "plot_s_norm" in self.plot_params and self.plot_params["plot_s_norm"]:
                    ax2.plot(fc_arr_tgt[0], fc_arr_tgt[1] * phi2s(fc_arr_tgt[0]), label="real_tgt")
                    ax2.plot(fc_arr_tgt[0], fc_arr_tgt[2] * phi2s(fc_arr_tgt[0]), label="imag_tgt")
                    #  prediction
                    ax2.plot(fc_arr_pre[0], fc_arr_pre[1] * phi2s(fc_arr_pre[0]), label="real_pre")
                    ax2.plot(fc_arr_pre[0], fc_arr_pre[2] * phi2s(fc_arr_pre[0]), label="imag_pre")

                    ax2.set_xlim(fc_arr_pre[0, 0], fc_arr_pre[0, -1])
                else:
                    ax2.plot(fc_arr_tgt[0], fc_arr_tgt[1], label="real_tgt")
                    ax2.plot(fc_arr_tgt[0], fc_arr_tgt[2], label="imag_tgt")
                    #  prediction
                    ax2.plot(fc_arr_pre[0], fc_arr_pre[1], label="real_pre")
                    ax2.plot(fc_arr_pre[0], fc_arr_pre[2], label="imag_pre")

                    ax2.set_xlim(fc_arr_pre[0], fc_arr_pre[-1])
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

                ax2.legend(loc=4)
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
                              "IoU:    {:1.2f}; MAR {:0.2f}\n"
                              "IoU_cr:       {:1.2f}; DoA (abs)     {:1.2f}\n"
                              "DoA (real)    {:1.2f}; DoA (imag)    {:1.2f}\n"
                              "DoA_cut(real) {:1.2f}; DoA_cut(imag) {:1.2f}".format(
                                pre_points[0][0], pre_points[0][1], pre_points[1][0],
                                pre_points[1][1], pre_points[2][0], pre_points[2][1],
                                tgt_points[0][0], tgt_points[0][1], tgt_points[1][0],
                                tgt_points[1][1], tgt_points[2][0], tgt_points[2][1],
                                iou_arr[i], min_aspect_ratio_arr[i],
                                iou_arr_cr[i], doa_abs_arr[i],
                                doa_real_arr[i], doa_imag_arr[i],
                                doa_real_arr_cut[i], doa_imag_arr_cut[i]))
                plt.grid()
                tgt_point_str = f"{tgt_points[0][0]}\t{tgt_points[0][1]}\t{tgt_points[1][0]}\t{tgt_points[1][1]}\t{tgt_points[2][0]}\t{tgt_points[2][1]}"
                pre_point_str = f"{pre_points[0][0]}\t{pre_points[0][1]}\t{pre_points[1][0]}\t{pre_points[1][1]}\t{pre_points[2][0]}\t{pre_points[2][1]}"
                calc_str = f"{intersetion_area / union_area}\t{min_aspect_ratio_arr[i]}\t{doa_real_arr[i]}\t{doa_imag_arr[i]}\t{doa_real_arr_cut[i]}\t{doa_imag_arr_cut[i]}"
                csv_str = tgt_point_str + "\t" + pre_point_str + "\t" + calc_str + "\n"
                csv_str_list.append(csv_str)
                self._pdf_pages.savefig(fig)
                plt.clf()
                plt.close()

        header_string = f"x1_t\ty1_t\tx2_t\ty2_t\tx3_t\ty3_t\tx1_p\ty1_p\tx2_p\ty2_p\tx3_p\ty3_p\tiou\tmar\tdoa_r\tdoa_i\tdoac_r\tdoac_i\n"
        self._pdf_pages.close()
        plt.figure("iou_cr-doa_abs-scatter")
        plt.grid()
        plt.scatter(iou_arr_cr, doa_abs_arr, s=1)
        plt.xlabel("iou_cr")
        plt.ylabel("doa_abs")
        plt.ylim(0.0, 1.0)
        plt.xlim(0.0, 1.0)
        plt.savefig(os.path.join(self._flags.model_dir, "iou_cr-doa_abs-scatter.pdf"))
        plt.clf()
        plt.close()

        plt.figure("iou-doa_mean-scatter")
        plt.grid()
        plt.scatter(iou_arr, (doa_real_arr + doa_imag_arr) / 2.0, s=1)
        plt.xlabel("iou")
        plt.ylabel("doa_mean")
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.savefig(os.path.join(self._flags.model_dir, "iou-doa_mean-scatter.pdf"))
        with open(os.path.join(self._flags.model_dir, self.plot_params["filename"][:-3] + "csv"), 'w') as f_obj:
            f_obj.write(header_string)
            f_obj.writelines(csv_str_list)
        print("selected: {}".format(select_counter))
        print("mean iou: {}".format(np.mean(iou_arr)))
        print("mean_mr iou: {}".format(np.mean(iou_arr_cr)))
        print("sum tgt area: {}; sum pre area: {}; p/t-area: {}".format(np.mean(tgt_area_arr), np.mean(pre_area_arr),
                                                                        np.sum(pre_area_arr) / np.sum(tgt_area_arr)))
        return {"iou": np.mean(iou_arr),
                "iou_cr": np.mean(iou_arr_cr),
                "pot-area": np.sum(pre_area_arr) / np.sum(tgt_area_arr)}


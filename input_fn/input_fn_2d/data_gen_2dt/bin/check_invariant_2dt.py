import random

# import input_fn.input_fn_2d.data_gen_2dt.util_2d.triangle_2d_helper as triangle_2d_helper
import input_fn.input_fn_2d.data_gen_2dt.util_2d.misc as misc
import input_fn.input_fn_2d.data_gen_2dt.util_2d.scatter as scatter
import input_fn.input_fn_2d.data_gen_2dt.util_2d.object_generator as object_generator
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

if __name__ == "__main__":
    print("run invariant translation")

    rnd_triangle = object_generator.generate_target_triangle()

    p1 = rnd_triangle[0]
    p2 = rnd_triangle[1]
    p3 = rnd_triangle[2]

    epsilon = 0.0001
    phi_arr = np.arange(0, np.pi, epsilon)
    alpha_deg_array = np.arange(0, 360, 1)
    alpha_rad_array = alpha_deg_array * (2 * np.pi) / 360.0

    # p1, p2, p3 = misc.rotate_triangle(p1, p2, p3, 0.0)
    p1, p2, p3 = misc.cent_triangle(p1, p2, p3)

    # print(phi_arr[:20], len(phi_arr))
    # for
    # f = np.array([triangle_2d_helper.multi_triangle_f(x, p1=p1, p2=p2, p3=p3, epsilon=epsilon * 2) for x in phi_arr])
    fcalc = scatter.ScatterCalculator2D(np.concatenate((np.expand_dims(p1, axis=0),
                                                            np.expand_dims(p2, axis=0),
                                                            np.expand_dims(p3, axis=0))), epsilon=epsilon * 2)
    f = fcalc.fc_of_phi(phi_arr)
    dx, dy = (random.uniform(-5.0, 5.0), random.uniform(-15.0, 15.0))

    p1_moved, p2_moved, p3_moved = misc.translation(p1, p2, p3, (dx, dy))
    fcalc = scatter.ScatterCalculator2D(np.concatenate((np.expand_dims(p1_moved, axis=0),
                                                            np.expand_dims(p2_moved, axis=0),
                                                            np.expand_dims(p3_moved, axis=0))), epsilon=epsilon * 2)
    f_moved = fcalc.fc_of_phi(phi_arr)

    p1_mirror, p2_mirror, p3_mirror = (-1.0 * p1, -1.0 * p2, -1.0 * p3)
    fcalc = scatter.ScatterCalculator2D(np.concatenate((np.expand_dims(p1_mirror, axis=0),
                                                            np.expand_dims(p2_mirror, axis=0),
                                                            np.expand_dims(p3_mirror, axis=0))), epsilon=epsilon * 2)
    f_mirror = fcalc.fc_of_phi(phi_arr)

    intensity_diff = np.abs(f) - np.abs(f_moved)
    real_diff = f.real - f_moved.real
    imag_diff = f.imag - f_moved.imag

    intensity_diff_mirror = np.abs(f) - np.abs(f_mirror)
    real_diff_mirror = f.real - f_mirror.real
    imag_diff_mirror = f.imag - f_mirror.imag

    fig, (ax1) = plt.subplots(1, 1)

    ax1.set_title("triangel")
    # aspectratio = 1.0
    # ratio_default = (ax1.get_xlim()[1] - ax1.get_xlim()[0]) / (ax1.get_ylim()[1] - ax1.get_ylim()[0])
    # ax1.set_aspect(ratio_default * aspectratio)
    patches = []
    triangle = Polygon((p1, p2, p3), True, alpha=0.8)
    patches.append(triangle)
    triangle = Polygon((p1_moved, p2_moved, p3_moved), True, alpha=0.4)
    patches.append(triangle)
    colors = 100 * np.random.rand(len(patches))
    p = PatchCollection(patches)
    ax1.set_xlim(-60.0, 60.0)
    ax1.set_ylim(-60.0, 60.0)
    p.set_array(np.array(colors))
    ax1.add_collection(p)

    plt.figure("intensity diffcerene while moving {:1.1f} {:1.1f}".format(dx, dy))
    plt.plot(intensity_diff)

    plt.figure("real part diffcerene while moving {:1.1f} {:1.1f}".format(dx, dy))
    plt.plot(real_diff)

    plt.figure("imag part diffcerene while moving {:1.1f} {:1.1f}".format(dx, dy))
    plt.plot(imag_diff)

    plt.figure("intensity diffcerene during point reflection")
    plt.plot(intensity_diff_mirror)

    plt.figure("real part diffcerene during point reflection")
    plt.plot(real_diff_mirror)

    plt.figure("imag part diffcerene during point reflection")
    plt.plot(imag_diff_mirror)

    plt.show()

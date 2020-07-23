import logging
import random

import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as geometry

import input_fn.input_fn_2d.data_gen_2dt.util_2d.convert as convert
import input_fn.input_fn_2d.data_gen_2dt.util_2d.misc as misc

logger = logging.getLogger("object_generator 2D")


# logger.setLevel("DEBUG")
# logger.setLevel("INFO")


def generate_target_triangle(center_of_weight=False, x_sorted=True, min_area=10, min_aspect_ratio=0.0, rng=None):
    if not rng:
        rng = np.random.Generator(np.random.PCG64())
    while True:
        points = rng.uniform(-50.0, 50, size=(3, 2)).astype(np.float32)
        a_x = points[0, 0]
        a_y = points[0, 1]
        b_x = points[1, 0]
        b_y = points[1, 1]
        c_x = points[2, 0]
        c_y = points[2, 1]

        area = np.abs((a_x * (b_y - c_y) + b_x * (c_y - a_y) + c_x * (a_y - b_y)) / 2.0)
        if area >= min_area and min_aspect_ratio < misc.get_min_aspect_ratio(points):
            break

    if center_of_weight:
        points[0], points[1], points[2] = misc.center_triangle(p1=points[0], p2=points[1], p3=points[2])
    if x_sorted:
        points = points[points[:, 0].argsort()]
    return points.astype(np.float32)


def generate_target_polygon(min_area=10, max_edges=6, min_edges=3, max_size=50, rng=None):
    if not rng:
        rng = np.random.Generator(np.random.PCG64())
    edges = rng.integers(min_edges, max_edges + 1)
    size = max_size
    # edges = 3
    logger.info("Generating polygon with {} edges".format(edges))
    while True:
        tuple_list = convert.array_to_tuples(rng.uniform(-size, size, size=(3, 2)).astype(np.float32))
        shots = 0
        while len(tuple_list) < edges:
            shots += 1
            if shots > 100:
                tuple_list = convert.array_to_tuples(
                    np.reshape([rng.uniform(-size, size) for x in range(6)], (3, 2)).astype(np.float32))
                shots = 0
            tuple_list_buffer = tuple_list.copy()
            tuple_list_buffer.insert(rng.integers(0, len(tuple_list)),
                                     (rng.uniform(-size, size), rng.uniform(-size, size)))
            linear_ring = geometry.LinearRing(tuple(tuple_list_buffer))
            if linear_ring.is_simple:
                pointy = 90
                for s_ in range(len(tuple_list_buffer)):
                    arr = convert.tuples_to_array(tuple_list_buffer)
                    logger.debug("{}; {}".format(arr[s_] - arr[s_ - 1], arr[s_ - 1] - arr[s_ - 2]))

                    angle_ = np.abs(misc.py_ang(arr[s_] - arr[s_ - 1], arr[s_ - 1] - arr[s_ - 2]))
                    angle_ = 90 - np.abs(angle_ - 90)
                    pointy = min(angle_, pointy)
                    logger.debug(angle_)
                if pointy > 15.0:
                    logger.debug("not pointy")
                    tuple_list = tuple_list_buffer.copy()

                logger.debug("simple")

            else:
                logger.debug("NOT simple")

        if misc.get_spin(tuple_list) < 0:
            logger.info("REVERSE LIST")
            tuple_list.reverse()
        polygon_points = tuple_list
        logger.info("polygon_points: {}".format(polygon_points))
        polygon_obj = geometry.Polygon(polygon_points)
        point_array = np.array([polygon_obj.exterior.xy[0][:-1], polygon_obj.exterior.xy[1][:-1]]).transpose()

        if polygon_obj.area >= min_area:
            logger.debug("area: {}".format(polygon_obj.area))
            break
    return point_array, edges


def generate_target_regular_polygon(min_radius=3, max_radius=50, min_edges=3, max_edges=8, rotation=True,
                                    translation=True):
    rpf_dict = generate_rpf(min_radius, max_radius, min_edges, max_edges, rotation, translation)
    pts_arr_rp = convert.rpf_to_points_array(rpf_dict)
    return pts_arr_rp, rpf_dict


def generate_rpf(min_radius=3, max_radius=50, min_edges=3, max_edges=8, rotation=True, translation=True, rng=None):
    """generate random Regular Polygon Format: dict(radius, rotation, translation, edges)"""
    if not rng:
        rng = np.random.Generator(np.random.PCG64())
    edges = rng.integers(min_edges, max_edges + 1)  # interval open on right side max_edge+1 is not included
    radius = rng.uniform(min_radius, max_radius)
    if translation:
        z_move = 1.0j * rng.uniform(0, max_radius - radius) + random.uniform(0, max_radius - radius)
    else:
        z_move = 0.0j + 0.0
    if rotation:
        rotation = 2.0 * np.pi * random.uniform(0, 1) / float(edges)
    else:
        rotation = 0.0
    return {"radius": radius,
            "rotation": rotation,
            "translation": np.array([z_move.real, z_move.imag]),
            "edges": edges}


def generate_target_star_polygon(min_radius=3, max_radius=30, edges=3, angle_epsilon=5.0, sectors=False, rng=None):
    if not rng:
        rng = np.random.Generator(np.random.PCG64())
    radius_array = rng.uniform(low=min_radius, high=max_radius, size=edges - 1)

    if sectors:
        phi_array = rng.uniform(low=angle_epsilon, high=360.0 / edges, size=edges - 1)
        add_epsilon = np.arange(edges - 1) * 360.0 / edges
        phi_array = phi_array + add_epsilon
    else:
        phi_array = rng.uniform(low=angle_epsilon, high=360.0 - ((edges - 1) * angle_epsilon), size=edges - 1)
        phi_array.sort()
        add_epsilon = np.arange(edges - 1) * angle_epsilon
        phi_array = phi_array + add_epsilon

    z_sum = np.sum(radius_array * np.exp(1.0j * 2 * np.pi * phi_array / 360.0))
    r_last = np.abs(z_sum)
    phi_last = -np.arctan2(z_sum.imag, -z_sum.real)
    # ToDo ensure 3 Points not on one line!
    assert np.isclose(z_sum + r_last * np.exp(1.0j * phi_last), 0.0), "star is not centered: {}; {}; {}".format(z_sum,
                                                                                                                r_last,
                                                                                                                phi_last)

    pts_arr_sp = np.zeros((edges, 2))
    pts_arr_sp[:edges - 1, 0] = radius_array
    pts_arr_sp[edges - 1, 0] = r_last
    pts_arr_sp[:edges - 1, 1] = phi_array
    pts_arr_sp[edges - 1, 1] = ((phi_last * 360.0 / (2 * np.pi)) + 360.0) % 360
    pts_arr_sp = pts_arr_sp[pts_arr_sp[:, 1].argsort()]

    return pts_arr_sp


# def generate_target_star_polygon2(min_radius=3, max_radius=30, edges=3, angle_epsilon=5.0, rng=None):
#     if not rng:
#         rng = np.random.Generator(np.random.PCG64())
#
#     def normalize(rphi_array):
#         negativ_radius_idx = np.argwhere(rphi_array[:, 0] < 0.0)
#         rphi_array[negativ_radius_idx, 1] += 180.0
#         rphi_array[negativ_radius_idx, 0] *= -1.0
#         rphi_array = rphi_array % 360.0
#         rphi_array = rphi_array[rphi_array[:, 1].argsort()]
#         return rphi_array
#
#     def check_epsilon_angle(rphi_array, epsilon=5.0):
#         roll_plus = np.all(np.abs(rphi_array[:, -1] - np.roll(rphi_array[:, -1], shift=1)) > epsilon)
#         roll_minus = np.all(np.abs(rphi_array[:, -1] - np.roll(rphi_array[:, -1], shift=-1)) > epsilon)
#         if roll_minus and roll_plus:
#             return True
#         else:
#             return False
#
#     def check_radius_range(rphi_array, min_radius=3, max_radius=50):
#         if (min_radius <= rphi_array[:, 0]).all() and (max_radius >= rphi_array[:, 0]).all():
#             return True
#         else:
#             return False
#
#     loop_counter = 1
#     while True:
#         plt.clf()
#         if loop_counter % 10 == 0:
#             logger.warning("{} attemps to generate star-polygon rphi-array".format(loop_counter))
#         loop_counter += 1
#         radius_array = rng.uniform(low=min_radius, high=max_radius, size=edges)
#         phi_array = rng.uniform(low=angle_epsilon, high=360.0 - ((edges - 1) * angle_epsilon), size=edges)
#         phi_array.sort()
#         phi_array += np.arange(edges) * angle_epsilon  # add increasing number of epsilons, avoids similar angles
#         rphi_array = np.stack((radius_array, phi_array), axis=1)
#         logger.info("initial rphi_array:\n{}".format(rphi_array))
#
#         pts_arr_rp = convert.rphi_array_to_points_array(rphi_array)
#         fig, ax1 = plt.subplots(1, 1, figsize=(9.5, 9.5), num=1)
#         plt.grid()
#         # ax1.fill(pts_arr_rp.transpose()[0], pts_arr_rp.transpose()[1], 'r', alpha=0.5)
#         ax1.set_xlim((-50, 50))
#         ax1.set_ylim((-50, 50))
#         ax1.set_aspect(aspect=1.0)
#
#         #  fix center-condition of polygon: sum(P_i)=0
#         try:
#             xy_array = np.exp(1.0j * 2 * np.pi * phi_array / 360.0)  # split phi in real and imag part
#             z = np.sum(radius_array * xy_array)  # sum all points in complex plane
#             idx = rng.choice(range(edges), 2)  # pick to points (i, j) to adjust their radius
#             logger.debug("z: {}".format(z))
#             logger.debug("xy_array: {}".format(xy_array))
#             logger.debug("point i,j indices: {}".format(idx))
#             #  solve 0 = z - r_i * e^i*phi_j + r_j * e^i*phi_i
#             counter = (z.real - ((xy_array[idx[0]].real * z.imag) / xy_array[idx[0]].imag))
#             denominator = xy_array[idx[1]].real - (
#                     xy_array[idx[1]].imag * xy_array[idx[0]].real / xy_array[idx[0]].imag)
#             r_1 = counter / denominator
#             r_0 = (z.real - (r_1 * xy_array[idx[1]].real)) / xy_array[idx[0]].real
#             logger.debug("r_j:{}, r_j:{}".format(r_0, r_1))
#             # apply raduius corrections
#             radius_array[idx] -= np.array([r_0, r_1])
#             rphi_array = np.stack((radius_array, phi_array), axis=1)
#             logger.info("corrected rphi_array:\n{}".format(rphi_array))
#             corrected_z = np.sum(radius_array * np.exp(1.0j * 2 * np.pi * phi_array / 360.0))
#             logger.info("corrected z_sum: {}".format(corrected_z))
#
#             # assert sum all points in complex plane is zero
#             assert np.isclose(corrected_z, np.array([0j])), "star polygon is not centered: {}\n{}".format(corrected_z,
#                                                                                                           rphi_array)
#         except IOError as ex:
#             # expected dive by zero error, assertion error
#             continue
#
#         logger.info("Succesfull correction!")
#
#         # add 180° to negativ radius points and -(raduis)
#         rphi_array = normalize(rphi_array)
#         points_array = convert.rphi_array_to_points_array(rphi_array)
#         ax1.fill(points_array.transpose()[0], points_array.transpose()[1], 'b', alpha=0.5)
#         logger.info("normalized rphi_array:\n{}".format(rphi_array))
#
#         radius_condition = check_radius_range(rphi_array, min_radius=min_radius, max_radius=max_radius)
#         angle_condition = check_epsilon_angle(rphi_array, epsilon=angle_epsilon)
#         logger.info("Valid Array Angle: {}".format(angle_condition))
#         logger.info("Valid Array Radius: {}".format(radius_condition))
#
#         if radius_condition and angle_condition:
#             logger.info("Target generation finished!")
#             break
#
#     plt.show()
#     return rphi_array

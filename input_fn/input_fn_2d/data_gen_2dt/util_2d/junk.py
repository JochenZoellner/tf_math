import math
import random

import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as geometry
from descartes import PolygonPatch
from scipy.spatial import Delaunay
from shapely.ops import cascaded_union, polygonize


def plot_polygon(polygon):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    margin = .3
    x_min, y_min, x_max, y_max = polygon.bounds
    ax.set_xlim([x_min - margin, x_max + margin])
    ax.set_ylim([y_min - margin, y_max + margin])
    patch = PolygonPatch(polygon, fc='#999999',
                         ec='#000000', fill=True,
                         zorder=-1)
    ax.add_patch(patch)
    return fig


def polygone_points(edge_points=3, rng=None):
    if not rng:
        rng = random.Random()
    edge_point_list = [(rng.uniform(-50, 50), rng.uniform(-50, 50)) for x in range(3)]
    # if edge_points > 3:
    #     for index in range(edge_points - 3):
    #         new_point = (rng.uniform(-50, 50), rng.uniform(-50, 50))
    #         dist_array = np.zeros(len(edge_point_list))
    #         for tupel_indexin range(len(edge_point_list)):


def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add((i, j))
        edge_points.append(coords[[i, j]])

    coords = np.array([point.coords[0] for point in points])
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = math.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = math.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        # Semiperimeter of triangle
        s = (a + b + c) / 2.0
        # Area of triangle by Heron's formula
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        # Here's the radius filter.
        # print circum_r
        if circum_r < 1.0 / alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points


if __name__ == "__main__":
    print("run polygon 2d helper")

    point_cloud = np.random.uniform(low=-50, high=50, size=(2, 12))
    print(point_cloud)

    point_list = [None] * point_cloud.shape[1]
    for i in range(point_cloud.shape[1]):
        point_list[i] = (point_cloud[0, i], point_cloud[1, i])

    print(point_list)
    point_collection = geometry.MultiPoint(point_cloud.transpose())

    point_collection.envelope

    concave_hull, edge_points = alpha_shape(point_collection,
                                            alpha=0.07)


    # plot_polygon(point_collection.envelope)

    # plot_polygon(point_collection.convex_hull)
    def get_biggest_part(multipolygon):

        # Get the area of all mutipolygon parts
        areas = [i.area for i in multipolygon]

        # Get the area of the largest part
        max_area = areas.index(max(areas))

        # Return the index of the largest area
        return multipolygon[max_area]


    plot_polygon(concave_hull)

    # plot_polygon(get_biggest_part(concave_hull))

    print(concave_hull)
    # plt.gca().add_collection(LineCollection(edge_points))
    plt.plot(point_cloud[0], point_cloud[1], 'o')
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)

    plt.show()

# if __name__ == "__main__":
#     test_star_polygon_transformation()
#     test_star_polygon()
#     # test_regular_polygon_transformation()

# if __name__ == "__main__":
#     phi_array = np.arange(0.0, np.pi, 0.01)
#     points = [(0, 0), (0, 10), (10, 0), (0, 0), (0, 10), (10, 0)]
#     polygon_calculator = ScatterCalculator2D(points)
#     f_list = [None]*phi_array.shape[0]
#     for index, phi in enumerate(phi_array):
#         q = polygon_calculator.q_of_phi(phi)
#         # scale = 1.0 / np.dot(q, q)
#         f_list[index] = polygon_calculator.F_of_qs(q=q, p0_=(1, 0), p1_=(0, 1)).real
#
#     plt.figure()
#     plt.plot(phi_array, f_list)
#     plt.show()


# if __name__ == "__main__":
#     print("test tuple array conversion")
#     rnd_array = np.random.uniform(-10, 10, (4, 2))
#     tuple_list = array_to_tuples(rnd_array)
#     print("tuple_list:", tuple_list)
#     array = tuples_to_array(tuple_list)
#     print("array:", array)
#     assert np.array_equal(rnd_array, array)

# if __name__ == "__main__":
#     print("time fc")
#     t1 = time.time()
#     loops = 100
#     for target in range(loops):
#         convex_polygon_arr = generate_target_polygon(max_edge=10)
#         convex_polygon_tuple = array_to_tuples(convex_polygon_arr)
#         polygon_calculator = ScatterCalculator2D(points=convex_polygon_tuple)
#
#         phi_array = np.arange(np.pi / 2 - 1.5, np.pi / 2 + 1.5, 0.001)
#         polygon_scatter_res = polygon_calculator.F_of_phi(phi=phi_array).astype(dtype=np.complex64)
#         # polygon_scatter_res = np.array(
#         #     [polygon_calculator.F_of_phi(phi=phi).astype(dtype=np.complex64) for phi in phi_array])
#
#     values = phi_array.shape[0] * loops
#     dT = time.time() - t1
#     print("time for {} values: {}".format(values, dT))
#     print("{} values per second".format(values / dT))

if __name__ == "__main__":
    t1 = time.time()
    for target in range(100):
        print(target)
        convex_polygon_arr = generate_target_polygon(max_edge=3)
        convex_polygon_tuple = array_to_tuples(convex_polygon_arr)
        polygon_calculator = Fcalculator(points=convex_polygon_tuple)

        dphi = 0.0001
        har = 1.0 / 180.0 * np.pi  # hole_half_angle_rad
        mac = 1.0 / 180.0 * np.pi  # max_angle_of_view_cut_rad
        phi_array = np.concatenate((np.arange(0 + har, np.pi / 2 - mac, dphi),
                                    np.arange(np.pi / 2 + har, np.pi - mac, dphi)))
        polygon_scatter_res = np.array(polygon_calculator.F_of_phi(phi=phi_array).astype(dtype=np.complex64))

        # print(convex_polygon_arr.shape)
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 14))
        # ax1.plot(phi_array, polygon_scatter_res.real, "-b", label="real_polygon")
        # ax1.plot(phi_array, polygon_scatter_res.imag, "-y", label="imag_polygon")
        # ax1.plot(phi_array, np.abs(polygon_scatter_res), "-y", label="abs_polygon")
        # ax2.fill(convex_polygon_arr.transpose()[0], convex_polygon_arr.transpose()[1])
        # ax2.set_xlim((-50, 50))
        # ax2.set_ylim((-50, 50))
        # ax2.set_aspect(aspect=1.0)
        # plt.show()
    print("Time: {:0.1f}".format(time.time() - t1))
# if __name__ == "__main__":
#     print("run polygon 2d helper")
#     import input_fn.input_fn_2d.data_gen_2dt.util_2d.triangle_2d_helper as triangle_2d_helper
#
#     # points = [(0, 0),  (0, 70),  (-30, 30), (70, 0),(4, 5)]
#     rng = random.Random()
#     for i in range(10):
#         convex_polygon_arr = generate_target_polygon(max_edge=3)
#         points = array_to_tuples(convex_polygon_arr)
#
#         phi_array = np.arange(np.pi / 2 - 1.5, np.pi / 2 + 1.5, 0.01)
#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 14))
#
#         triangel_calculator = triangle_2d_helper.ScatterCalculator2D(p1=[points[0][0], points[0][1]],
#                                                              p2=[points[1][0], points[1][1]],
#                                                              p3=[points[2][0], points[2][1]])
#         triangle_scatter_res = triangel_calculator.call_on_array(phi_array).astype(dtype=np.complex64)
#         plt.suptitle("p0={:1.1f},{:1.1f};p1={:1.1f},{:1.1f};p2={:1.1f},{:1.1f};".format(points[0][0], points[0][1],
#                                                                                         points[1][0], points[1][1],
#                                                                                         points[2][0], points[2][1]))
#         ax1.plot(phi_array, triangle_scatter_res.real, "+r", label="real_triangle")
#         ax1.plot(phi_array, triangle_scatter_res.imag, "+g", label="imag_triangle")
#         # print("phi_array:", phi_array)
#         # print("##############")
#         # print("triangle scatter res:", triangle_scatter_res)
#         polygon_calculator = ScatterCalculator2D(points)
#
#         polygon_scatter_res = polygon_calculator.F_of_phi(phi=phi_array).astype(dtype=np.complex64)
#         # polygon_scatter_res = np.array(
#         #     [polygon_calculator.F_of_phi(phi=phi).astype(dtype=np.complex64) for phi in phi_array])
#         # print(polygon_scatter_res, polygon_scatter_res.shape)
#         np.set_printoptions(precision=6, suppress=True)
#         # print("polygon_scatter_res:", polygon_scatter_res)
#         # print("##############")
#
#         q2 = np.array(
#             [np.dot(polygon_calculator.q_of_phi(phi=phi), polygon_calculator.q_of_phi(phi=phi)) for phi in phi_array])
#         q2_reverse = np.array(
#             [np.dot(polygon_calculator.q_of_phi(phi=phi), polygon_calculator.q_of_phi(phi=phi)) for phi in
#              reversed(phi_array)])
#         ax1.plot(phi_array, polygon_scatter_res.real, "-b", label="real_polygon")
#         ax1.plot(phi_array, polygon_scatter_res.imag, "-y", label="imag_polygon")
#         # ax1.plot(phi_array, np.abs(polygon_scatter_res), "-y", label="abs_polygon")
#
#         ax1.plot(phi_array, q2 - q2_reverse, label="q^2")
#         ax1.legend(loc=4)
#
#         points_array = np.array([list(i) for i in points])
#         # print(points_array.shape)
#         ax2.fill(points_array.transpose()[0], points_array.transpose()[1])
#         plt.show()



def make_scatter_data(points, epsilon=0.002, phi_arr=None, dphi=0.001, complex_phi=False):
    if phi_arr is None:
        phi_arr = np.arange(0, np.pi, dphi)
    fcalc = Fcalculator(p1=points[0], p2=points[1], p3=points[2], epsilon=np.array(epsilon), complex_phi=complex_phi)
    f = fcalc.call_on_array(phi_arr)
    # print(f.real)
    # print(f.imag)
    if not complex_phi:
        one_data = np.stack((phi_arr, f.real, f.imag), axis=0).astype(np.float32)
    else:
        one_data = np.stack((phi_arr.real, phi_arr.imag, f.real, f.imag), axis=0).astype(np.float32)

    return one_data




def limited_f(phi, p1=np.array([0.0, 0.0]), p2=np.array([1.0, 0.0]), p3=np.array([0.0, 1.0]), epsilon=0.001,
              no_check=False):
    """legacy version of case_f"""
    if not no_check:  # skip check if valid input is ensured for better performance
        assert np.sum(np.square(np.abs(p1 - p2))) > (10 * epsilon) ** 2
        assert np.sum(np.square(np.abs(p2 - p3))) > (10 * epsilon) ** 2
        assert np.sum(np.square(np.abs(p3 - p1))) > (10 * epsilon) ** 2

        if phi - epsilon < 0:
            logging.warning("input phi is smaller zero; phi: {}".format(phi))
            return np.nan
        elif phi + epsilon > np.pi / 2.0 > phi - epsilon:
            logging.warning("input phi to close to pi/2; phi: {}".format(phi))
            return np.nan
        elif phi - epsilon > np.pi:
            logging.warning("input phi greater pi; phi: {}".format(phi))
            return np.nan

    a = np.cos(phi)
    b = np.sin(phi) - 1.0

    f = 1.0 / (a * b * (b - a)) * (b * (np.exp(1.0j * a) - 1) - a * (np.exp(1.0j * b) - 1.0))

    return f


def case_f(phi, p1=np.array([0.0, 0.0]), p2=np.array([1.0, 0.0]), p3=np.array([0.0, 1.0]), epsilon=0.001,
           no_check=False):
    """legacy version of multi_triangle_f"""
    if not no_check:  # skip check if valid input is ensured for better performance
        assert np.sum(np.square(np.abs(p1 - p2))) > (10 * epsilon) ** 2
        assert np.sum(np.square(np.abs(p2 - p3))) > (10 * epsilon) ** 2
        assert np.sum(np.square(np.abs(p3 - p1))) > (10 * epsilon) ** 2
        if phi < 0 or phi > np.pi:
            logging.error("input phi is out of range; phi: {}".format(phi))
            return np.nan

    a = np.cos(phi)
    b = np.sin(phi) - 1.0

    if np.abs(a - b) > epsilon and np.abs(a - epsilon) > 0 and np.abs(b - epsilon) > 0:
        logging.info("case1, a!=b, a!=0, b!=0")
        f_ = 1.0 / (a * b * (b - a)) * (b * (np.exp(1.0j * a) - 1) -
                                        a * (np.exp(1.0j * b) - 1.0))
    elif np.abs(a - b) > epsilon and np.abs(b - epsilon) > 0:
        logging.info("case2, a!=b, a=0, b!=0")
        f_ = 1.0j / b - 1 / b ** 2 * (np.exp(1.0j * b) - 1.0)
    elif np.abs(a - b) > epsilon and np.abs(a - epsilon) > 0:
        logging.info("case3, a!=b, b=0, a!=0")
        f_ = 1.0j / a - 1 / a ** 2 * (np.exp(1.0j * a) - 1.0)
    elif np.abs(a) <= epsilon and np.abs(b) - epsilon <= 0:
        assert np.abs(a - b) <= epsilon  # a and b have same monotonie for phi > pi
        logging.info("case4, a=b, a=0, b=0")
        f_ = 0.5
    elif np.abs(a - b) <= epsilon and np.abs(a - epsilon) > 0 and np.abs(b - epsilon):
        logging.info("case5, a=b, b!=0, a!=0")
        f_ = np.exp(1.0j * a) / (1.0j * a) + (np.exp(1.0j * a) - 1.0) / a ** 2
    else:
        logging.error("unexpected values for a and b!; a={}; b={}".format(a, b))
        return np.nan

    return f_
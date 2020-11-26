import matplotlib
import numpy as np
import pytest
import logging
import tensorflow as tf
import shapely.geometry as geometry
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from input_fn.input_fn_2d.data_gen_2dt.util_2d import misc, misc_tf, convert, object_generator


def test_center_triangle():
    rng = np.random.Generator(np.random.PCG64(1235))
    for sample in range(10):
        triangle = object_generator.generate_target_triangle(rng=rng)
        triangle_array = np.array([misc.center_triangle_list(triangle[0], triangle[1], triangle[2])], dtype=np.float32)
        triangle_tf_centered = misc_tf.center_triangle(triangle)
        assert np.allclose(triangle_tf_centered, triangle_array)

def test_flip():
    rng = np.random.Generator(np.random.PCG64(1235))
    triangle_list = []
    for sample in range(10):
        triangle_list.append(object_generator.generate_target_triangle(rng=rng))

    batch = np.stack(triangle_list, axis=0)
    batch_x_flip = misc_tf.flip_along_axis(batch, 'x')
    batch_xy_flip = misc_tf.flip_along_axis(batch_x_flip, 'y')
    assert tf.reduce_sum(batch) == -1.0 * tf.reduce_sum(batch_xy_flip)
    pass




if __name__ == '__main__':
    # logging.basicConfig(level="WARNING")
    # pytest.main(["test_misc.py", "-v", "-s"])
    logging.basicConfig(level="INFO")
    # pytest.main(["test_misc.py", "-k test_get_area_triangle", "-v", "-s"])
    # test_center_triangle()
    test_flip()

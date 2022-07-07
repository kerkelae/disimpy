from bitarray import test
import numba
import math
import numpy as np
from numba import cuda
import numpy.testing as npt
from disimpy.disimpy import simulations

def test__cuda_crossing():
    @cuda.jit()
    def test_kernel(triangle, r0, step, step_l, epsilon):
        thread_id = cuda.grid(1)
        if thread_id >= r0.shape[0]:
            return
        r0 = r0[thread_id]
        step = step[thread_id]
        triangle = triangle[thread_id]
        d = simulations._cuda_ray_triangle_intersection_check(triangle, r0, step)
        if d > 0 and d < step_l:
            normal = cuda.local.array(3, numba.float64)
            simulations._cuda_triangle_normal(triangle, normal)
            simulations._cuda_crossing(r0, step, d, normal, epsilon)
        return

    triangle = np.zeros((3, 3))
    triangle[0, 0] = 1
    triangle[0, 2] = 1
    triangle[1, 0] = -1
    triangle[1, 2] = 1
    triangle[2, 1] = 1
    triangle[2, 2] = 1
    triangle = triangle[np.newaxis, ...]
    r0 = np.array([0, 0, 0])[np.newaxis, ...]
    step = np.array([0, 1/math.sqrt(5), 2/math.sqrt(5)])[np.newaxis, ...]
    step_l = 1.5
    epsilon = 1e-10
    stream = cuda.stream()
    test_kernel[1, 128, stream](triangle, r0, step, step_l, epsilon)
    stream.synchronize()
    npt.assert_almost_equal(r0, np.array([[0.0, 0.5, 1 + epsilon]]))
    return

test__cuda_crossing()
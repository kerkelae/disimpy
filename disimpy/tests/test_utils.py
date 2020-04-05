import os
import matplotlib
import matplotlib.pyplot as plt
import numpy.testing as npt

from .. import utils

def test_show_traj():
    traj_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'example_traj.txt')
    fig, ax = utils.show_traj(traj_file, title=None, show=False)
    npt.assert_equal(isinstance(fig, matplotlib.figure.Figure), True)
    plt.close(fig)
    return

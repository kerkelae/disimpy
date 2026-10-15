import os

import pytest

from . import test_gradients as test_gradients
from . import test_simulations as test_simulations
from . import test_utils


def test_all():
    """Execute all tests in tests package."""
    return pytest.main([os.path.dirname(test_utils.__file__)])

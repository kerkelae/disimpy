import os
import pytest
from . import test_utils, test_simulations, test_gradients


def test_all():
    """Execute all tests in tests package."""
    return pytest.main([os.path.dirname(test_utils.__file__)])

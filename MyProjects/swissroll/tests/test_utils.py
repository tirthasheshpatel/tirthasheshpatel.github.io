import pytest
import numpy as np
from swissroll.utils import ecdf


def test_ecdf():
    samples = np.array([1.0, 5.0, 2.0, 4.0, 3.0])

    x_ecdf_expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_ecdf_expected = np.array([1.0 / 5.0, 2.0 / 5.0, 3.0 / 5.0, 4.0 / 5.0, 5.0 / 5.0])

    x_ecdf, y_ecdf = ecdf(samples)

    assert np.all(x_ecdf == x_ecdf_expected)
    assert np.all(y_ecdf == y_ecdf_expected)


if __name__ == "__main__":
    pytest.main()

import os

import numpy as np
import pytest

from pneumonia.preprocess import compute_standard_params


@pytest.fixture
def temp_dir():
    # Create a temporary directory with some test images
    out_dir = 'test_dir'
    os.makedirs(out_dir, exist_ok=True)
    shape = (10, 10)
    yield out_dir, shape
    # Remove the temporary directory after the test
    for filename in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, filename))
    os.rmdir(out_dir)


def test_compute_standard_params_std_zero(temp_dir):
    out_dir, shape = temp_dir
    for i in range(5):
        img = np.full(shape, i, dtype=np.float32)
        np.save(os.path.join(out_dir, f'test_img_{i}.npy'), img)

    params = compute_standard_params(out_dir, shape)
    # The test images are filled with constant values, so the mean should be equal to the value,
    # and the standard deviation should be 0
    for i in range(5):
        assert np.isclose(params.pixel_mean, i)
        assert np.isclose(params.pixel_std, 0)


def test_compute_standard_params_std_one(temp_dir):
    out_dir, shape = temp_dir
    # Test for standard deviation of 1
    for i in range(5):
        img = np.random.normal(loc=0, scale=1, size=shape).astype(np.float32)
        np.save(os.path.join(out_dir, f'test_img_{i}.npy'), img)

    params = compute_standard_params(out_dir, shape)
    assert np.isclose(params.pixel_mean, 0, atol=0.1)
    assert np.isclose(params.pixel_std, 1, atol=0.1)

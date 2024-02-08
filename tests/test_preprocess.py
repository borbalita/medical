import os
import unittest

import numpy as np


class TestComputeStandardParams(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory with some test images
        self.out_dir = 'test_dir'
        os.makedirs(self.out_dir, exist_ok=True)
        self.shape = (10, 10)

    def tearDown(self):
        # Remove the temporary directory after the test
        for filename in os.listdir(self.out_dir):
            os.remove(os.path.join(self.out_dir, filename))
        os.rmdir(self.out_dir)

    def test_compute_standard_params_std_zero(self):
        for i in range(5):
            img = np.full(self.shape, i, dtype=np.float32)
            np.save(os.path.join(self.out_dir, f'test_img_{i}.npy'), img)

        params = compute_standard_params(self.out_dir, self.shape)
        # The test images are filled with constant values, so the mean should be equal to the value,
        # and the standard deviation should be 0
        for i in range(5):
            self.assertAlmostEqual(params.pixel_mean, i)
            self.assertAlmostEqual(params.pixel_std, 0)

    def test_compute_standard_params_std_one(self):
        # Test for standard deviation of 1
        for i in range(5):
            img = np.random.normal(loc=0, scale=1, size=self.shape).astype(np.float32)
            np.save(os.path.join(self.out_dir, f'test_img_{i}.npy'), img)

        params = compute_standard_params(self.out_dir, self.shape)
        self.assertAlmostEqual(params.pixel_mean, 0, places=1)
        self.assertAlmostEqual(params.pixel_std, 1, places=1)

if __name__ == '__main__':
    unittest.main()
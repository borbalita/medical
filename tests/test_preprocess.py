import math
import os
import tempfile

import numpy as np

from pneumonia.preprocess import StandardParams, compute_standard_params


def test_compute_standard_params():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create a subdirectory for the 'train' data
        train_dir = os.path.join(tmpdirname, 'train', '0')
        os.makedirs(train_dir)

        # Create some test images
        for i in range(1, 4):
            img = np.full((100, 100), i, dtype=np.float32)
            np.save(os.path.join(train_dir, f'image_{i}.npy'), img)

        # Call the function
        params = compute_standard_params(tmpdirname, (100, 100))

        # Check the results
        assert isinstance(params, StandardParams)
        assert math.isclose(params.mean, 2)
        assert math.isclose(params.std, 0.8165, rel_tol=1e-4)

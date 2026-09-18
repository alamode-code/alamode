import unittest

import numpy as np

from analyzer.interpolate import Interpolator


class TimeReversalUnfoldTest(unittest.TestCase):
    def test_spatial_stars_stay_separate(self):
        points = np.array([[0, 0, 0], [0.25, 0, 0], [0.5, 0, 0], [-0.25, 0, 0]])
        interpol = Interpolator([4, 1, 1], points, np.ones(4), np.eye(3)[None])
        np.testing.assert_array_equal(interpol.bz2irb, [0, 1, 2, 3])

    def test_time_reversed_rotated_images(self):
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        rotations = np.array([np.linalg.matrix_power(rotation, i) for i in range(4)])
        grid = np.array([4, 4, 4])
        indices = np.indices(grid).reshape(3, -1).T
        seen = set()
        representatives, weights, expected = [], [], np.empty(64, dtype=int)
        for index in indices:
            key = tuple(index)
            if key in seen:
                continue
            star = {
                tuple((sign * rot.dot(index)) % grid)
                for sign in (-1, 1)
                for rot in rotations
            }
            label = len(representatives)
            representatives.append(index / grid)
            weights.append(len(star))
            for member in star:
                expected[np.ravel_multi_index(member, grid)] = label
            seen.update(star)
        interpol = Interpolator(grid, np.array(representatives), weights, rotations)
        np.testing.assert_array_equal(interpol.bz2irb, expected)

    def test_inconsistent_weight_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "expected weight"):
            Interpolator([4, 1, 1], np.array([[0.25, 0, 0]]), [3], np.eye(3)[None])


if __name__ == "__main__":
    unittest.main()

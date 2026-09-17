import unittest

import numpy as np

from bubble_cv.detection import _contour_consensus_qc


def ellipse_points(cx, cy, major, minor, n, phase=0.0):
    """Generate ordered points exactly on an ellipse."""
    t = np.linspace(
        0.0,
        2.0 * np.pi,
        n,
        endpoint=False,
    ) + phase

    a = major / 2.0
    b = minor / 2.0

    x = cx + a * np.cos(t)
    y = cy + b * np.sin(t)

    return np.column_stack(
        [x, y]
    ).astype(np.float32)


def as_contour(points):
    return np.asarray(
        points,
        dtype=np.float32,
    ).reshape(-1, 1, 2)


class TestContourConsensusQC(unittest.TestCase):

    def test_clean_ellipse_passes(self):
        pts = ellipse_points(
            cx=100.0,
            cy=90.0,
            major=80.0,
            minor=60.0,
            n=100,
        )

        result = _contour_consensus_qc(
            as_contour(pts)
        )

        self.assertTrue(
            result["valid"]
        )

        self.assertEqual(
            result["n_points"],
            100,
        )

        self.assertGreaterEqual(
            result["support_fraction"],
            0.50,
        )

    def test_minority_contamination_still_passes(self):
        good = ellipse_points(
            cx=100.0,
            cy=90.0,
            major=80.0,
            minor=60.0,
            n=100,
        )

        # 40 deterministic outliers:
        # 100/140 = 71.4% of points still support
        # the physical ellipse.
        rng = np.random.default_rng(
            12345
        )

        outliers = np.column_stack(
            [
                rng.uniform(
                    30.0,
                    170.0,
                    40,
                ),
                rng.uniform(
                    20.0,
                    160.0,
                    40,
                ),
            ]
        ).astype(np.float32)

        pts = np.vstack(
            [good, outliers]
        )

        result = _contour_consensus_qc(
            as_contour(pts)
        )

        self.assertTrue(
            result["valid"]
        )

        self.assertGreaterEqual(
            result["support_fraction"],
            0.50,
        )

    def test_no_majority_ellipse_fails(self):
        # Two spatially separated, individually plausible
        # ellipses. Each contributes only 45% of the points.
        # Therefore neither has majority support.
        left = ellipse_points(
            cx=70.0,
            cy=90.0,
            major=70.0,
            minor=55.0,
            n=45,
        )

        right = ellipse_points(
            cx=210.0,
            cy=90.0,
            major=70.0,
            minor=55.0,
            n=45,
            phase=0.031,
        )

        # Remaining 10% deliberately remote.
        noise = np.column_stack(
            [
                np.linspace(
                    120.0,
                    160.0,
                    10,
                ),
                np.linspace(
                    170.0,
                    190.0,
                    10,
                ),
            ]
        ).astype(np.float32)

        pts = np.vstack(
            [
                left,
                right,
                noise,
            ]
        )

        result = _contour_consensus_qc(
            as_contour(pts)
        )

        self.assertFalse(
            result["valid"]
        )

        self.assertEqual(
            result["n_points"],
            100,
        )

        self.assertLess(
            result["support_fraction"],
            0.50,
        )

    def test_too_few_points_fails_explicitly(self):
        pts = np.array(
            [
                [10.0, 10.0],
                [11.0, 10.0],
                [12.0, 11.0],
                [11.0, 12.0],
            ],
            dtype=np.float32,
        )

        result = _contour_consensus_qc(
            as_contour(pts)
        )

        self.assertFalse(
            result["valid"]
        )

        self.assertEqual(
            result["n_points"],
            4,
        )

        self.assertEqual(
            result["reason"],
            "fewer_than_5_points",
        )

    def test_result_is_deterministic(self):
        good = ellipse_points(
            cx=100.0,
            cy=90.0,
            major=80.0,
            minor=60.0,
            n=80,
        )

        rng = np.random.default_rng(
            54321
        )

        outliers = np.column_stack(
            [
                rng.uniform(
                    30.0,
                    170.0,
                    30,
                ),
                rng.uniform(
                    20.0,
                    160.0,
                    30,
                ),
            ]
        ).astype(np.float32)

        contour = as_contour(
            np.vstack(
                [good, outliers]
            )
        )

        result_a = _contour_consensus_qc(
            contour
        )

        result_b = _contour_consensus_qc(
            contour
        )

        self.assertEqual(
            result_a,
            result_b,
        )


if __name__ == "__main__":
    unittest.main()

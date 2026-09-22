"""
RC3 regression tests: BODY candidates have strict priority over fallbacks.

Fallback candidates may rescue a frame only when no normal BODY candidate
survives. They must never compete with or displace a normal BODY candidate.
"""

import unittest

import bubble_cv.detection as detection


def make_candidate(score, radius_mm, rmse, px_to_mm=34.08):
    equiv_diameter_px = radius_mm * 2.0 * px_to_mm

    props = {
        "equiv_diameter_px": equiv_diameter_px,
    }

    fit_quality = {
        "residual_rmse": rmse,
    }

    return (
        score,
        props,
        None,
        fit_quality,
        0,
    )


class TestBodyFallbackPriority(unittest.TestCase):

    def test_primary_candidate_has_absolute_priority(self):
        """
        A fallback with a much larger score and better RMSE must NOT
        displace an existing normal BODY candidate.
        """

        primary = make_candidate(
            score=100.0,
            radius_mm=1.00,
            rmse=0.05,
        )

        fallback = make_candidate(
            score=100000.0,
            radius_mm=1.40,
            rmse=0.001,
        )

        result = detection._select_body_candidate_pool(
            primary_candidates=[primary],
            fallback_candidates=[fallback],
            px_to_mm=34.08,
        )

        self.assertIs(
            result["selected"],
            primary,
            "Fallback must never compete with an available primary BODY candidate",
        )

        self.assertFalse(result["used_fallback"])


    def test_fallback_is_used_when_primary_pool_is_empty(self):
        """
        A fallback is allowed to rescue the frame when no normal BODY
        candidate survives.
        """

        fallback = make_candidate(
            score=100.0,
            radius_mm=1.00,
            rmse=0.05,
        )

        result = detection._select_body_candidate_pool(
            primary_candidates=[],
            fallback_candidates=[fallback],
            px_to_mm=34.08,
        )

        self.assertIs(result["selected"], fallback)
        self.assertTrue(result["used_fallback"])


if __name__ == "__main__":
    unittest.main()

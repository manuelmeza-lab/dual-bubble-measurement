"""
tests/test_body_selector.py
===========================
Deterministic unit tests for the conservative body-candidate selector
introduced in Fix 2 (_select_body_candidate in bubble_cv/detection.py).

Design goals
------------
* stdlib only - no pytest, no external images.
* All candidate tuples are synthetic; no frame processing occurs.
* Tests are repeatable across Python versions.

Candidate tuple layout (as produced by _detect_drop_in_roi loop):
    (score, props_global, global_cnt, fit_quality, body_start_y_global)

where:
    score              : float - historical scoring function value
    props_global       : dict  - must contain 'equiv_diameter_px'
    global_cnt         : any   - opaque; we use None in tests
    fit_quality        : dict  - must contain 'residual_rmse'
    body_start_y_global: any   - opaque; we use 0 in tests
"""

from __future__ import annotations

import math
import sys
import unittest

sys.path.insert(0, ".")

from bubble_cv.detection import (
    _select_body_candidate,
    BODY_CONSENSUS_RADIUS_EPS_MM,
    BODY_CONSENSUS_MIN_SUPPORT,
    BODY_RMSE_RATIO_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helper: build a minimal synthetic candidate tuple
# ---------------------------------------------------------------------------

def _make_candidate(
    score: float,
    equiv_diameter_px: float,
    residual_rmse,
) -> tuple:
    """Return a 5-tuple compatible with the candidate list in _detect_drop_in_roi."""
    props = {"equiv_diameter_px": equiv_diameter_px}
    fit_quality = {"residual_rmse": residual_rmse}
    return (score, props, None, fit_quality, 0)


# Convenience: radius in mm -> equiv_diameter_px for a given px_to_mm
def _r_to_dpx(radius_mm: float, px_to_mm: float) -> float:
    return radius_mm * 2.0 * px_to_mm


# Default px_to_mm used across most tests
_PX_TO_MM = 50.0


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestBodySelectorNoSwitch(unittest.TestCase):
    """Cases where Top-1 must be returned unchanged."""

    # Test 1 - single candidate: no switch possible
    def test_01_single_candidate_returns_top1(self):
        """No switch with a single candidate (nothing to compare against)."""
        cand = _make_candidate(score=1000.0,
                               equiv_diameter_px=_r_to_dpx(1.0, _PX_TO_MM),
                               residual_rmse=0.05)
        candidates = [cand]
        result = _select_body_candidate(candidates, px_to_mm=_PX_TO_MM)
        self.assertIs(result["selected"], cand,
                      "Single candidate must be returned as-is.")
        self.assertFalse(result["switched"])

    # Test 2 - support < BODY_CONSENSUS_MIN_SUPPORT: no switch
    def test_02_no_switch_when_support_lt_min(self):
        """support_consensus = 1 (< 2): no switch, even with large rmse_ratio."""
        # Two candidates far apart in radius (> EPS) -> each has support = 1
        r1, r2 = 1.0, 1.1   # gap = 0.1 mm >> 0.010 mm
        top1 = _make_candidate(score=2000.0,
                               equiv_diameter_px=_r_to_dpx(r1, _PX_TO_MM),
                               residual_rmse=0.50)   # high RMSE
        alt  = _make_candidate(score=1000.0,
                               equiv_diameter_px=_r_to_dpx(r2, _PX_TO_MM),
                               residual_rmse=0.01)   # low RMSE -> rmse_ratio = 50
        result = _select_body_candidate([top1, alt], px_to_mm=_PX_TO_MM)
        self.assertIs(result["selected"], top1)
        self.assertFalse(result["switched"])

    # Test 3 - rmse_ratio exactly == 2.0: strictly >, no switch
    def test_03_no_switch_rmse_ratio_exactly_threshold(self):
        """rmse_ratio == 2.0 must NOT trigger switch (condition is strictly > 2.0)."""
        r = 1.0
        rmse_consensus = 0.10
        rmse_top1      = rmse_consensus * BODY_RMSE_RATIO_THRESHOLD   # = 0.20 exactly
        top1 = _make_candidate(score=2000.0,
                               equiv_diameter_px=_r_to_dpx(r, _PX_TO_MM),
                               residual_rmse=rmse_top1)
        alt  = _make_candidate(score=1000.0,
                               equiv_diameter_px=_r_to_dpx(r, _PX_TO_MM),
                               residual_rmse=rmse_consensus)
        result = _select_body_candidate([top1, alt], px_to_mm=_PX_TO_MM)
        self.assertIs(result["selected"], top1)
        self.assertFalse(result["switched"])
        # rmse_ratio should have been computed
        self.assertAlmostEqual(result["rmse_ratio"], BODY_RMSE_RATIO_THRESHOLD, places=10)

    # Test 4 - rmse_ratio < 2.0: no switch
    def test_04_no_switch_rmse_ratio_below_threshold(self):
        """rmse_ratio = 1.5 < 2.0: no switch."""
        r = 1.0
        rmse_consensus = 0.10
        rmse_top1      = 0.15   # ratio = 1.5
        top1 = _make_candidate(score=2000.0,
                               equiv_diameter_px=_r_to_dpx(r, _PX_TO_MM),
                               residual_rmse=rmse_top1)
        alt  = _make_candidate(score=1000.0,
                               equiv_diameter_px=_r_to_dpx(r, _PX_TO_MM),
                               residual_rmse=rmse_consensus)
        result = _select_body_candidate([top1, alt], px_to_mm=_PX_TO_MM)
        self.assertIs(result["selected"], top1)
        self.assertFalse(result["switched"])

    # Test 8 - px_to_mm=None: selector disabled
    def test_08_px_to_mm_none_returns_top1(self):
        """px_to_mm=None disables the selector; Top-1 must be returned."""
        top1 = _make_candidate(score=3000.0,
                               equiv_diameter_px=100.0,
                               residual_rmse=0.50)
        alt  = _make_candidate(score=1000.0,
                               equiv_diameter_px=100.0,
                               residual_rmse=0.01)
        result = _select_body_candidate([top1, alt], px_to_mm=None)
        self.assertIs(result["selected"], top1)
        self.assertFalse(result["switched"])
        self.assertIsNone(result["top1_radius_mm"])
        self.assertIsNone(result["rmse_ratio"])

    # Test 9a - residual_rmse None: no switch, no exception
    def test_09a_rmse_none_top1_no_switch(self):
        """Top-1 residual_rmse=None: no switch."""
        r = 1.0
        top1 = _make_candidate(score=2000.0,
                               equiv_diameter_px=_r_to_dpx(r, _PX_TO_MM),
                               residual_rmse=None)
        alt  = _make_candidate(score=1000.0,
                               equiv_diameter_px=_r_to_dpx(r, _PX_TO_MM),
                               residual_rmse=0.01)
        result = _select_body_candidate([top1, alt], px_to_mm=_PX_TO_MM)
        self.assertIs(result["selected"], top1)
        self.assertFalse(result["switched"])

    # Test 9b - residual_rmse NaN: no switch, no exception
    def test_09b_rmse_nan_top1_no_switch(self):
        """Top-1 residual_rmse=NaN: no switch."""
        r = 1.0
        top1 = _make_candidate(score=2000.0,
                               equiv_diameter_px=_r_to_dpx(r, _PX_TO_MM),
                               residual_rmse=float("nan"))
        alt  = _make_candidate(score=1000.0,
                               equiv_diameter_px=_r_to_dpx(r, _PX_TO_MM),
                               residual_rmse=0.01)
        result = _select_body_candidate([top1, alt], px_to_mm=_PX_TO_MM)
        self.assertIs(result["selected"], top1)
        self.assertFalse(result["switched"])

    # Test 9c - residual_rmse inf (consensus): no switch, no exception
    def test_09c_rmse_inf_consensus_no_switch(self):
        """Consensus residual_rmse=inf: no switch (division would be unsafe)."""
        r = 1.0
        top1 = _make_candidate(score=2000.0,
                               equiv_diameter_px=_r_to_dpx(r, _PX_TO_MM),
                               residual_rmse=0.50)
        alt  = _make_candidate(score=1000.0,
                               equiv_diameter_px=_r_to_dpx(r, _PX_TO_MM),
                               residual_rmse=math.inf)
        result = _select_body_candidate([top1, alt], px_to_mm=_PX_TO_MM)
        self.assertIs(result["selected"], top1)
        self.assertFalse(result["switched"])

    # Test 10 - no switch: returned tuple is exactly Top-1
    def test_10_no_switch_returns_exact_top1_tuple(self):
        """When no switch fires, the returned tuple must be the identical Top-1 object."""
        top1 = _make_candidate(score=5000.0,
                               equiv_diameter_px=_r_to_dpx(1.0, _PX_TO_MM),
                               residual_rmse=0.30)
        alt  = _make_candidate(score=2000.0,
                               equiv_diameter_px=_r_to_dpx(1.0 + 0.10, _PX_TO_MM),
                               residual_rmse=0.01)
        result = _select_body_candidate([top1, alt], px_to_mm=_PX_TO_MM)
        # support < 2 because radii are far apart -> no switch
        self.assertIs(result["selected"], top1)


class TestBodySelectorSwitch(unittest.TestCase):
    """Cases where the switch must fire."""

    # Test 5 - switch fires: support >= 2 AND rmse_ratio > 2.0
    def test_05_switch_fires_when_conditions_met(self):
        """Switch when support >= 2 AND rmse_ratio > 2.0."""
        r_close = 1.0      # consensus cluster
        r_far   = 1.2      # top-1, isolated

        # top-1: highest score, bad RMSE, isolated radius
        top1 = _make_candidate(score=3000.0,
                               equiv_diameter_px=_r_to_dpx(r_far, _PX_TO_MM),
                               residual_rmse=0.40)
        # two consensus candidates close together, good RMSE
        alt1 = _make_candidate(score=1500.0,
                               equiv_diameter_px=_r_to_dpx(r_close, _PX_TO_MM),
                               residual_rmse=0.05)
        alt2 = _make_candidate(score=1000.0,
                               equiv_diameter_px=_r_to_dpx(r_close + 0.005, _PX_TO_MM),
                               residual_rmse=0.06)

        result = _select_body_candidate([top1, alt1, alt2], px_to_mm=_PX_TO_MM)
        self.assertTrue(result["switched"])
        # Must not return top1
        self.assertIsNot(result["selected"], top1)
        # rmse_ratio > 2.0
        self.assertGreater(result["rmse_ratio"], BODY_RMSE_RATIO_THRESHOLD)

    # Test 6 - consensus = max support + min RMSE tie-break
    def test_06_consensus_max_support_then_min_rmse(self):
        """Consensus is selected by max support, then min RMSE for tie."""
        r_group = 1.0
        r_lone  = 1.2   # top-1, isolated (support=1)

        top1 = _make_candidate(score=3000.0,
                               equiv_diameter_px=_r_to_dpx(r_lone, _PX_TO_MM),
                               residual_rmse=0.50)
        # Two candidates in the group - same radius, different RMSE
        better_rmse = _make_candidate(score=1200.0,
                                      equiv_diameter_px=_r_to_dpx(r_group, _PX_TO_MM),
                                      residual_rmse=0.02)
        worse_rmse  = _make_candidate(score=1000.0,
                                      equiv_diameter_px=_r_to_dpx(r_group, _PX_TO_MM),
                                      residual_rmse=0.08)

        result = _select_body_candidate([top1, better_rmse, worse_rmse], px_to_mm=_PX_TO_MM)
        self.assertTrue(result["switched"])
        # The selected candidate must be the one with the lower RMSE in the group
        self.assertIs(result["selected"], better_rmse)

    # Test 7 - do NOT use global min-RMSE when another group has more support
    def test_07_does_not_use_global_min_rmse_over_higher_support(self):
        """A lone candidate with the globally best RMSE must NOT win over a
        group with higher support."""
        r_group = 1.0    # two candidates -> support = 2
        r_lone  = 1.2    # one candidate -> support = 1, but rmse = 0.001 (best global)

        top1 = _make_candidate(score=3000.0,
                               equiv_diameter_px=_r_to_dpx(r_group, _PX_TO_MM),
                               residual_rmse=0.60)   # top-1 is in the group
        grp2 = _make_candidate(score=1500.0,
                               equiv_diameter_px=_r_to_dpx(r_group + 0.003, _PX_TO_MM),
                               residual_rmse=0.07)
        lone = _make_candidate(score=800.0,
                               equiv_diameter_px=_r_to_dpx(r_lone, _PX_TO_MM),
                               residual_rmse=0.001)  # globally best RMSE, but isolated

        result = _select_body_candidate([top1, grp2, lone], px_to_mm=_PX_TO_MM)
        # The group (support=2) should win over the lone (support=1)
        self.assertEqual(result["support"], 2)
        # The selected must NOT be the lone candidate
        self.assertIsNot(result["selected"], lone)


class TestBodySelectorEpsilon(unittest.TestCase):
    """Boundary conditions for the radius tolerance epsilon."""

    # Test 11a - exactly 0.010 mm counts as within support
    def test_11a_exact_eps_counts_as_within(self):
        """abs(radius_i - radius_j) == BODY_CONSENSUS_RADIUS_EPS_MM -> counts as support.

        Construction avoids _r_to_dpx() round-trip to prevent float rounding.
        With px_to_mm=50:
            c1: dpx=0.0   -> radius = 0.0/(2*50) = 0.0   (exact)
            c2: dpx=1.0   -> radius = 1.0/(2*50) = 0.01  (exact in IEEE-754)
            diff = 0.01 - 0.0 = 0.01 = BODY_CONSENSUS_RADIUS_EPS_MM (exact)
        Note: _r_to_dpx(1.0+EPS, px) introduces rounding (1.01*100 != 101 in float).
        """
        _PX = 50.0

        # top-1: isolated radius (r=3.0), high RMSE
        top1 = (3000.0, {"equiv_diameter_px": 300.0}, None, {"residual_rmse": 0.50}, 0)

        # c1: radius = 0.0 (exact)
        c1 = (1200.0, {"equiv_diameter_px": 0.0}, None, {"residual_rmse": 0.05}, 0)

        # c2: radius = 1.0/(2*50) = 0.01 (exact); diff from c1 = 0.01 == EPS
        c2 = (1000.0, {"equiv_diameter_px": 1.0}, None, {"residual_rmse": 0.06}, 0)

        result = _select_body_candidate([top1, c1, c2], px_to_mm=_PX)
        # c1 and c2 are exactly eps apart -> mutual support = 2 each
        # top1 is far away (r=3.0) -> support = 1
        # rmse_ratio = 0.50 / 0.05 = 10.0 > 2.0 -> switch must fire
        self.assertTrue(result["switched"],
                        "Candidates with radius diff == BODY_CONSENSUS_RADIUS_EPS_MM "
                        "must count as mutual support and trigger switch.")
        self.assertEqual(result["support"], 2)

    # Test 11b - clearly above 0.010 mm does NOT count
    def test_11b_above_eps_does_not_count(self):
        """abs(radius_i - radius_j) > BODY_CONSENSUS_RADIUS_EPS_MM -> does NOT count.

        Construction:
            px_to_mm=50, top1: dpx=100.0 -> r=1.0, alt: dpx=101.1 -> r=1.011
            diff = 0.011 > BODY_CONSENSUS_RADIUS_EPS_MM -> not mutual support.
        """
        _PX = 50.0

        top1 = (3000.0, {"equiv_diameter_px": 100.0}, None, {"residual_rmse": 0.50}, 0)
        # radius = 100.0/(2*50) = 1.0 (exact)

        alt  = (1000.0, {"equiv_diameter_px": 101.1}, None, {"residual_rmse": 0.01}, 0)
        # radius = 101.1/(2*50) = 1.011; diff = 0.011 > 0.010

        result = _select_body_candidate([top1, alt], px_to_mm=_PX)
        # Each candidate has support = 1 -> no switch
        self.assertFalse(result["switched"],
                         "Candidates with radius diff > BODY_CONSENSUS_RADIUS_EPS_MM "
                         "must NOT share support.")
        self.assertIs(result["selected"], top1)


class TestBodySelectorConstants(unittest.TestCase):
    """Verify the frozen constant values are correct."""

    def test_constants_frozen(self):
        self.assertAlmostEqual(BODY_CONSENSUS_RADIUS_EPS_MM, 0.010, places=10)
        self.assertEqual(BODY_CONSENSUS_MIN_SUPPORT, 2)
        self.assertAlmostEqual(BODY_RMSE_RATIO_THRESHOLD, 2.0, places=10)


class TestBodySelectorRobustness(unittest.TestCase):
    """Robustness fixes: np.ndarray in tuples and single-candidate diagnostics."""

    # ------------------------------------------------------------------ #
    # Score-tie with np.ndarray — Ajuste 1
    # ------------------------------------------------------------------ #
    def test_score_tie_no_exception_first_wins(self):
        """When two candidates have identical scores, no ValueError is raised
        (old candidates.index() would fail on np.ndarray equality) and the
        FIRST candidate in the list is returned as Top-1 (historical behaviour
        of max() with a stable first-match guarantee from max over range)."""
        import numpy as np

        # Both candidates have score=1000 (exact tie)
        # global_cnt is a real np.ndarray — triggers the bug in .index() if present
        cnt = np.zeros((10, 1, 2), dtype=np.int32)
        c1 = (1000.0, {"equiv_diameter_px": 100.0}, cnt, {"residual_rmse": 0.05}, 0)
        c2 = (1000.0, {"equiv_diameter_px": 100.0}, cnt, {"residual_rmse": 0.04}, 0)

        # Must not raise any exception
        try:
            result = _select_body_candidate([c1, c2], px_to_mm=_PX_TO_MM)
        except Exception as exc:
            self.fail(f"_select_body_candidate raised unexpectedly: {exc!r}")

        # First candidate must be Top-1 (historical tie-break = first in list)
        self.assertIs(result["selected"], c1,
                      "First candidate must win on exact score tie.")
        # No switch: both have equal radius -> max support shared -> rmse_ratio
        # = 0.05/0.04 = 1.25 <= 2.0 -> no switch
        self.assertFalse(result["switched"])

    # ------------------------------------------------------------------ #
    # Single-candidate diagnostics with px_to_mm — Ajuste 2
    # ------------------------------------------------------------------ #
    def test_single_candidate_radius_diagnostic_populated(self):
        """With px_to_mm available and a single candidate, top1_radius_mm and
        selected_radius_mm must be set (not None) even though the selector is
        disabled and Top-1 is returned unchanged."""
        dpx = 100.0   # equiv_diameter_px
        cand = (1000.0, {"equiv_diameter_px": dpx}, None, {"residual_rmse": 0.05}, 0)

        result = _select_body_candidate([cand], px_to_mm=_PX_TO_MM)

        self.assertIs(result["selected"], cand)
        self.assertFalse(result["switched"])

        expected_r = dpx / (2.0 * _PX_TO_MM)   # = 1.0
        self.assertAlmostEqual(result["top1_radius_mm"], expected_r, places=10)
        self.assertAlmostEqual(result["selected_radius_mm"], expected_r, places=10)

    def test_single_candidate_radius_none_when_no_px_to_mm(self):
        """With px_to_mm=None and a single candidate, radius fields remain None."""
        cand = (1000.0, {"equiv_diameter_px": 100.0}, None, {"residual_rmse": 0.05}, 0)
        result = _select_body_candidate([cand], px_to_mm=None)

        self.assertIs(result["selected"], cand)
        self.assertIsNone(result["top1_radius_mm"])
        self.assertIsNone(result["selected_radius_mm"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

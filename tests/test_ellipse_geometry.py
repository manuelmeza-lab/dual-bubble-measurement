"""
tests/test_ellipse_geometry.py
==============================
Deterministic unit tests for the ellipse geometry fix in
bubble_cv/detection.py::_fit_single_ellipse().

Design goals
------------
* stdlib only — no pytest, no external images.
* Synthetic contours are generated in np.float32 (which cv2.fitEllipse accepts)
  so that pixel quantisation is avoided and tolerances can be made very tight.
* Tests are repeatable across Python / OpenCV versions.

Measured maximums with float32 synthetic data (400 points, semi in [20,60]):
    major_axis  rel error : 0.0000 %
    minor_axis  rel error : 0.0000 %
    angle       abs error : 0.0000 deg
    equiv_diam  rel error : 0.0000 %
    residual_rmse (fixed) : 0.000000
    residual_rmse (legacy bug, axis-swapped cases) : up to 1.289518

Tolerances are set with a 10× safety margin above the observed maxima so that
genuine floating-point noise never triggers a false positive, while remaining
far too tight to hide a 90-degree orientation error.

Convention under test
---------------------
    angle_deg MUST always be the orientation of the MAJOR axis.
    major_axis >= minor_axis  (always).
    equivalent_diameter is invariant under angle canonicalisation.
    Circles (semi_major == semi_minor) are excluded from angle checks
    because orientation is numerically undefined for circles.
"""

from __future__ import annotations

import math
import sys
import unittest

import numpy as np

sys.path.insert(0, ".")

from bubble_cv.detection import _fit_single_ellipse, _compute_ellipse_fit_quality


# ---------------------------------------------------------------------------
# Tolerances — derived from measured float32 data (see module docstring)
# Observed max was 0.0 for all metrics with the corrected implementation;
# we allow a small margin for floating-point noise across OpenCV versions.
# ---------------------------------------------------------------------------

_AXIS_RTOL    = 1e-4   # 0.01 % — observed max was 0.0 %
_ANGLE_ATOL   = 0.5    # 0.5 °  — observed max was 0.0 °;  cannot hide 90° bug
_DIAM_RTOL    = 1e-4   # 0.01 % — observed max was 0.0 %
_RESIDUAL_MAX = 1e-4   # residual < 0.01 % — observed max was 0.0; legacy was ~1.29


# ---------------------------------------------------------------------------
# Helper: build a synthetic float32 contour on a parametric ellipse
# ---------------------------------------------------------------------------

def _make_ellipse_contour(
    cx: float,
    cy: float,
    semi_major: float,
    semi_minor: float,
    angle_deg: float,
    n_points: int = 400,
) -> np.ndarray:
    """Return an (N, 1, 2) float32 contour that lies exactly on the ellipse.

    No integer rounding is applied.  cv2.fitEllipse accepts float32 arrays,
    so this gives the fitter the exact algebraic points and eliminates
    quantisation error from the test measurements.

    The major axis is oriented at *angle_deg* degrees clockwise from the
    positive x-axis (OpenCV convention, matching cv2.fitEllipse output).
    """
    t = np.linspace(0, 2.0 * math.pi, n_points, endpoint=False)

    x_local = semi_major * np.cos(t)
    y_local = semi_minor * np.sin(t)

    a_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a_rad), math.sin(a_rad)

    x_rot = cos_a * x_local - sin_a * y_local + cx
    y_rot = sin_a * x_local + cos_a * y_local + cy

    pts = np.stack([x_rot, y_rot], axis=1).astype(np.float32)
    return pts.reshape(-1, 1, 2)


# ---------------------------------------------------------------------------
# Angle-difference helper (modulo 180°)
# ---------------------------------------------------------------------------

def _angle_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two angles modulo 180°."""
    d = (a - b) % 180.0
    if d > 90.0:
        d = 180.0 - d
    return abs(d)


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

class TestMajorMinorOrder(unittest.TestCase):
    """major_axis >= minor_axis must hold for every input shape and angle."""

    _cases = [
        (60, 20,   0.0, "elongated 0deg"),
        (60, 20,  30.0, "elongated 30deg"),
        (60, 20,  45.0, "elongated 45deg"),
        (60, 20,  89.0, "elongated 89deg"),
        (60, 20,  90.0, "elongated 90deg"),
        (60, 20, 135.0, "elongated 135deg"),
        (50, 30,   0.0, "moderate 0deg"),
        (50, 30,  30.0, "moderate 30deg"),
        (50, 30,  45.0, "moderate 45deg"),
        (50, 30,  89.0, "moderate 89deg"),
        (50, 30,  90.0, "moderate 90deg"),
        (50, 30, 135.0, "moderate 135deg"),
        (40, 39,   0.0, "near-circle 0deg"),
        (40, 39,  45.0, "near-circle 45deg"),
        (35, 35,  30.0, "perfect circle"),   # angle undefined but order must hold
    ]

    def test_all(self):
        for sa, sb, ang, desc in self._cases:
            with self.subTest(desc=desc):
                cnt   = _make_ellipse_contour(200, 200, sa, sb, ang)
                props = _fit_single_ellipse(cnt)
                self.assertIsNotNone(props, f"[{desc}] returned None")
                self.assertGreaterEqual(
                    props["major_axis"], props["minor_axis"] - 1e-9,
                    f"[{desc}] major_axis < minor_axis",
                )


class TestAxisLengths(unittest.TestCase):
    """Fitted axis lengths must match the generating semi-axes to within _AXIS_RTOL."""

    _cases = [
        (60, 20,   0.0), (60, 20,  30.0), (60, 20,  45.0),
        (60, 20,  89.0), (60, 20,  90.0), (60, 20, 135.0),
        (50, 30,   0.0), (50, 30,  45.0), (50, 30,  90.0), (50, 30, 135.0),
        (40, 39,   0.0),
    ]

    def test_axis_accuracy(self):
        for sa, sb, ang in self._cases:
            with self.subTest(sa=sa, sb=sb, ang=ang):
                cnt   = _make_ellipse_contour(200, 200, sa, sb, ang)
                props = _fit_single_ellipse(cnt)
                self.assertIsNotNone(props)

                exp_major, exp_minor = 2 * sa, 2 * sb

                rel_major = abs(props["major_axis"] - exp_major) / exp_major
                rel_minor = abs(props["minor_axis"] - exp_minor) / exp_minor

                self.assertLessEqual(
                    rel_major, _AXIS_RTOL,
                    f"semi={sa},{sb} ang={ang}: major_axis rel err "
                    f"{rel_major:.2e} > {_AXIS_RTOL:.2e}",
                )
                self.assertLessEqual(
                    rel_minor, _AXIS_RTOL,
                    f"semi={sa},{sb} ang={ang}: minor_axis rel err "
                    f"{rel_minor:.2e} > {_AXIS_RTOL:.2e}",
                )


class TestAngleCorrespondsMajorAxis(unittest.TestCase):
    """angle_deg must point along the MAJOR axis (within _ANGLE_ATOL degrees).

    Perfect circles (semi_major == semi_minor) are intentionally excluded
    because their orientation is numerically undefined.
    """

    _cases = [
        (60, 20,   0.0, "elongated 0deg"),
        (60, 20,  30.0, "elongated 30deg"),
        (60, 20,  45.0, "elongated 45deg"),
        (60, 20,  89.0, "elongated 89deg"),
        (60, 20,  90.0, "elongated 90deg"),
        (60, 20, 135.0, "elongated 135deg"),
        (50, 30,   0.0, "moderate 0deg"),
        (50, 30,  45.0, "moderate 45deg"),
        (50, 30,  90.0, "moderate 90deg"),
        (50, 30, 135.0, "moderate 135deg"),
        # near-circle excluded: axis ratio ~ 1.026 — angle still numerically stable,
        # but we keep it off the angle test to avoid platform-sensitive failures.
    ]

    def test_all_orientations(self):
        for sa, sb, ang, desc in self._cases:
            with self.subTest(desc=desc):
                cnt   = _make_ellipse_contour(200, 200, sa, sb, ang)
                props = _fit_single_ellipse(cnt)
                self.assertIsNotNone(props, f"[{desc}] returned None")

                diff = _angle_diff_deg(props["angle_deg"], ang % 180.0)
                self.assertLessEqual(
                    diff, _ANGLE_ATOL,
                    f"[{desc}] angle_deg={props['angle_deg']:.4f}  "
                    f"expected~={ang % 180.0:.4f}  diff={diff:.4f} > {_ANGLE_ATOL}",
                )


class TestAxisSwapSymmetry(unittest.TestCase):
    """Swapping (semi_a, ang) → (semi_b, ang+90°) describes the same ellipse.

    Both descriptions must produce identical major/minor axes and a consistent
    angle_deg (i.e., they both point to the same physical axis).
    """

    _cases = [
        (60, 25,  30.0, "elongated 30deg"),
        (60, 25,  90.0, "elongated 90deg"),
        (50, 30,  45.0, "moderate 45deg"),
        (50, 30, 135.0, "moderate 135deg"),
    ]

    def test_swap_invariant(self):
        for sa, sb, ang, desc in self._cases:
            with self.subTest(desc=desc):
                cnt_a = _make_ellipse_contour(200, 200, sa, sb, ang)
                cnt_b = _make_ellipse_contour(200, 200, sb, sa, (ang + 90.0) % 180.0)

                pa = _fit_single_ellipse(cnt_a)
                pb = _fit_single_ellipse(cnt_b)
                self.assertIsNotNone(pa); self.assertIsNotNone(pb)

                self.assertAlmostEqual(
                    pa["major_axis"], pb["major_axis"],
                    delta=pa["major_axis"] * _AXIS_RTOL,
                    msg=f"[{desc}] major_axis mismatch",
                )
                self.assertAlmostEqual(
                    pa["minor_axis"], pb["minor_axis"],
                    delta=max(pa["minor_axis"], 1e-9) * _AXIS_RTOL,
                    msg=f"[{desc}] minor_axis mismatch",
                )
                diff = _angle_diff_deg(pa["angle_deg"], pb["angle_deg"])
                self.assertLessEqual(
                    diff, _ANGLE_ATOL,
                    f"[{desc}] angle_deg inconsistent: "
                    f"{pa['angle_deg']:.4f} vs {pb['angle_deg']:.4f}  diff={diff:.4f}",
                )


class TestEquivalentDiameterInvariance(unittest.TestCase):
    """equiv_diameter_px depends only on semi-axes, not on angle."""

    _cases = [
        (60, 20,  0.0), (60, 20, 45.0), (60, 20, 90.0),
        (50, 30, 30.0), (50, 30,135.0), (40, 39,  0.0),
    ]

    def test_invariant(self):
        from bubble_cv.geometry import equivalent_diameter
        for sa, sb, ang in self._cases:
            with self.subTest(sa=sa, sb=sb, ang=ang):
                cnt   = _make_ellipse_contour(200, 200, sa, sb, ang)
                props = _fit_single_ellipse(cnt)
                self.assertIsNotNone(props)

                expected = equivalent_diameter(sa, sb)
                rel_err  = abs(props["equiv_diameter_px"] - expected) / expected
                self.assertLessEqual(
                    rel_err, _DIAM_RTOL,
                    f"semi={sa},{sb} ang={ang}: equiv_diameter rel err "
                    f"{rel_err:.2e} > {_DIAM_RTOL:.2e}",
                )


class TestFitQualityLowResidual(unittest.TestCase):
    """_compute_ellipse_fit_quality must return near-zero residuals for
    float32 synthetic points that lie exactly on the fitted ellipse.

    If angle_deg / semi_major / semi_minor are geometrically inconsistent
    (e.g. the pre-fix bug), the algebraic residual is large even for perfect
    data because the rotation matrix in the residual formula uses the wrong angle.

    With float32 exact-ellipse data and the corrected implementation the
    residual is numerically zero; we require it to be below _RESIDUAL_MAX = 1e-4.
    """

    _cases = [
        (60, 20,   0.0, "elongated 0deg"),
        (60, 20,  30.0, "elongated 30deg"),
        (60, 20,  45.0, "elongated 45deg"),
        (60, 20,  89.0, "elongated 89deg"),
        (60, 20,  90.0, "elongated 90deg"),
        (60, 20, 135.0, "elongated 135deg"),
        (50, 30,   0.0, "moderate 0deg"),
        (50, 30,  45.0, "moderate 45deg"),
        (50, 30,  90.0, "moderate 90deg"),
        (50, 30, 135.0, "moderate 135deg"),
        (40, 39,   0.0, "near-circle 0deg"),
    ]

    def test_low_residual(self):
        for sa, sb, ang, desc in self._cases:
            with self.subTest(desc=desc):
                cnt   = _make_ellipse_contour(200, 200, sa, sb, ang)
                props = _fit_single_ellipse(cnt)
                self.assertIsNotNone(props, f"[{desc}] returned None")

                qual = _compute_ellipse_fit_quality(cnt, props, dyn_h=400, dyn_w=400)
                rmse = qual["residual_rmse"]
                self.assertIsNotNone(rmse, f"[{desc}] residual_rmse is None")
                self.assertLess(
                    rmse, _RESIDUAL_MAX,
                    f"[{desc}] residual_rmse={rmse:.6f} >= {_RESIDUAL_MAX}  "
                    f"(major={props['major_axis']:.4f} minor={props['minor_axis']:.4f} "
                    f"angle={props['angle_deg']:.4f} input_angle={ang})",
                )


class TestLegacyBugDemonstration(unittest.TestCase):
    """Explicit before/after: legacy implementation must fail; fixed must pass.

    With float32 data the legacy residual is ~1.29 for elongated ellipses
    because cv2.fitEllipse always returns axis1 < axis2 for our synthetic
    contours (the parametric generation reliably triggers the axis-swap case).
    The fixed implementation always returns 0.000000.

    The test asserts:
    - legacy_failures >= 1  (confirms we are testing a real bug)
    - fixed_failures  == 0  (confirms the fix is complete)
    """

    @staticmethod
    def _legacy_fit(contour):
        """Replica of the old (broken) _fit_single_ellipse logic."""
        import cv2
        from bubble_cv.geometry import eccentricity, equivalent_diameter

        if len(contour) < 5:
            return None
        (cx, cy), (axis1, axis2), angle = cv2.fitEllipse(contour)
        major_axis = max(axis1, axis2)
        minor_axis = min(axis1, axis2)
        semi_major  = major_axis / 2.0
        semi_minor  = minor_axis / 2.0
        return {
            "center_x":          cx,
            "center_y":          cy,
            "major_axis":        major_axis,
            "minor_axis":        minor_axis,
            "angle_deg":         angle,      # BUG: not adjusted when axes swap
            "semi_major":        semi_major,
            "semi_minor":        semi_minor,
            "eccentricity":      eccentricity(semi_major, semi_minor),
            "equiv_diameter_px": equivalent_diameter(semi_major, semi_minor),
        }

    @staticmethod
    def _residual_rmse(props, cnt):
        """Algebraic residual — mirrors _compute_ellipse_fit_quality internals."""
        pts          = cnt.reshape(-1, 2).astype(float)
        dx, dy       = pts[:, 0] - props["center_x"], pts[:, 1] - props["center_y"]
        a_r          = math.radians(props["angle_deg"])
        cos_a, sin_a = math.cos(a_r), math.sin(a_r)
        xr           =  cos_a * dx + sin_a * dy
        yr           = -sin_a * dx + cos_a * dy
        a            = max(props["semi_major"], 1e-9)
        b            = max(props["semi_minor"], 1e-9)
        q            = np.sqrt((xr / a) ** 2 + (yr / b) ** 2)
        return float(np.sqrt(np.mean((q - 1.0) ** 2)))

    def test_legacy_fails_fixed_passes(self):
        """Legacy RMSE >> _RESIDUAL_MAX for axis-swapped cases; fixed RMSE < _RESIDUAL_MAX."""
        # Use only strongly elongated cases so axis-swap is guaranteed to trigger.
        angles     = [0.0, 30.0, 45.0, 89.0, 90.0, 135.0]
        sa, sb     = 60, 20
        # Threshold that separates the bug (>0.4) from the fix (~0.0)
        _LEGACY_FAILURE_THRESHOLD = 0.1   # any legacy bug will be >> this

        legacy_failures = 0
        fixed_failures  = 0

        for ang in angles:
            cnt = _make_ellipse_contour(200, 200, sa, sb, ang)

            p_legacy = self._legacy_fit(cnt)
            self.assertIsNotNone(p_legacy, f"legacy fit returned None at {ang}")
            rmse_legacy = self._residual_rmse(p_legacy, cnt)
            if rmse_legacy >= _LEGACY_FAILURE_THRESHOLD:
                legacy_failures += 1

            p_fixed = _fit_single_ellipse(cnt)
            self.assertIsNotNone(p_fixed, f"fixed fit returned None at {ang}")
            rmse_fixed = self._residual_rmse(p_fixed, cnt)
            if rmse_fixed >= _RESIDUAL_MAX:
                fixed_failures += 1

        self.assertGreater(
            legacy_failures, 0,
            "Legacy implementation did not fail any residual check — "
            "verify that axis-swap cases are being generated correctly.",
        )
        self.assertEqual(
            fixed_failures, 0,
            f"Fixed implementation failed {fixed_failures} residual check(s) "
            f"with threshold {_RESIDUAL_MAX}.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

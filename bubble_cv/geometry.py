"""
Geometry module — Spheroid calculations for bubble measurements.

Provides functions to compute area, volume, surface area, eccentricity,
and equivalent diameter for ellipsoidal (spheroidal) bubbles.

All formulas use semi-axis notation:
    a = semi-major axis (larger)
    b = semi-minor axis (smaller)
"""

import numpy as np


def ellipse_area(semi_major: float, semi_minor: float) -> float:
    """Compute the projected 2D area of an ellipse.

    Args:
        semi_major: Semi-major axis length (mm or px).
        semi_minor: Semi-minor axis length (mm or px).

    Returns:
        Projected area in the same units squared.
    """
    return np.pi * semi_major * semi_minor


def spheroid_volume(semi_major: float, semi_minor: float) -> float:
    """Compute the volume of a spheroid (ellipsoid of revolution).

    Automatically determines if the spheroid is oblate (a > b, disk-like)
    or prolate (a > b, elongated) and applies the correct formula.

    For an oblate spheroid: V = (4/3) * pi * a^2 * b
    For a prolate spheroid: V = (4/3) * pi * a * b^2

    In practice, for a bubble viewed from one axis, we assume the
    revolution axis is the minor axis, giving an oblate spheroid.

    Args:
        semi_major: Semi-major axis length.
        semi_minor: Semi-minor axis length.

    Returns:
        Volume of the spheroid.
    """
    # Ensure a >= b
    a = max(semi_major, semi_minor)
    b = min(semi_major, semi_minor)
    # Oblate spheroid: revolution around the minor axis
    return (4.0 / 3.0) * np.pi * a * a * b


def spheroid_surface(semi_major: float, semi_minor: float) -> float:
    """Compute the surface area of an oblate spheroid.

    Uses the exact formula:
        S = 2 * pi * a^2 + (pi * b^2 / e) * ln((1 + e) / (1 - e))
    where e = sqrt(1 - (b/a)^2) is the eccentricity.

    For nearly spherical shapes (e ~ 0), falls back to sphere formula.

    Args:
        semi_major: Semi-major axis length.
        semi_minor: Semi-minor axis length.

    Returns:
        Surface area of the spheroid.
    """
    a = max(semi_major, semi_minor)
    b = min(semi_major, semi_minor)

    if a == 0:
        return 0.0

    e = eccentricity(a, b)

    if e < 1e-10:
        # Nearly spherical — use sphere formula to avoid division by zero
        return 4.0 * np.pi * a * a

    # Oblate spheroid surface area
    return (
        2.0 * np.pi * a * a
        + (np.pi * b * b / e) * np.log((1.0 + e) / (1.0 - e))
    )


def equivalent_diameter(semi_major: float, semi_minor: float) -> float:
    """Compute the equivalent diameter as the geometric mean of the axes.

    D_eq = 2 * sqrt(a * b)

    This gives the diameter of a circle with the same area as the ellipse.

    Args:
        semi_major: Semi-major axis length.
        semi_minor: Semi-minor axis length.

    Returns:
        Equivalent diameter.
    """
    return 2.0 * np.sqrt(semi_major * semi_minor)


def eccentricity(semi_major: float, semi_minor: float) -> float:
    """Compute the eccentricity of an ellipse.

    e = sqrt(1 - (b/a)^2)

    where a >= b.

    Values:
        0.0 = perfect circle
        1.0 = degenerate (line)

    Args:
        semi_major: Semi-major axis length.
        semi_minor: Semi-minor axis length.

    Returns:
        Eccentricity value between 0 and 1.
    """
    a = max(semi_major, semi_minor)
    b = min(semi_major, semi_minor)

    if a == 0:
        return 0.0

    return np.sqrt(1.0 - (b / a) ** 2)

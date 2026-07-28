"""
BubbleCV — Bubble Evaporation Analysis System

A computer vision toolkit for measuring bubble geometry from
DinoLite microscope images and videos under low-illumination conditions.

Modules:
    preprocessing : CLAHE contrast enhancement and channel selection
    detection     : Two-stage bubble detection (Hough + Ellipse)
    geometry      : Spheroid volume, surface area, and eccentricity
    calibration   : Spatial calibration (px to mm)
    visualization : Detection overlay and time-series plots
    io_utils      : CSV export and video frame iteration
"""

__version__ = "1.0.0"
__author__ = "Manuel"

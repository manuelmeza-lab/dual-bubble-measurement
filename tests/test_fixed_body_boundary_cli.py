"""
CLI contract for optional fixed physical BODY boundaries.

The public command-line coordinates are global frame Y pixels.
Both sides are independent and default to automatic RC3 behaviour.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import analyze_video


class _StopAfterDetection(Exception):
    """Sentinel used to stop main immediately after detector wiring."""


class TestFixedBodyBoundaryCliParsing(unittest.TestCase):

    def test_defaults_are_none(self):

        argv = [
            "analyze_video.py",
            "--input",
            "dummy.mp4",
        ]

        with patch.object(
            sys,
            "argv",
            argv,
        ):
            args = analyze_video.parse_args()

        self.assertIsNone(
            args.control_body_start_y
        )

        self.assertIsNone(
            args.sample_body_start_y
        )


    def test_independent_cli_values_are_parsed(self):

        argv = [
            "analyze_video.py",
            "--input",
            "dummy.mp4",
            "--control-body-start-y",
            "61",
            "--sample-body-start-y",
            "54",
        ]

        with patch.object(
            sys,
            "argv",
            argv,
        ):
            args = analyze_video.parse_args()

        self.assertEqual(
            args.control_body_start_y,
            61,
        )

        self.assertEqual(
            args.sample_body_start_y,
            54,
        )


class TestFixedBodyBoundaryCliWiring(unittest.TestCase):

    def test_main_forwards_boundaries_to_detector(self):

        frame = np.zeros(
            (120, 160, 3),
            dtype=np.uint8,
        )

        with tempfile.TemporaryDirectory() as tmp:

            video_path = (
                Path(tmp)
                / "dummy.mp4"
            )

            # main() only requires that the path exists before
            # frame_iterator is called. No real video is needed.
            video_path.touch()

            argv = [
                "analyze_video.py",
                "--input",
                str(video_path),
                "--calibration",
                "34.08",
                "--control-body-start-y",
                "61",
                "--sample-body-start-y",
                "54",
            ]

            with (
                patch.object(
                    sys,
                    "argv",
                    argv,
                ),
                patch.object(
                    analyze_video,
                    "frame_iterator",
                    return_value=iter(
                        [
                            (
                                0,
                                frame,
                            )
                        ]
                    ),
                ),
                patch.object(
                    analyze_video,
                    "detect_bubbles",
                    side_effect=
                        _StopAfterDetection,
                ) as detector,
            ):

                with self.assertRaises(
                    _StopAfterDetection
                ):
                    analyze_video.main()

        detector.assert_called_once()

        kwargs = (
            detector.call_args.kwargs
        )

        self.assertEqual(
            kwargs[
                "control_body_start_y_global"
            ],
            61,
        )

        self.assertEqual(
            kwargs[
                "sample_body_start_y_global"
            ],
            54,
        )


if __name__ == "__main__":
    unittest.main()

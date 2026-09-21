import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import run_bubblecv


class TestGuidedRunner(unittest.TestCase):

    def test_parse_fraction(self):
        self.assertEqual(
            run_bubblecv.parse_fraction("10/1"),
            10.0,
        )
        self.assertAlmostEqual(
            run_bubblecv.parse_fraction("40/3"),
            13.333333333333334,
        )
        self.assertEqual(
            run_bubblecv.parse_fraction("30"),
            30.0,
        )
        self.assertIsNone(
            run_bubblecv.parse_fraction("1/0")
        )
        self.assertIsNone(
            run_bubblecv.parse_fraction("abc")
        )
        self.assertIsNone(
            run_bubblecv.parse_fraction(None)
        )

    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.bin"
            path.write_bytes(b"BubbleCV")

            digest = run_bubblecv.sha256_file(path)

            self.assertEqual(
                digest,
                "f7020f4bbcf44c75d76dc53575bfb18c5e35a75f5b63ff8b2d48c95903b0e118",
            )

    def test_video5_preview_builds_expected_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            video = tmp / "hexa_v5_student.mp4"
            video.touch()

            output_dir = tmp / "results_video5"

            answers = [
                str(video),
                "34.08",
                "10",
                "10",
                "1",
                "2",
                "56",
                "",
                "",
                "",
                "",
                str(output_dir),
                "n",
            ]

            metadata = {
                "codec_name": "h264",
                "width": "640",
                "height": "480",
                "r_frame_rate": "10/1",
                "avg_frame_rate": "10/1",
                "duration": "879.200000",
            }

            buffer = io.StringIO()

            with patch(
                "run_bubblecv.input",
                side_effect=answers,
            ), patch(
                "run_bubblecv.probe_video",
                return_value=metadata,
            ), redirect_stdout(buffer):

                status = run_bubblecv.main()

            text = buffer.getvalue()

            self.assertEqual(status, 0)

            self.assertIn(
                "--calibration 34.08",
                text,
            )
            self.assertIn(
                "--fps 10.0",
                text,
            )
            self.assertIn(
                "--skip 10",
                text,
            )
            self.assertIn(
                "--sample-body-start-y 56",
                text,
            )
            self.assertNotIn(
                "--control-body-start-y",
                text,
            )
            self.assertFalse(
                output_dir.exists()
            )

    def test_confirmed_execution_creates_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            video = tmp / "video.mp4"
            video.write_bytes(b"fake-video")

            output_dir = tmp / "run_output"

            answers = [
                str(video),
                "34.08",
                "10",
                "10",
                "1",
                "2",
                "56",
                "",
                "",
                "",
                "",
                str(output_dir),
                "s",
            ]

            metadata = {
                "codec_name": "h264",
                "width": "640",
                "height": "480",
                "r_frame_rate": "10/1",
                "avg_frame_rate": "10/1",
                "duration": "100.0",
            }

            buffer = io.StringIO()

            with patch(
                "run_bubblecv.input",
                side_effect=answers,
            ), patch(
                "run_bubblecv.probe_video",
                return_value=metadata,
            ), patch(
                "run_bubblecv.environment_info",
                return_value={
                    "git_commit": "abc123",
                    "python": "test",
                },
            ), patch(
                "run_bubblecv.execute_analysis",
                return_value=0,
            ), redirect_stdout(buffer):

                status = run_bubblecv.main()

            self.assertEqual(status, 0)
            self.assertTrue(output_dir.is_dir())

            manifest_path = (
                output_dir / "run_manifest.json"
            )

            self.assertTrue(
                manifest_path.is_file()
            )

            manifest = json.loads(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                manifest["status"],
                "completed",
            )
            self.assertEqual(
                manifest["exit_code"],
                0,
            )
            self.assertEqual(
                manifest["parameters"][
                    "calibration_px_per_mm"
                ],
                34.08,
            )
            self.assertEqual(
                manifest["parameters"][
                    "sample_body_start_y_global"
                ],
                56,
            )
            self.assertIsNone(
                manifest["parameters"][
                    "control_body_start_y_global"
                ]
            )

    def test_existing_output_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            video = tmp / "video.mp4"
            video.touch()

            output_dir = tmp / "existing"
            output_dir.mkdir()

            answers = [
                str(video),
                "34.08",
                "10",
                "10",
                "1",
                "1",
                "",
                "",
                "",
                "",
                str(output_dir),
                "s",
            ]

            metadata = {
                "avg_frame_rate": "10/1",
            }

            buffer = io.StringIO()

            with patch(
                "run_bubblecv.input",
                side_effect=answers,
            ), patch(
                "run_bubblecv.probe_video",
                return_value=metadata,
            ), redirect_stdout(buffer):

                status = run_bubblecv.main()

            self.assertEqual(status, 3)
            self.assertIn(
                "la carpeta de resultados ya existe",
                buffer.getvalue(),
            )

    def test_missing_video_returns_error(self):
        missing = (
            Path(tempfile.gettempdir())
            / "bubblecv_file_that_does_not_exist.mp4"
        )

        buffer = io.StringIO()

        with patch(
            "run_bubblecv.input",
            side_effect=[str(missing)],
        ), redirect_stdout(buffer):

            status = run_bubblecv.main()

        self.assertEqual(status, 2)
        self.assertIn(
            "ERROR: no existe el archivo",
            buffer.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()

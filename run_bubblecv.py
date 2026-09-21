#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import math
import platform
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib
import numpy
import pandas


def ask_text(prompt: str, default: str | None = None) -> str:
    while True:
        if default is None:
            raw = input(f"{prompt}: ").strip()
        else:
            raw = input(f"{prompt} [{default}]: ").strip()

        if raw:
            return raw

        if default is not None:
            return default

        print("Se requiere un valor.")


def ask_float(
    prompt: str,
    default: float | None = None,
    minimum: float | None = None,
) -> float:
    while True:
        default_text = None if default is None else str(default)
        raw = ask_text(prompt, default_text)

        try:
            value = float(raw)
        except ValueError:
            print("Introduce un número válido.")
            continue

        if not math.isfinite(value):
            print("El valor debe ser finito.")
            continue

        if minimum is not None and value < minimum:
            print(f"El valor debe ser >= {minimum}.")
            continue

        return value


def ask_int(
    prompt: str,
    default: int | None = None,
    minimum: int | None = None,
) -> int:
    while True:
        default_text = None if default is None else str(default)
        raw = ask_text(prompt, default_text)

        try:
            value = int(raw)
        except ValueError:
            print("Introduce un número entero válido.")
            continue

        if minimum is not None and value < minimum:
            print(f"El valor debe ser >= {minimum}.")
            continue

        return value


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    marker = "S/n" if default else "s/N"

    while True:
        raw = input(f"{prompt} [{marker}]: ").strip().lower()

        if not raw:
            return default

        if raw in {"s", "si", "sí", "y", "yes"}:
            return True

        if raw in {"n", "no"}:
            return False

        print("Responde s o n.")


def ask_body(side: str) -> int | None:
    print()
    print(f"BODY — {side.upper()}")
    print("  1 = automático")
    print("  2 = frontera física fija")

    while True:
        choice = input("Selecciona 1 o 2 [1]: ").strip()

        if not choice:
            return None

        if choice == "1":
            return None

        if choice == "2":
            return ask_int(
                f"Coordenada Y global fija para {side.upper()} (px)",
                minimum=0,
            )

        print("Selecciona 1 o 2.")


def probe_video(path: Path) -> dict[str, str]:
    ffprobe = shutil.which("ffprobe")

    if ffprobe is None:
        return {}

    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-of",
        "default=noprint_wrappers=1",
        str(path),
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return {}

    data = {}

    for line in result.stdout.splitlines():
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()

    return data


def parse_fraction(value: str | None) -> float | None:
    if not value:
        return None

    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            denominator_value = float(denominator)

            if denominator_value == 0:
                return None

            return float(numerator) / denominator_value

        return float(value)
    except ValueError:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def git_value(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    value = result.stdout.strip()
    return value or None


def environment_info(root: Path) -> dict:
    return {
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "opencv": cv2.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "matplotlib": matplotlib.__version__,
        "git_commit": git_value(root, "rev-parse", "HEAD"),
        "git_branch": git_value(root, "branch", "--show-current"),
        "git_tag_exact": git_value(
            root,
            "describe",
            "--tags",
            "--exact-match",
            "HEAD",
        ),
    }


def write_manifest(
    path: Path,
    data: dict,
) -> None:
    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def execute_analysis(command: list[str]) -> int:
    result = subprocess.run(command)
    return int(result.returncode)


def main() -> int:
    root = Path(__file__).resolve().parent
    analyzer = root / "analyze_video.py"

    print()
    print("=" * 72)
    print("BubbleCV Dual — Guided Runner")
    print("v0.2.1 development")
    print("=" * 72)
    print()
    print("Convención fija:")
    print("  CONTROL = gota izquierda")
    print("  SAMPLE  = gota derecha")
    print()

    raw_video = ask_text("Ruta del video")
    video = Path(raw_video).expanduser().resolve()

    if not video.is_file():
        print()
        print("ERROR: no existe el archivo:")
        print(video)
        return 2

    metadata = probe_video(video)
    detected_fps = parse_fraction(
        metadata.get("avg_frame_rate")
    )

    print()
    print("===== VIDEO =====")
    print(f"Archivo       : {video}")

    if metadata:
        print(f"Codec         : {metadata.get('codec_name', 'N/D')}")
        print(
            f"Resolución    : "
            f"{metadata.get('width', 'N/D')}x"
            f"{metadata.get('height', 'N/D')}"
        )
        print(
            f"r_frame_rate  : "
            f"{metadata.get('r_frame_rate', 'N/D')}"
        )
        print(
            f"avg_frame_rate: "
            f"{metadata.get('avg_frame_rate', 'N/D')}"
        )
        print(
            f"Duración      : "
            f"{metadata.get('duration', 'N/D')} s"
        )
    else:
        print("ffprobe no disponible o no pudo leer metadatos.")

    print()
    calibration = ask_float(
        "Calibración (px/mm)",
        minimum=0.000001,
    )

    if detected_fps is not None and detected_fps > 0:
        fps_default = detected_fps
    else:
        fps_default = 30.0

    fps = ask_float(
        "FPS que usará BubbleCV",
        default=fps_default,
        minimum=0.000001,
    )

    skip_default = max(1, int(round(fps)))

    skip = ask_int(
        "Procesar 1 de cada N frames (--skip)",
        default=skip_default,
        minimum=1,
    )

    control_body = ask_body("control")
    sample_body = ask_body("sample")

    print()
    print("===== PARÁMETROS DE ANÁLISIS =====")

    clip_limit = ask_float(
        "CLAHE clip-limit",
        default=3.0,
        minimum=0.000001,
    )

    max_eccentricity = ask_float(
        "Excentricidad máxima",
        default=0.85,
        minimum=0.0,
    )

    smooth = ask_int(
        "Ventana temporal de suavizado (0 = desactivado)",
        default=0,
        minimum=0,
    )

    bin_size = ask_float(
        "Tamaño de bin (s)",
        default=10.0,
        minimum=0.000001,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = (
        root
        / "results"
        / f"{video.stem}_{timestamp}"
    )

    output_dir = Path(
        ask_text(
            "Carpeta de resultados",
            str(default_output),
        )
    ).expanduser().resolve()

    command = [
        sys.executable,
        str(analyzer),
        "--input",
        str(video),
        "--output",
        str(output_dir / "results.csv"),
        "--summary-output",
        str(output_dir / "summary.csv"),
        "--binned-output",
        str(output_dir / "binned.csv"),
        "--calibration",
        str(calibration),
        "--fps",
        str(fps),
        "--skip",
        str(skip),
        "--clip-limit",
        str(clip_limit),
        "--max-eccentricity",
        str(max_eccentricity),
        "--smooth",
        str(smooth),
        "--r2-fit",
        "--bin-size-s",
        str(bin_size),
    ]

    if control_body is not None:
        command.extend(
            [
                "--control-body-start-y",
                str(control_body),
            ]
        )

    if sample_body is not None:
        command.extend(
            [
                "--sample-body-start-y",
                str(sample_body),
            ]
        )

    print()
    print("=" * 72)
    print("REVISIÓN ANTES DE EJECUTAR")
    print("=" * 72)
    print()
    print(f"Video              : {video}")
    print(f"Calibración        : {calibration} px/mm")
    print(f"FPS                : {fps}")
    print(f"Skip               : {skip}")
    print(
        "CONTROL BODY       : "
        + (
            "automático"
            if control_body is None
            else f"fijo y={control_body}px"
        )
    )
    print(
        "SAMPLE BODY        : "
        + (
            "automático"
            if sample_body is None
            else f"fijo y={sample_body}px"
        )
    )
    print(f"Clip limit         : {clip_limit}")
    print(f"Max eccentricity   : {max_eccentricity}")
    print(f"Smooth             : {smooth}")
    print(f"Bin                 : {bin_size} s")
    print(f"Resultados         : {output_dir}")

    print()
    print("===== COMANDO =====")
    print()
    print(shlex.join(command))

    print()

    if not ask_yes_no(
        "¿Ejecutar el análisis con estos parámetros?",
        default=False,
    ):
        print()
        print("=" * 72)
        print("PREVIEW COMPLETADO")
        print("=" * 72)
        print()
        print("No se ejecutó ningún análisis.")
        print("No se creó ninguna carpeta de resultados.")
        return 0

    if output_dir.exists():
        print()
        print("ERROR: la carpeta de resultados ya existe:")
        print(output_dir)
        print()
        print("Elige una carpeta nueva para evitar sobrescrituras.")
        return 3

    print()
    print("Calculando SHA-256 del video...")

    video_sha256 = sha256_file(video)

    output_dir.mkdir(parents=True)

    started_at = datetime.now(timezone.utc).isoformat()

    manifest = {
        "bubblecv_guided_runner": "v0.2.1-development",
        "status": "running",
        "started_at_utc": started_at,
        "input": {
            "path": str(video),
            "sha256": video_sha256,
            "ffprobe": metadata,
        },
        "parameters": {
            "calibration_px_per_mm": calibration,
            "fps": fps,
            "skip": skip,
            "control_body_start_y_global": control_body,
            "sample_body_start_y_global": sample_body,
            "clip_limit": clip_limit,
            "max_eccentricity": max_eccentricity,
            "smooth": smooth,
            "r2_fit": True,
            "bin_size_s": bin_size,
        },
        "outputs": {
            "directory": str(output_dir),
            "results_csv": str(output_dir / "results.csv"),
            "summary_csv": str(output_dir / "summary.csv"),
            "binned_csv": str(output_dir / "binned.csv"),
        },
        "environment": environment_info(root),
        "command": command,
        "exit_code": None,
    }

    manifest_path = output_dir / "run_manifest.json"
    write_manifest(
        manifest_path,
        manifest,
    )

    print()
    print("=" * 72)
    print("EJECUTANDO ANÁLISIS")
    print("=" * 72)
    print()

    exit_code = execute_analysis(command)

    manifest["exit_code"] = exit_code
    manifest["finished_at_utc"] = (
        datetime.now(timezone.utc).isoformat()
    )
    manifest["status"] = (
        "completed"
        if exit_code == 0
        else "failed"
    )

    write_manifest(
        manifest_path,
        manifest,
    )

    print()
    print("=" * 72)
    print("RESULTADO")
    print("=" * 72)
    print()
    print(f"exit_code          : {exit_code}")
    print(f"SHA-256            : {video_sha256}")
    print(f"Manifest           : {manifest_path}")
    print(f"Resultados         : {output_dir}")

    summary = output_dir / "summary.csv"

    if summary.is_file():
        print()
        print("===== SUMMARY =====")
        print(summary.read_text(encoding="utf-8"))

    if exit_code == 0:
        print("Análisis completado correctamente.")
    else:
        print("El análisis terminó con errores.")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

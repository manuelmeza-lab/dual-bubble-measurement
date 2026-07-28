# Guía de Análisis

Esta guía cubre el flujo de trabajo completo para analizar imágenes y videos
de gotas usando BubbleCV.

---

## Flujo de trabajo recomendado

```
1. Capturar → 2. Convertir video → 3. Calibrar → 4. Analizar → 5. Interpretar CSV
```

---

## Paso 1 — Preparar los archivos

Organiza tus datos en una carpeta fuera del repositorio:

```
mi_experimento/
├── calibration.png       ← imagen del objeto de referencia
├── video.mp4             ← video del experimento (convertido)
└── results/              ← aquí se guardarán los CSV y gráficas
```

> ⚠️ **Nunca pongas tus videos o datos dentro de la carpeta del repositorio.**
> Están excluidos por `.gitignore`, pero mantener los datos separados del
> código es una buena práctica.

---

## Paso 2 — Activar el entorno virtual

Antes de cada sesión de análisis:

```bash
# macOS/Linux
source venv/bin/activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

---

## Paso 3 — Calibrar

Obtén el valor px/mm a partir de una imagen de referencia.

Consulta → [`docs/calibration.md`](./calibration.md) para el procedimiento completo.

```bash
# Verificar calibración automática
python analyze_image.py \
    --input path/to/calibration.png \
    --calibrate-from path/to/calibration.png \
    --ref-diameter 4.0 \
    --verbose
```

Anota el valor px/mm reportado (ejemplo: `114.0`).

---

## Paso 4 — Analizar imágenes (opcional)

Si tienes imágenes individuales en lugar de video:

```bash
# Una sola imagen
python analyze_image.py \
    --input path/to/drop.png \
    --calibration 114.0 \
    --output results/output.csv \
    --visualize \
    --vis-dir results/annotated

# Directorio completo de imágenes
python analyze_image.py \
    --input path/to/images/ \
    --calibration 114.0 \
    --output results/output.csv
```

---

## Paso 5 — Analizar video

### Análisis básico

```bash
python analyze_video.py \
    --input path/to/video.mp4 \
    --calibration 114.0 \
    --fps 30 \
    --output results/results_video.csv
```

### Con salto de frames (videos largos)

```bash
# Procesar 1 frame por segundo (en video a 30 FPS)
python analyze_video.py \
    --input path/to/video.mp4 \
    --calibration 114.0 \
    --fps 30 \
    --skip 30 \
    --output results/results_video.csv
```

### Con suavizado y gráficas

```bash
python analyze_video.py \
    --input path/to/video.mp4 \
    --calibration 114.0 \
    --fps 30 \
    --skip 30 \
    --smooth 5 \
    --plot \
    --output results/results_video.csv
```

### Con frames anotados (diagnóstico visual)

```bash
python analyze_video.py \
    --input path/to/video.mp4 \
    --calibration 114.0 \
    --fps 30 \
    --skip 30 \
    --visualize \
    --vis-dir results/annotated_frames
```

---

## Referencia de parámetros

### `analyze_image.py`

| Bandera | Descripción | Por defecto |
|---------|-------------|-------------|
| `--input`, `-i` | Imagen o directorio | (requerido) |
| `--output`, `-o` | Archivo CSV de salida | `results_images.csv` |
| `--calibration`, `-c` | Relación px/mm | Sin calibrar |
| `--calibrate-from` | Imagen para auto-calibración | — |
| `--ref-diameter` | Diámetro de referencia (mm) | `4.0` |
| `--min-radius` | Radio mínimo (px) | `50` |
| `--max-radius` | Radio máximo (px) | `500` |
| `--clip-limit` | Límite CLAHE | `3.0` |
| `--visualize`, `-v` | Guardar imágenes anotadas | No |
| `--vis-dir` | Carpeta de imágenes anotadas | `annotated/` |
| `--verbose` | Log detallado | No |

### `analyze_video.py`

| Bandera | Descripción | Por defecto |
|---------|-------------|-------------|
| `--input`, `-i` | Archivo de video | (requerido) |
| `--output`, `-o` | Archivo CSV de salida | `results_video.csv` |
| `--fps` | Fotogramas por segundo | `30.0` |
| `--skip`, `-s` | Analizar cada N frames | `1` |
| `--calibration`, `-c` | Relación px/mm | Sin calibrar |
| `--calibrate-from` | Imagen para auto-calibración | — |
| `--ref-diameter` | Diámetro de referencia (mm) | `4.0` |
| `--min-radius` | Radio mínimo (px) | `50` |
| `--max-radius` | Radio máximo (px) | `500` |
| `--clip-limit` | Límite CLAHE | `3.0` |
| `--smooth` | Ventana de suavizado temporal | `0` (sin suavizado) |
| `--visualize`, `-v` | Guardar frames anotados | No |
| `--vis-dir` | Carpeta de frames anotados | `annotated_frames/` |
| `--plot`, `-p` | Generar gráficas | No |
| `--verbose` | Log detallado | No |

---

## Paso 6 — Interpretar el CSV

### Columnas del CSV (imágenes y video)

| Columna | Unidad | Descripción |
|---------|--------|-------------|
| `filename` | — | Nombre del archivo o número de frame |
| `frame_id` | — | Número de frame (solo video) |
| `timestamp_s` | s | Tiempo en segundos (solo video) |
| `center_x_px` | px | Coordenada X del centro |
| `center_y_px` | px | Coordenada Y del centro |
| `major_axis_px` | px | Eje mayor de la elipse |
| `minor_axis_px` | px | Eje menor de la elipse |
| `angle_deg` | ° | Ángulo de orientación de la elipse |
| `equiv_diameter_px` | px | Diámetro equivalente (√(a·b)) |
| `eccentricity` | — | 0 = círculo perfecto, 1 = línea |
| `major_axis_mm`* | mm | Eje mayor en milímetros |
| `minor_axis_mm`* | mm | Eje menor en milímetros |
| `equiv_diameter_mm`* | mm | Diámetro equivalente en mm |
| `area_mm2`* | mm² | Área proyectada |
| `surface_mm2`* | mm² | Superficie del esferoide |
| `volume_mm3`* | mm³ | Volumen del esferoide |
| `evap_rate_mm3_s`* | mm³/s | Tasa de evaporación instantánea (solo video) |
| `confidence` | 0–1 | Confianza de la detección |
| `method` | — | `hough+ellipse` o `hough_only` |

*Requiere calibración (`--calibration` o `--calibrate-from`). Sin calibración, estas columnas quedan vacías.

### Señales de una buena detección

- `confidence` ≥ 0.7
- `method` = `hough+ellipse` (pipeline completo)
- `eccentricity` < 0.5 para gotas aproximadamente esféricas
- Valores de `major_axis_mm` y `minor_axis_mm` consistentes entre frames

### Señales de problema

- `confidence` < 0.5 → revisar parámetros de detección
- `method` = `hough_only` → el ajuste elíptico falló (solo se usó el círculo de Hough)
- Valores NaN o vacíos en columnas `_mm` → falta calibración
- Saltos bruscos entre frames → usar `--smooth` para suavizar

---

## Recomendaciones de parámetros por caso

| Situación | Parámetro a ajustar | Valor sugerido |
|-----------|---------------------|----------------|
| Imágenes oscuras, bajo contraste | `--clip-limit` | 4.0–6.0 |
| Gota muy pequeña en la imagen | `--min-radius` | 20–40 |
| Gota muy grande en la imagen | `--max-radius` | 600–800 |
| Video muy ruidoso | `--smooth` | 5–10 |
| Video muy largo | `--skip` | 15–30 |
| Diagnosticar detecciones | `--visualize` + `--verbose` | — |

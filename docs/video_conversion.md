# Conversión de Video

Esta guía explica cómo preparar tus videos para analizarlos con BubbleCV,
incluyendo conversión de formato, verificación de FPS y extracción de segmentos.

---

## ¿Por qué convertir videos?

OpenCV (la librería de visión que usa BubbleCV) tiene soporte limitado para
algunos formatos y codecs. Los formatos más compatibles son:

| Formato | Compatibilidad | Notas |
|---------|----------------|-------|
| `.mp4` (H.264) | ✅ Excelente | Recomendado en todas las plataformas |
| `.avi` (MJPEG) | ✅ Buena | Compatible, mayor tamaño de archivo |
| `.mov` (H.264) | ⚠️ Variable | Funciona bien en macOS, puede fallar en Windows |
| `.mov` (H.265/HEVC) | ❌ Problemático | Requiere codecs adicionales |
| `.wmv` | ⚠️ Variable | Mejor en Windows |
| `.mkv` | ⚠️ Variable | Depende del codec interno |

**Recomendación:** Convierte siempre a `.mp4` con codec H.264 antes de analizar.

---

## Instalar ffmpeg

`ffmpeg` es la herramienta estándar para conversión de video. Es gratuita
y de código abierto.

### macOS

```bash
# Con Homebrew (recomendado)
brew install ffmpeg

# Verificar
ffmpeg -version
```

### Windows

1. Descarga el build desde [ffmpeg.org/download.html](https://ffmpeg.org/download.html)
   (elige "Windows builds from gyan.dev")
2. Descarga el ZIP "release full"
3. Descomprime en una carpeta fija, por ejemplo `C:\ffmpeg\`
4. Agrega `C:\ffmpeg\bin` al PATH del sistema:
   - Busca "Variables de entorno" en el menú inicio
   - En "Path" del usuario → Agregar `C:\ffmpeg\bin`
5. Abre un CMD nuevo y verifica:
   ```
   ffmpeg -version
   ```

---

## Conversiones comunes

### Convertir a MP4 (H.264) — uso general

```bash
ffmpeg -i path/to/input.mov -c:v libx264 -crf 18 -preset slow path/to/output.mp4
```

| Parámetro | Significado |
|-----------|-------------|
| `-c:v libx264` | Codec de video H.264 |
| `-crf 18` | Calidad (0=mejor, 51=peor; 18-23 es rango recomendado) |
| `-preset slow` | Mayor compresión a cambio de más tiempo |

### Convertir sin pérdida de calidad (lossless)

```bash
ffmpeg -i path/to/input.mov -c:v libx264 -crf 0 path/to/output_lossless.mp4
```

### Convertir desde formato de cámara microscópica (AVI sin comprimir)

```bash
ffmpeg -i path/to/microscope_raw.avi -c:v libx264 -crf 18 path/to/output.mp4
```

### Extraer un segmento (recortar en tiempo)

Útil para analizar solo una parte del video:

```bash
# Extraer desde 0:30 hasta 2:00 (90 segundos)
ffmpeg -i path/to/input.mp4 -ss 00:00:30 -t 00:01:30 -c copy path/to/segment.mp4
```

| Parámetro | Significado |
|-----------|-------------|
| `-ss` | Tiempo de inicio (hh:mm:ss) |
| `-t` | Duración del segmento |
| `-c copy` | Copia sin recodificar (más rápido) |

---

## Verificar el FPS del video

Antes de analizar, es importante conocer los fotogramas por segundo (FPS)
de tu video para que el eje de tiempo del CSV sea correcto.

```bash
# Con ffmpeg
ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate \
    -of default=noprint_wrappers=1:nokey=1 path/to/video.mp4
```

La salida será una fracción como `30/1` (30 FPS) o `25/1` (25 FPS).

**También puedes verlo con:**
```bash
ffprobe -v quiet -print_format json -show_streams path/to/video.mp4 | grep r_frame_rate
```

### FPS comunes y su uso en BubbleCV

| FPS del video | Bandera en BubbleCV | Notas |
|--------------|---------------------|-------|
| 30 FPS | `--fps 30` | Estándar para cámaras USB/web |
| 25 FPS | `--fps 25` | PAL (Europa/México común) |
| 60 FPS | `--fps 60` | Cámaras de alta velocidad |
| 15 FPS | `--fps 15` | Algunas cámaras microscópicas |

---

## Verificar que el video es legible por OpenCV

```bash
python - <<'EOF'
import cv2
cap = cv2.VideoCapture("path/to/video.mp4")
print("Abierto:", cap.isOpened())
print("FPS:", cap.get(cv2.CAP_PROP_FPS))
print("Frames totales:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Resolución:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
      "x", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
cap.release()
EOF
```

Si `Abierto: False`, el video tiene un codec no soportado → convierte con ffmpeg.

---

## Videos muy largos: estrategia de análisis

Para videos de horas de duración, usa `--skip` para reducir el tiempo de procesamiento:

```bash
# Procesar 1 frame por segundo (en un video a 30 FPS)
python analyze_video.py \
    --input path/to/video.mp4 \
    --calibration 114.0 \
    --fps 30 \
    --skip 30
```

Consulta → [`docs/analysis_guide.md`](./analysis_guide.md) para más estrategias.

# Conversión de Video

Esta guía explica cómo preparar los videos para analizarlos con BubbleCV Dual,
incluyendo conversión de formato, verificación de FPS y extracción de segmentos.

---

## ¿Por qué convertir videos?

OpenCV tiene soporte limitado para algunos formatos y codecs. Los formatos
más compatibles son:

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

### macOS

```bash
brew install ffmpeg
ffmpeg -version
```

### Windows

1. Descarga desde [ffmpeg.org/download.html](https://ffmpeg.org/download.html)
   (elige "Windows builds from gyan.dev" → "release full").
2. Descomprime en `C:\ffmpeg\`.
3. Añade `C:\ffmpeg\bin` al PATH del sistema.
4. Abre un CMD nuevo y verifica: `ffmpeg -version`

---

## Inspeccionar la tasa de cuadros del video

La cadencia del video debe inspeccionarse con ffprobe antes de analizar. No
debe suponerse a partir del nombre del archivo ni del FPS nominal del equipo
de captura.

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=r_frame_rate,avg_frame_rate \
  -of default=noprint_wrappers=1:nokey=1 VIDEO.mp4
```

`r_frame_rate` y `avg_frame_rate` son metadatos útiles para detectar
discrepancias. En archivos VFR pueden diferir y no deben interpretarse
automáticamente como una cronología exacta frame a frame.

También puedes ver más información del stream:

```bash
ffprobe -v quiet -print_format json -show_streams VIDEO.mp4 | grep r_frame_rate
```

---

## Conversión a MP4 H.264 (uso general)

```bash
ffmpeg -i input.mov -c:v libx264 -crf 18 output.mp4
```

| Parámetro | Significado |
|-----------|-------------|
| `-c:v libx264` | Codec de video H.264 |
| `-crf 18` | Calidad (0=mejor, 51=peor; 18 es alta calidad) |

Para mayor compresión a cambio de más tiempo de procesamiento:
```bash
ffmpeg -i input.mov -c:v libx264 -crf 18 -preset slow output.mp4
```

---

## Extracción de un segmento temporal

Para analizar solo una parte del video, usa recodificación H.264 en lugar
de `-c copy`. La opción `-c copy` es más rápida pero puede producir
timestamps irregulares (VFR), lo que afecta el cálculo de tiempo en BubbleCV.

```bash
ffmpeg -i input.mov \
  -ss INICIO_EN_SEGUNDOS \
  -t DURACION_EN_SEGUNDOS \
  -c:v libx264 \
  -crf 18 \
  output_segmento.mp4
```

### Ejemplo: protocolo histórico de este proyecto (150–650 s)

En el protocolo histórico de validación de este proyecto se ha utilizado
la ventana **150–650 s** (500 s de duración) en videos suficientemente largos.
Esta ventana se eligió experimentalmente para ese protocolo específico y
**no es una regla universal**: si el protocolo cambia, la ventana debe
justificarse de nuevo.

```bash
ffmpeg -i input.mov \
  -ss 150 \
  -t 500 \
  -c:v libx264 \
  -crf 18 \
  output_150_650.mp4
```

---

## Conversión sin pérdida de calidad (lossless)

```bash
ffmpeg -i input.mov -c:v libx264 -crf 0 output_lossless.mp4
```

---

## Verificar que el video es legible por OpenCV

```bash
python - <<'EOF'
import cv2
cap = cv2.VideoCapture("VIDEO.mp4")
print("Abierto:", cap.isOpened())
print("FPS (metadato):", cap.get(cv2.CAP_PROP_FPS))
print("Frames totales:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Resolución:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
      "x", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
cap.release()
EOF
```

> Nota: OpenCV y ffprobe pueden reportar valores distintos cuando la
temporización del archivo es irregular. Si existe discrepancia, inspecciona
el archivo y normalízalo a CFR antes del análisis.

---

## Estrategia para videos largos

Para videos de muchos minutos o horas, usa `--skip` para reducir el tiempo
de procesamiento:

```bash
# Plantilla para un archivo CFR verificado a 30 FPS.
# Sustituye XX.XX por la calibración correspondiente.
python analyze_video.py \
    --input VIDEO.mp4 \
    --calibration XX.XX \
    --fps 30 \
    --skip 30
```

Consulta → [`docs/analysis_guide.md`](./analysis_guide.md) para el flujo completo.

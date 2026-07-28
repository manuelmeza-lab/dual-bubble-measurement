# Solución de Problemas

Esta guía cubre los errores más comunes al usar BubbleCV y cómo resolverlos.

---

## Errores de instalación

### "No module named 'cv2'"

OpenCV no está instalado o el entorno virtual no está activo.

```bash
# 1. Verificar que el entorno virtual esté activo (debe verse "(venv)" en el prompt)
# Si no está activo:
source venv/bin/activate        # macOS/Linux
venv\Scripts\Activate.ps1       # Windows PowerShell

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Verificar
python -c "import cv2; print(cv2.__version__)"
```

### "No module named 'pandas'" / "No module named 'matplotlib'"

Mismo problema. Ejecuta `pip install -r requirements.txt` con el entorno activo.

---

## Errores al leer archivos

### "Failed to load image"

- Verifica que la ruta sea correcta (sin errores de tipeo)
- Verifica que el formato sea soportado: `.png`, `.jpg`, `.bmp`, `.tiff`
- Si la ruta tiene espacios, enciérrala en comillas:
  ```bash
  python analyze_image.py --input "path/to/my image.png" --calibration 114.0
  ```

### "Failed to open video" / `cap.isOpened()` devuelve `False`

El video tiene un formato o codec no soportado por OpenCV.

```bash
# Verificar si OpenCV puede leer el video
python - <<'EOF'
import cv2
cap = cv2.VideoCapture("path/to/video.mp4")
print("Abierto:", cap.isOpened())
cap.release()
EOF
```

**Soluciones:**

1. Convierte el video a MP4 H.264 con ffmpeg:
   ```bash
   ffmpeg -i path/to/input.mov -c:v libx264 -crf 18 path/to/output.mp4
   ```
   Consulta → [`docs/video_conversion.md`](./video_conversion.md)

2. En macOS, reinstala OpenCV con soporte headless:
   ```bash
   pip uninstall opencv-python
   pip install opencv-python-headless
   ```

---

## Errores de detección

### "No bubble detected" / Detección fallida en todos los frames

La gota no fue detectada. Causas posibles:

| Causa | Solución |
|-------|----------|
| Gota fuera del rango de tamaño esperado | Ajustar `--min-radius` y `--max-radius` |
| Imagen muy oscura, bajo contraste | Aumentar `--clip-limit` (ej: `5.0` o `7.0`) |
| Fondo complejo, mucho ruido | Aumentar `--clip-limit` y reducir `--min-radius` |
| Gota parcialmente fuera del encuadre | No hay solución automática; revisar el video |

**Diagnóstico visual:**
```bash
python analyze_video.py \
    --input path/to/video.mp4 \
    --calibration 114.0 \
    --fps 30 \
    --skip 30 \
    --visualize \
    --vis-dir results/debug_frames \
    --verbose
```

Revisa los frames en `results/debug_frames/` para ver dónde falla la detección.

### La elipse detectada no coincide con la gota

- Aumenta `--clip-limit` para mejorar el contraste y definir mejor el borde
- Ajusta `--min-radius` / `--max-radius` para restringir la búsqueda
- Si la detección usa `method = hough_only`, el ajuste elíptico falló:
  intenta con `--clip-limit` más alto o verifica que la gota tenga al
  menos ~50 px de radio

### Eccentricidad alta y errática entre frames

Causas habituales:
- Iluminación inconsistente entre frames
- Reflejo o brillo central dentro de la gota
- Gota parcialmente fuera de foco

Soluciones:
```bash
# Aplicar suavizado temporal
python analyze_video.py --input path/to/video.mp4 --calibration 114.0 \
    --fps 30 --smooth 5

# Aumentar el contraste
python analyze_video.py --input path/to/video.mp4 --calibration 114.0 \
    --fps 30 --clip-limit 5.0
```

### Mediciones ruidosas o con saltos bruscos

```bash
# Suavizado temporal con ventana de 5 frames
python analyze_video.py --input path/to/video.mp4 --calibration 114.0 \
    --fps 30 --smooth 5

# Aumentar el salto de frames para reducir ruido de alta frecuencia
python analyze_video.py --input path/to/video.mp4 --calibration 114.0 \
    --fps 30 --skip 30 --smooth 5
```

---

## Problemas de calibración

### Columnas `_mm` vacías en el CSV

No se proporcionó calibración al ejecutar el análisis.
```bash
# Agrega --calibration con el valor px/mm
python analyze_video.py --input path/to/video.mp4 --calibration 114.0 --fps 30
```

### La calibración automática detecta el objeto incorrecto

- Asegúrate de que la imagen de calibración tenga un único objeto circular/esférico visible
- Ajusta `--min-radius` y `--max-radius` para que coincidan con el tamaño del objeto de referencia en píxeles
- Usa `--verbose` para ver qué detectó el sistema

---

## Problemas específicos de macOS

### Error de permisos al acceder a la carpeta de videos

macOS protege algunas carpetas. Ve a:
**Ajustes del Sistema → Privacidad y Seguridad → Acceso total al disco**
y agrega Terminal.

### El proceso se interrumpe por suspensión del sistema

Usa `caffeinate` para mantener el sistema activo:
```bash
caffeinate -i python analyze_video.py --input path/to/video.mp4 ...
```
Consulta → [`docs/prevent_sleep.md`](./prevent_sleep.md)

### Error "Operation not permitted" en macOS Sonoma

Puede ocurrir con archivos descargados de internet (atributo de cuarentena):
```bash
xattr -rd com.apple.quarantine /ruta/a/tu/carpeta/
```

---

## Problemas específicos de Windows

### PowerShell no puede ejecutar el script de activación del venv

```powershell
# Ejecutar una sola vez como administrador
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Error con rutas que tienen espacios

```powershell
# Incorrecto
python analyze_video.py --input Mi Carpeta\video.mp4

# Correcto
python analyze_video.py --input "Mi Carpeta\video.mp4"
```

### El video `.mov` no se abre en Windows

Los archivos `.mov` con codec H.265/HEVC no son soportados directamente.
Convierte con ffmpeg:
```bash
ffmpeg -i video.mov -c:v libx264 -crf 18 video.mp4
```
Consulta → [`docs/video_conversion.md`](./video_conversion.md)

---

## Recursos adicionales

- Documentación de OpenCV: [docs.opencv.org](https://docs.opencv.org)
- Abre un Issue en el repositorio si el error persiste

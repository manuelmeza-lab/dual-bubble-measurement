# Solución de Problemas

Esta guía cubre los errores y situaciones problemáticas más comunes al
usar BubbleCV Dual.

**Primera herramienta de diagnóstico:** los frames anotados generados con
`--visualize`. Revisar esas imágenes antes de modificar cualquier parámetro.

---

## Errores de instalación

### "No module named 'cv2'"

OpenCV no está instalado o el entorno virtual no está activo.

```bash
# 1. Verificar que el entorno virtual esté activo (debe verse "(venv)" en el prompt)
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

- Verifica que la ruta sea correcta (sin errores de tipeo).
- Verifica que el formato sea soportado: `.png`, `.jpg`, `.bmp`, `.tiff`.
- Si la ruta tiene espacios, enciérrala en comillas:
  ```bash
  python analyze_image.py --input "path/to/mi imagen.png" --calibration XX.XX
  ```

### "Failed to open video" / `cap.isOpened()` devuelve `False`

El video tiene un formato o codec no soportado por OpenCV.

```bash
python - <<'EOF'
import cv2
cap = cv2.VideoCapture("path/to/video.mp4")
print("Abierto:", cap.isOpened())
cap.release()
EOF
```

**Solución:**

Convierte a MP4 H.264 con ffmpeg:
```bash
ffmpeg -i path/to/input.mov -c:v libx264 -crf 18 path/to/output.mp4
```

Consulta → [`docs/video_conversion.md`](./video_conversion.md)

---

## Problemas de detección dual

### "dual detection failed" en muchos frames

El detector no encontró una o ambas gotas en esos frames.

**Diagnóstico — primero:**
```bash
python analyze_video.py \
    --input VIDEO.mp4 \
    --calibration XX.XX \
    --fps FPS_REAL \
    --skip N \
    --visualize \
    --vis-dir RESULTS/debug \
    --verbose
```

Revisa los frames en `RESULTS/debug/` para ver dónde y por qué falla.
Causas comunes:

| Causa | Señal |
|-------|-------|
| Gota fuera del ROI definido | Elipse en posición incorrecta o ausente |
| Gota muy pequeña / muy grande | Sin detección Hough |
| Imagen de bajo contraste | Segmentación incorrecta visible en frame anotado |
| Capilar visible dentro del contorno | Cuerpo libre no aislado correctamente |

### geometry_quality_valid = False en muchos frames

El ajuste bodyellipse produce `residual_rmse > 0.08`.

**Diagnóstico:**

1. Revisa los frames anotados. ¿La elipse verde sigue el borde externo de la
   gota o se inclina / sobredimensiona?
2. ¿La línea amarilla (body_start) aparece dentro del capilar en lugar de en
   la transición cuello-cuerpo?
3. ¿Hay reflejos internos brillantes que el detector confunde con el borde?
4. ¿La gota vibra físicamente o está fuera de foco?

**Acciones según causa:**

| Causa probable | Acción |
|----------------|--------|
| Frames del inicio/fin del experimento (gota inestable) | Recortar la ventana temporal |
| Vibración física | Investigar la causa experimental; no es corregible en el software |
| Reflejos internos dominando la segmentación | Revisar condiciones de iluminación |
| body_start dentro del capilar | El pipeline detecta la transición; si falla sistemáticamente, revisar si la gota tiene forma inusual |

**Lo que NO debes hacer:** cambiar `BODYELLIPSE_MAX_RESIDUAL_RMSE = 0.08`
para "reducir" el rechazo. Ese umbral refleja calidad geométrica real.

### residual_rmse > 0.08 en frames aislados (no agrupados)

Es esperable en algunos frames por perturbaciones momentáneas. El filtro
los excluye automáticamente. Revisa que no estén agrupados temporalmente.

### La elipse no sigue el borde externo de la gota

- Revisa en los frames anotados si el contorno magenta (puntos enviados a
  fitEllipse) incluye el capilar superior.
- Revisa si la línea body_start (amarilla) está demasiado alta.
- No aumentes `--clip-limit` arbitrariamente para "arreglar" esto; puede
  introducir artefactos en otros frames.

### El contorno incluye reflejos internos

Los reflejos especulares internos a la gota pueden hacer que la segmentación
incluya esa región como parte del objeto. Esto produce contornos irregulares
y residuos altos. Diagnóstico: ver el contorno magenta en los frames anotados.

### Bodyellipse no produce resultado (body_contour_lt_5_points)

El contorno físico del cuerpo libre tiene menos de 5 puntos después de filtrar
por body_start_y. Causa probable: la gota es muy pequeña en esa región o la
segmentación la fragmenta. No se inventa ningún punto; el frame se descarta.

---

## Problemas de calibración

### Columnas `_mm` vacías o NaN en el CSV

No se proporcionó calibración. Añade `--calibration XX.XX` con el valor
medido para tu configuración óptica.

### La calibración automática detecta el objeto incorrecto

- Verifica que la imagen tenga un único objeto circular bien visible.
- Ajusta `--min-radius` / `--max-radius` para restringir la búsqueda al
  tamaño del objeto de referencia en píxeles.
- Usa `--verbose` para ver qué detectó el sistema.
- Si no converge, usa calibración manual (ver `docs/calibration.md`).

### El FPS es incorrecto

Si `--fps` no corresponde a la cadencia del archivo analizado, el eje de
tiempo del CSV será incorrecto y K quedará escalado incorrectamente.
Inspecciona los metadatos con ffprobe:

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=r_frame_rate,avg_frame_rate \
  -of default=noprint_wrappers=1:nokey=1 VIDEO.mp4
```

---

## Problemas de calidad de resultados

### R² bajo en el ajuste lineal

No es directamente un problema del software. Causas posibles:

- La gota no está en régimen estacionario de evaporación (ventana temporal
  mal elegida: inicio o fin de experimento).
- Perturbaciones externas durante el experimento.
- Frames rechazados por QC agrupados temporalmente (revisar distribución
  de `geometry_quality_valid = False` en el CSV).
- Mezcla de regímenes de evaporación distintos.

**No** se debe ajustar parámetros del software para "mejorar" el R².

### Saltos o escalones en la serie temporal

- Revisa los frames anotados en la región del salto.
- Puede ser un cambio real en la gota (perturbación física, vibración, contacto).
- Puede ser un frame con detección errónea que pasó el QC.

### Tendencia no lineal visible

El modelo r_eq²(t) = r0² − K·t es lineal. Si la tendencia observada no es
lineal, el régimen de evaporación puede ser diferente o la ventana temporal
incluye fases distintas.

---

## Problemas específicos de macOS

### Error de permisos al acceder a la carpeta de videos

Ve a **Ajustes del Sistema → Privacidad y Seguridad → Acceso total al disco**
y añade Terminal.

### El proceso se interrumpe por suspensión del sistema

```bash
caffeinate -i python analyze_video.py --input VIDEO.mp4 ...
```

Consulta → [`docs/prevent_sleep.md`](./prevent_sleep.md)

### Error "Operation not permitted" en macOS Sonoma

Puede ocurrir con archivos descargados de internet:
```bash
xattr -rd com.apple.quarantine /ruta/a/tu/carpeta/
```

---

## Problemas específicos de Windows

### PowerShell no puede ejecutar el script de activación del venv

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Rutas con espacios

```powershell
# Incorrecto
python analyze_video.py --input Mi Carpeta\video.mp4

# Correcto
python analyze_video.py --input "Mi Carpeta\video.mp4"
```

### El video `.mov` no se abre en Windows

Convierte con ffmpeg:
```bash
ffmpeg -i video.mov -c:v libx264 -crf 18 video.mp4
```

---

## Recursos adicionales

- Documentación de OpenCV: [docs.opencv.org](https://docs.opencv.org)
- Manual operativo: [`docs/analysis_guide.md`](./analysis_guide.md)

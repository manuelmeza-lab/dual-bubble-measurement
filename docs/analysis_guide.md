# Manual de Análisis — BubbleCV Dual

Este documento es el manual operativo principal. Está dirigido a quien quiera
analizar un nuevo experimento sin necesidad de conocer el historial de desarrollo.

---

## A. Flujo general

```
1.  Activar entorno virtual
2.  Revisar / convertir el video
3.  Definir la ventana experimental
4.  Calibrar (px/mm)
5.  Ejecutar el análisis
6.  Revisar QC (tracking_valid, geometry_quality_valid)
7.  Revisar summary y binned
8.  Aceptar la corrida o repetirla
9.  Obtener K a partir de la pendiente
10. Calcular D (si corresponde, con el modelo físico y condiciones T/RH)
```

---

## B. Preparación del video

### Formato recomendado

Usar **MP4 con codec H.264**. Otros formatos (MOV, AVI, MKV) pueden funcionar
según el sistema operativo, pero MP4/H.264 es el más compatible con OpenCV.

Consulta → [`docs/video_conversion.md`](./video_conversion.md)

### Verificar el FPS del video con ffprobe

BubbleCV calcula el tiempo de cada frame como `frame_num / fps`. Por ello,
el valor de `--fps` debe corresponder a la cadencia del archivo que realmente
se analizará y no debe suponerse a partir del equipo de captura.

Inspecciona tanto `r_frame_rate` como `avg_frame_rate`. Son metadatos útiles
para diagnosticar la cadencia del video, pero en archivos VFR pueden diferir
y ninguno debe interpretarse automáticamente como una cronología exacta
frame a frame:

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=r_frame_rate,avg_frame_rate \
  -of default=noprint_wrappers=1 VIDEO.mp4
```

Si los dos valores difieren o el archivo es VFR (variable frame rate),
conviene normalizar a CFR antes de analizar (ver más abajo).

### Conversión a CFR (cuando es necesario)

Para convertir el video a una cadencia de cuadros constante (CFR), aplica
un filtro de FPS explícito durante la conversión:

```bash
ffmpeg -i input.mov \
  -vf "fps=FPS_OBJETIVO" \
  -c:v libx264 \
  -crf 18 \
  -pix_fmt yuv420p \
  output_cfr.mp4
```

`FPS_OBJETIVO` debe elegirse y justificarse según el video y el protocolo
experimental. No fijes 30 FPS de forma genérica.

Después de convertir, vuelve a verificar el archivo resultante con ffprobe.

### Ventana temporal

La elección de la ventana depende del protocolo experimental. Para que la
regresión lineal sea válida, la gota debe estar en régimen estacionario de
evaporación (ni en la etapa inicial de estabilización ni en colapso).

**En el protocolo histórico de validación de este proyecto** se ha utilizado
la ventana **150–650 s** en videos suficientemente largos. Esta ventana se
eligió experimentalmente para ese protocolo y **no es una ley universal**.
Si el protocolo cambia, la ventana debe justificarse de nuevo.

Si tu video pertenece al mismo protocolo y dura lo suficiente, el segmento
puede extraerse aplicando simultáneamente el recorte temporal y la normalización
CFR (ver `docs/video_conversion.md` para el comando completo).

---

## C. Calibración

La calibración expresa la relación **píxeles por milímetro (px/mm)** de la
configuración óptica usada. Este valor **nunca debe tomarse de otro experimento
o de un valor genérico**: depende de la resolución, zoom, distancia de trabajo
y posición de la cámara. Recalibrar si cambia cualquiera de estos factores.

Consulta → [`docs/calibration.md`](./calibration.md) para el procedimiento
completo.

### Cómo funciona la calibración automática

La calibración automática está implementada en `bubble_cv/calibration.py`
y usa `detect_reference_object()` de `bubble_cv/detection.py`.

Este detector es **completamente independiente** del pipeline de detección dual:

- Opera sobre el **frame completo** (sin ROIs izquierda/derecha).
- No aplica filtros de gota colgante (no hay ventana de posición vertical,
  no hay umbral de excentricidad de gota, no hay aislamiento cuello/cuerpo).
- No asume que haya un capilar presente.
- La métrica primaria de calibración es el **diámetro del círculo Hough**
  (`2 × radio_Hough`), robusto para un objeto esférico simétrico.
- Se realiza adicionalmente un ajuste elíptico sobre la máscara Otsu,
  pero solo como diagnóstico: si la diferencia entre el diámetro Hough y el
  diámetro equivalente de la elipse supera el 5%, se emite una advertencia
  en el log. La métrica devuelta al llamador es siempre el diámetro Hough.

El resultado se calcula como:

```
px_to_mm = diámetro_Hough_px / diámetro_conocido_mm
```

Comando de calibración automática:

```bash
python analyze_image.py \
  --input referencia.png \
  --calibrate-from referencia.png \
  --ref-diameter 4.0 \
  --verbose
```

La línea de salida relevante es:

```
[INFO] Calibration: reference diameter detected = XX.XX px | known diameter = Y.YY mm | calibration = ZZ.ZZZZ px/mm
```

Usa el valor `ZZ.ZZZZ` en `--calibration` para el análisis de video.

---

## D. Comando estándar para análisis dual

Copia y adapta esta plantilla (sustituye todos los valores en MAYÚSCULAS):

```bash
python analyze_video.py \
  --input "VIDEO.mp4" \
  --calibration PX_PER_MM \
  --fps FPS_REAL \
  --skip N \
  --clip-limit 3.0 \
  --max-eccentricity 0.85 \
  --r2-fit \
  --bin-size-s 10 \
  --output "RESULTS/results_dual.csv" \
  --summary-output "RESULTS/summary_dual.csv" \
  --binned-output "RESULTS/results_dual_binned.csv" \
  --visualize \
  --vis-dir "RESULTS/annotated" \
  --plot
```

### Parámetros clave

| Parámetro | Valor por defecto (código) | Notas |
|-----------|--------------------------|-------|
| `--calibration PX_PER_MM` | — (requerido) | Valor medido con tu configuración óptica |
| `--fps FPS_REAL` | 30.0 | Verificar con ffprobe; no suponer |
| `--skip N` | 1 | Para ≈1 medición/s: `N ≈ FPS_REAL` |
| `--clip-limit` | **3.0** | Contraste local CLAHE |
| `--max-eccentricity` | **0.85** | Umbral QC de excentricidad |
| `--bin-size-s` | None (desactivado) | Ventana de agrupado temporal en segundos |
| `--r2-fit` | desactivado | Activa el ajuste lineal r_eq² vs tiempo |
| `--visualize` | desactivado | Genera frames anotados para auditoría |

Los valores listados son los **valores por defecto del software**. No hay
evidencia en el código de que algún valor específico distinto al default
haya sido usado en una validación experimental publicada.

> **Importante:** No modifiques `--clip-limit` ni `--max-eccentricity` para
> que K o R² "se vean mejor". Estos parámetros son de preprocesamiento y QC
> geométrico, respectivamente; ajustarlos para mejorar resultados introduce
> sesgo.

---

## E. Control de calidad (QC)

Existen **dos niveles de QC independientes**, ambos exportados al CSV.

### Nivel 1 — `tracking_valid`

Se evalúa por gota en cada frame. Es `False` si:
- excentricidad > `--max-eccentricity` (default 0.85)
- `equiv_diameter_mm ≤ 0` o `None`
- `volume_mm3 ≤ 0` o `None`

### Nivel 2 — `geometry_quality_valid`

Se evalúa a partir de la calidad del ajuste bodyellipse:

```
geometry_quality_valid = (bodyellipse_residual_rmse ≤ 0.08)
```

El umbral está declarado como constante en `analyze_video.py`:

```python
BODYELLIPSE_MAX_RESIDUAL_RMSE = 0.08
```

Este umbral se estableció observando que los ajustes normales producen
`residual_rmse ≈ 0.035–0.05`, mientras que detecciones incorrectas forman
una población separada con `residual_rmse > 0.14–0.20`. El valor 0.08 se
sitúa entre ambas poblaciones y se eligió por criterio geométrico,
**no optimizando K ni R²**.

### Qué significa un frame con `geometry_quality_valid = False`

- Sus medidas (major_axis, minor_axis, volume, etc.) **se conservan en el CSV**.
- Los frames rechazados pueden visualizarse con `--visualize` para auditoría.
- **No se interpolan**.
- **No se usan** para regresión (r_eq² vs tiempo), binning ni cálculo de K.

### Motivos de rechazo registrados

| `geometry_quality_rejection_reason` | Significado |
|------------------------------------|-------------|
| `bodyellipse_residual_rmse` | RMSE del ajuste > 0.08 |
| `bodyellipse_residual_rmse_missing` | RMSE no disponible o no finito para esa detección. Para determinar la causa concreta, inspeccionar el frame anotado y el log con `--verbose`. |

---

## F. Archivos de salida

| Archivo | Contenido |
|---------|-----------|
| `results_dual.csv` | Todas las mediciones frame a frame con todas las banderas QC. Cada fila tiene columnas `control_*` y `sample_*`. |
| `summary_dual.csv` | Pendiente r_eq² vs tiempo (`slope_radius2_mm2_s`), intercepto, R², y conteos de frames válidos/rechazados. |
| `results_dual_binned.csv` | Datos agrupados en bins temporales (`--bin-size-s`). Solo incluye frames con `tracking_valid=True` y `geometry_quality_valid=True`. |
| `annotated/` | Frames anotados: elipse verde (ajuste bodyellipse), contorno físico (magenta), línea body_start (amarilla), métricas. Primera herramienta de auditoría visual. |
| Gráficas `.png` | Curvas temporales de ambas gotas superpuestas. Útil para inspección rápida de tendencia y outliers. |

### Columnas QC en results_dual.csv

```
control_tracking_valid                    True / False
control_rejection_reason                  (cadena, vacía si válido)
control_geometry_quality_valid            True / False
control_geometry_quality_rejection_reason (cadena, vacía si válido)
sample_tracking_valid
...
```

---

## G. Interpretación de K

El modelo de decaimiento lineal de r_eq² predice:

```
r_eq²(t) = r0² + m·t
```

donde `m` es la pendiente del ajuste lineal (negativa, porque r² disminuye).

Para reportar la **tasa de evaporación K** como cantidad positiva con
unidades mm²/s:

```
K = −m = −slope_radius2_mm2_s
```

donde `slope_radius2_mm2_s` es la columna del archivo `summary_dual.csv`.

> **Atención:** K tiene unidades **mm²/s** y corresponde al decaimiento lineal
> de r_eq². No es lo mismo que dV/dt, que tiene unidades mm³/s.

El cálculo del coeficiente de difusión D a partir de K queda fuera de esta
guía y depende del modelo físico adoptado para el experimento y de las
condiciones experimentales (T, HR, presión).

---

## H. Criterios para revisar una corrida

No existe un porcentaje universal de rechazo ni un R² mínimo universal.
Una corrida **requiere revisión** cuando se observa alguna de las siguientes
situaciones:

- Fracción elevada de frames con `geometry_quality_valid = False`.
- Los frames rechazados aparecen **agrupados temporalmente** (no aleatorios),
  lo que puede indicar vibración física, perturbación pasajera o cambio de
  iluminación.
- R² claramente bajo.
- La serie temporal muestra saltos o escalones bruscos.
- En los frames anotados, la elipse verde no sigue el cuerpo libre de la gota.
- La línea body_start (amarilla) aparece dentro del capilar.
- La gota vibra o se deforma físicamente.
- Cambios bruscos en `center_y_px`, `major_axis_px` o `minor_axis_px`.

### Lo que NO debe hacerse

- Eliminar manualmente filas del CSV porque el valor no gusta.
- Subir `--clip-limit` arbitrariamente hasta que la detección "mejore".
- Usar `--smooth` para esconder outliers reales.
- Cambiar `BODYELLIPSE_MAX_RESIDUAL_RMSE = 0.08` buscando obtener
  una pendiente K más favorable.

---

## I. Validación interna del pipeline

Esta sección registra los resultados de las corridas de validación internas
(water-water) realizadas durante el desarrollo del sistema. Se incluyen como
**registro histórico**, no como valores que un nuevo experimento deba
reproducir.

Con el filtro geométrico `RMSE ≤ 0.08` aplicado:

**Corrida 3 (water-water)**
- control: K ≈ 0.000444 mm²/s
- sample:  K ≈ 0.000488 mm²/s

**Corrida 4 (water-water)**
- control: K ≈ 0.000462 mm²/s
- sample:  K ≈ 0.000451 mm²/s

Los valores de control y sample son similares entre sí en cada corrida y
consistentes entre corridas. Esta consistencia interna apoya la validación
del pipeline y demuestra reproducibilidad bajo las condiciones de ensayo
utilizadas. No constituye validación universal para otros líquidos,
condiciones ópticas o protocolos distintos.

---

*Siguiente referencia técnica: [`docs/calibration.md`](./calibration.md)*

# Calibración px/mm

Para obtener mediciones en unidades físicas (mm, mm², mm³), el sistema
necesita conocer la relación **píxeles por milímetro** (px/mm) de tu
configuración óptica.

---

## ¿Por qué es necesario calibrar?

Sin calibración, el sistema reporta solo medidas en píxeles. Con
calibración, las columnas `_mm`, `_mm2` y `_mm3` del CSV se llenan
con valores físicos reales.

La relación px/mm depende de todos los factores del sistema óptico:

- resolución de captura de la cámara
- nivel de zoom / objetivo del microscopio
- distancia de trabajo
- posición axial de la cámara

**Debe recalibrarse si cambia cualquiera de estos parámetros.**

Nunca uses un valor px/mm de otro experimento, de otra sesión con diferente
zoom, ni un valor genérico: la calibración es específica de cada configuración.

---

## Consideraciones de precisión

- Calibra con la **misma resolución, zoom, distancia de trabajo y posición
  óptica** que usarás durante el experimento.
- No muevas la cámara entre la calibración y el experimento.
- La relación px/mm puede variar entre la parte central y los bordes de la
  imagen (distorsión de lente). Para mayor precisión, coloca el objeto de
  referencia **cerca de la posición donde aparece la gota** en el experimento.
- En sistemas de doble gota, la posición izquierda y derecha pueden tener
  factores px/mm ligeramente distintos si hay distorsión espacial. BubbleCV
  aplica un único factor a ambas gotas; si la distorsión es significativa,
  documéntalo como fuente de incertidumbre.

---

## Objetos de referencia

Usa un objeto de **dimensiones conocidas y estables**.

El objeto utilizado en este proyecto es una **esfera de 4 mm de diámetro**.
El valor de `--ref-diameter` debe corresponder al diámetro real de tu objeto
de referencia en milímetros.

---

## Método A — Calibración automática (recomendada)

La calibración automática está implementada en `bubble_cv/calibration.py`
y usa la función `detect_reference_object()` de `bubble_cv/detection.py`.

### Cómo funciona el detector de calibración

Este detector es **completamente independiente** del pipeline de detección
dual de gotas:

- Opera sobre el **frame completo** (sin ROIs izquierda/derecha).
- No aplica filtros de gota colgante (no hay ventana de posición vertical,
  no hay umbral de excentricidad de gota, no hay aislamiento cuello/cuerpo,
  no hay restricción de posición del capilar).
- No asume que haya un capilar presente.
- La métrica primaria es el **diámetro del círculo de Hough**
  (`2 × radio_Hough`), que es robusto para un objeto esférico simétrico.
- Se realiza adicionalmente un ajuste elíptico sobre la máscara Otsu,
  pero **únicamente como diagnóstico**: si la diferencia entre el diámetro
  Hough y el diámetro equivalente de la elipse supera el 5%, se emite una
  advertencia en el log. La métrica devuelta al llamador es siempre el
  diámetro Hough.

El px/mm se calcula como:
```
px_to_mm = diámetro_Hough_px / diámetro_conocido_mm
```

### Comando

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

Si la calibración automática falla (no detecta el objeto), verifica que
la imagen tenga un único objeto circular bien definido y ajusta
`--min-radius` / `--max-radius` para que coincidan con el tamaño del
objeto en píxeles. Usa `--verbose` para ver qué detectó el sistema.

---

## Método B — Calibración manual

1. Captura una imagen del objeto de referencia con exactamente la misma
   configuración óptica que usarás para el experimento.

2. Mide el diámetro del objeto en píxeles con un visor de imágenes:
   - **macOS (Preview):** Herramientas → Anotar → Regla
   - **Windows (Paint):** posición del cursor en la barra de estado
   - **Fiji/ImageJ:** herramienta de línea + `Ctrl+M`

3. Calcula la relación:
   ```
   px_to_mm = diámetro_en_píxeles / diámetro_real_en_mm
   ```

4. Usa ese valor con `--calibration`:
   ```bash
   python analyze_video.py --input VIDEO.mp4 --calibration XX.XX --fps FPS_REAL
   ```

---

Siguiente paso → [`docs/analysis_guide.md`](./analysis_guide.md)

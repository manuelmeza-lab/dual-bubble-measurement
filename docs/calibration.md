# Calibración px/mm

Para obtener mediciones en unidades físicas (mm, mm², mm³), el sistema
necesita conocer la relación **píxeles por milímetro** (px/mm) de tu
configuración óptica. Esta guía explica cómo obtener ese valor.

---

## ¿Por qué es necesario calibrar?

Sin calibración, el sistema reporta solo medidas en píxeles. Con
calibración, las columnas `_mm`, `_mm2` y `_mm3` del CSV se llenan
con valores físicos reales.

La relación px/mm depende de:
- El aumento del objetivo del microscopio
- La resolución de captura de la cámara
- La distancia de trabajo

**Debe recalibrarse si cambias cualquiera de estos parámetros.**

---

## Objetos de referencia recomendados

Usa un objeto de **dimensiones conocidas y estables**. Ejemplos comunes:

| Objeto | Tamaño típico | Notas |
|--------|--------------|-------|
| Esfera de referencia | 4.0 mm de diámetro | Alta precisión, fácil de medir en imagen |
| Regla micrométrica | Graduada en µm | Ideal para calibración precisa |
| Papel milimétrico | 1 mm por cuadro | Útil para estimaciones rápidas |
| Objeto de dimensión conocida | Variable | Cualquier objeto medible |

---

## Método A — Calibración manual

1. Captura una imagen del objeto de referencia con la misma
   configuración que usarás para tus gotas (mismo zoom, misma distancia)

2. Abre la imagen en cualquier visor y mide el objeto en píxeles:

   - **macOS (Preview):** Abre la imagen → Menú `Herramientas → Mostrar Inspector`
     → Pestaña de regla. También puedes usar `Herramientas → Anotar → Regla`.
   - **Windows (Paint):** Abre la imagen → pasa el cursor sobre los extremos
     del objeto. Las coordenadas se muestran en la barra de estado.
   - **Fiji/ImageJ (multiplataforma):** Abre la imagen → usa la herramienta
     de Línea (`m`) y mide con `Ctrl+M`.

3. Calcula la relación:
   ```
   px_to_mm = diámetro_en_píxeles / diámetro_real_en_mm

   Ejemplo: si la esfera de 4 mm mide 456 px en la imagen:
   px_to_mm = 456 / 4.0 = 114.0
   ```

4. Usa ese valor con la bandera `--calibration`:
   ```bash
   python analyze_image.py --input path/to/drop.png --calibration 114.0
   python analyze_video.py --input path/to/video.mp4 --calibration 114.0 --fps 30
   ```

---

## Método B — Calibración automática

Deja que el sistema detecte automáticamente el objeto de referencia
y calcule la relación px/mm:

```bash
python analyze_image.py \
    --input path/to/drop.png \
    --calibrate-from path/to/calibration.png \
    --ref-diameter 4.0
```

| Bandera | Descripción |
|---------|-------------|
| `--calibrate-from` | Imagen del objeto de referencia |
| `--ref-diameter` | Diámetro real del objeto de referencia en mm |

El sistema detectará la esfera en la imagen de calibración usando el
mismo pipeline de Hough Circle, calculará el px/mm automáticamente
y lo aplicará al análisis.

> **Consejo:** Si la calibración automática falla, usa el Método A
> para obtener el valor manualmente y pásalo con `--calibration`.

---

## Verificar la calibración

Usa `--verbose` para ver el valor calculado en la terminal:

```bash
python analyze_image.py \
    --input path/to/drop.png \
    --calibrate-from path/to/calibration.png \
    --ref-diameter 4.0 \
    --verbose
```

La salida incluirá una línea similar a:
```
[INFO] Calibration: 114.25 px/mm (error: 0.2%)
```

Un error menor al 5% se considera aceptable para la mayoría de los análisis.

---

## Tabla de valores típicos (referencia)

Los siguientes valores son solo de referencia. **Siempre calibra con tu
propia imagen** porque los valores reales dependen de tu equipo específico.

| Configuración | px/mm aproximado |
|--------------|-----------------|
| Resolución baja (640×480), zoom estándar | ~50–80 |
| Resolución media (1280×720), zoom estándar | ~100–150 |
| Resolución alta (1920×1080), zoom estándar | ~180–250 |

---

## Consideraciones importantes

- **Calibra en las mismas condiciones que el experimento:** misma distancia,
  mismo zoom, misma resolución.
- **No muevas la cámara** entre la calibración y el experimento.
- **Si la cámara se mueve**, repite la calibración.
- La relación px/mm puede variar entre la parte central y los bordes de
  la imagen (distorsión de lente). Para máxima precisión, coloca el objeto
  de referencia en la región donde aparecerá la gota.

---

Siguiente paso → [`docs/analysis_guide.md`](./analysis_guide.md)

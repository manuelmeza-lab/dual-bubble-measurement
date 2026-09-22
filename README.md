# BubbleCV Dual

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python ≥ 3.9](https://img.shields.io/badge/Python-%E2%89%A53.9-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green.svg)](https://opencv.org/)

Herramienta de visión por computadora para medir y comparar simultáneamente
**dos gotas colgantes** (Control vs. Muestra) a partir de videos de microscopio.

---

## Convención espacial

| Posición en la imagen | Etiqueta |
|---|---|
| Gota izquierda | `control` |
| Gota derecha | `sample` |

Esta convención es fija durante todo el análisis.

---

## Forma recomendada de uso

En `v0.2.1`, la forma recomendada para una corrida ordinaria es:

`python run_bubblecv.py`

El Guided Runner solicita paso a paso:

- video de entrada;
- calibración en px/mm;
- FPS;
- frecuencia de muestreo (`skip`);
- BODY automático o fijo para cada gota;
- parámetros de análisis;
- carpeta de resultados.

Antes de ejecutar muestra la configuración completa y solicita confirmación.

Cada corrida genera además un archivo `run_manifest.json` con:

- SHA-256 del video;
- parámetros utilizados;
- entorno de software;
- commit Git;
- comando ejecutado;
- código de salida.

Guía rápida:

→ [`docs/quickstart.md`](./docs/quickstart.md)

---

## Pipeline científico

Cada frame se procesa de manera independiente:

1. Localización gruesa mediante Hough Circle por ROI fija.
2. Recorte dinámico alrededor de cada gota.
3. Segmentación mediante umbral adaptativo y operaciones morfológicas.
4. Selección del contorno físico.
5. Separación cuello / cuerpo libre (BODY).
6. Ajuste `bodyellipse` sobre los puntos físicos del cuerpo libre.
7. Filtros geométricos y tracking.
8. Diagnósticos de calidad del ajuste.
9. `contour_consensus` QC cuando se utiliza BODY fijo.
10. Análisis temporal independiente de control y muestra.
11. Binning temporal y ajuste robusto Theil–Sen de `r_eq²` frente al tiempo.

### BODY automático y BODY fijo

Por defecto BubbleCV determina automáticamente la transición cuello → cuerpo.

También puede fijarse una coordenada Y global explícita para CONTROL y/o SAMPLE
cuando existe una frontera física previamente establecida.

### BODYELLIPSE RMSE

El residual RMSE del ajuste `bodyellipse` se conserva como **diagnóstico
geométrico**.

Un RMSE elevado **no constituye por sí mismo un veto analítico** en la versión
actual del pipeline.

### Contour consensus

El QC `contour_consensus`:

- sólo es aplicable cuando se especifica BODY fijo;
- no sustituye el cálculo de radio ni el ajuste geométrico;
- funciona como control de calidad analítico del contorno físico;
- con BODY automático no aplica este veto.

---

## Resultado de evaporación

El análisis independiente produce estimaciones de la pendiente de:

`r_eq² vs tiempo`

El resumen incluye:

- ajuste frame a frame;
- ajuste sobre datos binned;
- ajuste robusto Theil–Sen sobre datos binned.

La estimación robusta de la tasa se reporta en:

`robust_binned_K_mm2_s`

junto con:

`robust_binned_r_squared`

---

## Requisitos

- Python ≥ 3.9
- OpenCV
- NumPy
- Pandas
- Matplotlib

Existen dos especificaciones de dependencias:

- `requirements.txt`: dependencias flexibles para instalación general.
- `requirements-lock.txt`: entorno de referencia con versiones exactas.

El lock de referencia de `v0.2.1` fue reconstruido y validado con:

- Python 3.9.6
- macOS ARM64
- OpenCV 5.0.0
- NumPy 2.0.2
- Pandas 2.3.3
- Matplotlib 3.9.4

---

## Instalación rápida

Clona el repositorio:

`git clone https://github.com/manuelmeza-lab/dual-bubble-measurement.git`

Entra a la carpeta:

`cd dual-bubble-measurement`

Crea y activa un entorno virtual:

`python3 -m venv venv`

`source venv/bin/activate`

Instala el entorno de referencia:

`python -m pip install --upgrade pip`

`python -m pip install -r requirements-lock.txt`

Guías específicas:

- macOS: [`docs/macos.md`](./docs/macos.md)
- Windows: [`docs/windows.md`](./docs/windows.md)

---

## Primera corrida

Ejecuta:

`python run_bubblecv.py`

Guía paso a paso:

→ [`docs/quickstart.md`](./docs/quickstart.md)

Manual técnico:

→ [`docs/analysis_guide.md`](./docs/analysis_guide.md)

---

## Datos experimentales

Los videos, imágenes de calibración, datos crudos y resultados numéricos son
externos al repositorio y no se versionan.

La carpeta `results/` generada por el Guided Runner también está excluida
mediante `.gitignore`.

---

## Tests

Ejecuta:

`python -m unittest discover -s tests -p 'test_*.py'`

---

## Licencia

MIT — ver [`LICENSE`](./LICENSE).

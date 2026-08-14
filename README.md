# BubbleCV Dual

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python ≥ 3.9](https://img.shields.io/badge/Python-%E2%89%A53.9-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green.svg)](https://opencv.org/)

Herramienta de visión por computadora para medir y comparar simultáneamente
**dos gotas colgantes** (Control vs. Muestra) a partir de videos de microscopio.

---

## Convención espacial

| Posición en la imagen | Etiqueta  |
|----------------------|-----------|
| Gota izquierda       | `control` |
| Gota derecha         | `sample`  |

Esta convención es fija y se aplica en cada frame independientemente.

---

## Pipeline de detección

Cada frame se procesa de forma independiente (sin información temporal):

1. **Localización gruesa** — Hough Circle por ROI fija (izquierda / derecha).
2. **Recorte dinámico** — ROI ajustada al radio Hough detectado.
3. **Segmentación** — `adaptiveThreshold` + cierre morfológico (`MORPH_CLOSE`).
4. **Componente principal** — contorno de mayor área dentro del recorte.
5. **Separación cuello / cuerpo libre** — perfil de anchura `width(y)`;
   `body_start_y` se determina detectando la transición cuello→cuerpo
   (expansión sostenida respecto a la anchura mediana del capilar).
6. **Ajuste bodyellipse** — `cv2.fitEllipse` sobre los puntos físicos del
   contorno con `y ≥ body_start_y`; sin frontera horizontal artificial.
7. **Filtros geométricos** — tamaño de ejes, posición del centro, recorte
   por borde, excentricidad.
8. **QC independiente** por gota:
   - `tracking_valid` — excentricidad, diámetro, volumen.
   - `geometry_quality_valid` — `residual_rmse ≤ 0.08` del ajuste bodyellipse.
9. **Ajuste lineal** — `r_eq² vs tiempo` para obtener la pendiente `K`.

La calibración se expresa en **px/mm** y debe medirse con la misma
configuración óptica del experimento.

---

## Lo que NO está en este repositorio

Los datos experimentales, videos, imágenes de calibración y resultados
numéricos son externos al repositorio y no se incluyen ni se versionan aquí.

---

## Requisitos

- Python ≥ 3.9
- Dependencias: `opencv-python`, `numpy`, `pandas`, `matplotlib`

---

## Instalación

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\Activate.ps1    # Windows PowerShell

pip install -r requirements.txt
```

---

## Manual operativo

→ [`docs/analysis_guide.md`](./docs/analysis_guide.md)

Cubre: preparación del video, calibración, comando estándar, interpretación
de QC, salidas, interpretación de K y criterios de revisión de una corrida.

---

## Licencia

MIT — ver [`LICENSE`](./LICENSE).

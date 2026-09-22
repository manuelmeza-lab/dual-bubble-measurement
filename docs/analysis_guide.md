# Manual de análisis — BubbleCV Dual

Este documento describe el uso operativo y la semántica científica del
pipeline actual de BubbleCV Dual.

Está dirigido a quien necesite analizar un experimento nuevo, interpretar sus
salidas y auditar la calidad de una corrida sin conocer el historial completo
de desarrollo.

Para una primera corrida sencilla consulta primero:

→ [`quickstart.md`](./quickstart.md)

---

## A. Convención espacial

BubbleCV utiliza siempre:

- gota izquierda = `control`
- gota derecha = `sample`

Esta clasificación espacial es fija durante el análisis.

La detección, validez y ajuste temporal de cada gota se conservan de forma
independiente.

La ausencia o rechazo analítico de una gota no invalida automáticamente a la
gota contralateral.

---

## B. Flujo recomendado

1. Activar el entorno virtual.
2. Revisar el video.
3. Confirmar su cadencia temporal.
4. Confirmar la calibración en px/mm.
5. Definir BODY automático o fijo para cada gota.
6. Ejecutar `run_bubblecv.py`.
7. Revisar el resumen de detección y QC.
8. Revisar `summary.csv`.
9. Revisar `binned.csv` y las gráficas.
10. Auditar `results.csv` cuando sea necesario.
11. Conservar `run_manifest.json` junto con los resultados.
12. Interpretar K dentro del modelo físico correspondiente al experimento.

---

## C. Preparación del video

### Formato recomendado

Se recomienda:

- contenedor MP4;
- codec H.264;
- cadencia CFR cuando el protocolo requiera una escala temporal uniforme.

Consulta:

→ [`video_conversion.md`](./video_conversion.md)

### FPS

BubbleCV calcula:

`timestamp_s = frame_num / fps`

Por ello, el valor de FPS debe corresponder al archivo que realmente se
analiza.

No debe suponerse a partir de la configuración nominal de la cámara si el
archivo fue convertido, recodificado o modificado.

Si `ffprobe` está disponible, el Guided Runner muestra:

- codec;
- resolución;
- `r_frame_rate`;
- `avg_frame_rate`;
- duración.

Estos metadatos son ayudas de diagnóstico.

En archivos VFR, una tasa media no sustituye una cronología frame a frame. Si
el protocolo requiere tiempo uniforme conviene normalizar previamente a CFR.

### `skip`

`skip` significa procesar uno de cada N frames.

Ejemplo:

- video = 10 FPS;
- `skip = 10`;

produce aproximadamente una medición por segundo.

El valor adecuado depende del protocolo experimental.

---

## D. Calibración

La calibración se expresa en:

`px/mm`

Debe corresponder a la misma configuración óptica utilizada para adquirir el
video.

Debe revisarse o repetirse si cambia, por ejemplo:

- resolución;
- zoom;
- óptica;
- distancia de trabajo;
- configuración de captura;
- geometría de la cámara.

Consulta:

→ [`calibration.md`](./calibration.md)

Para análisis físicos de `radius_eq_mm2`, volumen y K se requiere una
calibración válida.

---

## E. Guided Runner

La forma recomendada para una corrida ordinaria en `v0.2.1` es:

`python run_bubblecv.py`

El Guided Runner solicita:

- video;
- calibración;
- FPS;
- skip;
- BODY de CONTROL;
- BODY de SAMPLE;
- `clip-limit`;
- excentricidad máxima;
- suavizado;
- tamaño de bin;
- carpeta de resultados.

Antes de ejecutar muestra el comando completo y solicita confirmación.

Si el usuario cancela, no crea la carpeta de resultados.

Si ejecuta, crea además:

`run_manifest.json`

con información suficiente para identificar la entrada, parámetros y entorno
de la corrida.

---

## F. Ejecución directa avanzada

El Guided Runner construye finalmente una llamada a `analyze_video.py`.

Una plantilla directa equivalente es:

`python analyze_video.py --input VIDEO.mp4 --calibration PX_PER_MM --fps FPS_REAL --skip N --clip-limit 3.0 --max-eccentricity 0.85 --smooth 0 --r2-fit --bin-size-s 10 --output RESULTS/results.csv --summary-output RESULTS/summary.csv --binned-output RESULTS/binned.csv`

BODY fijo puede añadirse de forma independiente:

CONTROL:

`--control-body-start-y Y_GLOBAL`

SAMPLE:

`--sample-body-start-y Y_GLOBAL`

Si el argumento se omite, ese lado conserva BODY automático.

### Parámetros principales

| Parámetro | Default del código | Significado |
|---|---:|---|
| `--fps` | 30.0 | FPS utilizados para construir el eje temporal |
| `--skip` | 1 | Procesar uno de cada N frames |
| `--clip-limit` | 3.0 | Contraste local CLAHE |
| `--max-eccentricity` | 0.85 | Límite de excentricidad para `tracking_valid` |
| `--smooth` | 0 | Mediana móvil temporal; 0 = desactivada |
| `--bin-size-s` | None | Tamaño del bin temporal |
| `--r2-fit` | desactivado | Activa ajustes de `r_eq²` frente al tiempo |
| `--control-body-start-y` | None | BODY automático para CONTROL |
| `--sample-body-start-y` | None | BODY automático para SAMPLE |

No deben modificarse parámetros después de observar K o R² buscando obtener
un resultado más favorable.

---

## G. BODY automático y BODY fijo

### BODY automático

Cuando no se proporciona una coordenada fija, BubbleCV determina la transición
cuello → cuerpo libre utilizando el selector automático del pipeline.

La coordenada resultante se conserva en las variables diagnósticas:

- `body_start_y_global`
- `body_start_y_local`

### BODY fijo

Puede definirse una frontera física BODY mediante una coordenada Y global del
frame.

Ejemplo:

`--sample-body-start-y 56`

El valor:

- debe ser un entero;
- está expresado en píxeles globales del frame;
- debe caer dentro del intervalo vertical de la ROI correspondiente;
- no se recorta silenciosamente si está fuera de rango.

BODY fijo debe utilizarse cuando existe una justificación física o experimental
previa.

No debe elegirse desplazándolo hasta mejorar K o R².

---

## H. Ajuste `bodyellipse`

El contorno físico del cuerpo libre se utiliza para ajustar una elipse.

BubbleCV conserva, entre otros:

- número de puntos utilizados;
- ejes mayor y menor;
- centro;
- excentricidad;
- área de la elipse ajustada;
- residual medio;
- residual RMSE;
- residual P95.

Los residuos describen la calidad geométrica del ajuste.

---

## I. `geometry_quality_valid`: diagnóstico, no veto analítico

BubbleCV continúa calculando:

`geometry_quality_valid`

a partir del residual RMSE de `bodyellipse`.

El criterio diagnóstico actual es:

`bodyellipse_residual_rmse <= 0.08`

Por ello pueden aparecer en `results.csv`:

- `geometry_quality_valid = True`
- `geometry_quality_valid = False`

y su correspondiente:

`geometry_quality_rejection_reason`

### Semántica actual

Esta bandera se conserva para auditoría geométrica.

**No forma parte por sí sola de la máscara que decide si una medición entra en
el análisis temporal.**

Por tanto:

`geometry_quality_valid = False`

no significa automáticamente:

- eliminar la medición;
- excluirla del ajuste;
- excluirla del binning independiente;
- invalidar a la gota contralateral.

El RMSE sigue siendo útil para localizar frames geométricamente inusuales.

---

## J. `tracking_valid`

La validez básica de cada detección se evalúa de manera independiente.

`tracking_valid` es falso si falla alguno de los criterios físicos básicos,
incluyendo:

- excentricidad superior a `--max-eccentricity`;
- `equiv_diameter_mm` ausente o no positivo;
- `volume_mm3` ausente o no positivo.

Las razones se conservan en:

`rejection_reason`

---

## K. Contour-consensus QC — Fix8

Cuando se utiliza una frontera BODY fija, BubbleCV aplica un control adicional
sobre el contorno físico seleccionado.

Las columnas principales son:

- `contour_consensus_applicable`
- `contour_consensus_valid`
- `contour_consensus_support_fraction`
- `contour_consensus_n_inliers`
- `contour_consensus_n_points`
- `contour_consensus_rejection_reason`

### Aplicabilidad

Con BODY fijo:

`contour_consensus_applicable = True`

Con BODY automático:

`contour_consensus_applicable = False`

y el consenso no actúa como veto.

### Qué evalúa

El procedimiento busca de forma determinista una elipse geométricamente
plausible apoyada por una fracción suficiente de los puntos del contorno.

La especificación congelada utiliza:

- tolerancia radial = 2 px;
- soporte mínimo = 50 % de los puntos;
- búsqueda RANSAC determinista;
- máximo de 3000 iteraciones.

Las restricciones básicas de plausibilidad son consistentes con el dominio
geométrico del detector:

- eje mayor entre 35 y 125 px;
- eje menor entre 25 y 100 px;
- relación `minor / major >= 0.65`.

### Importante

Contour consensus:

- no reemplaza la elipse medida;
- no modifica `radius_eq_mm2`;
- no cambia la selección original de la medición;
- conserva la medición original en el CSV incluso si la veta analíticamente.

Es un QC analítico independiente aplicado únicamente cuando corresponde.

---

## L. Validez analítica real de cada gota

La máscara analítica central se construye de forma independiente para CONTROL
y SAMPLE.

Una observación de un lado entra en el análisis cuando:

1. `tracking_valid == True`
2. `radius_eq_mm2` está disponible
3. y, si `contour_consensus_applicable == True`,
   entonces `contour_consensus_valid == True`

En forma conceptual:

`analytical_valid = tracking_valid AND radius_available AND consensus_pass_if_applicable`

### Lo que NO veta por sí solo

No se utiliza como veto independiente:

- `geometry_quality_valid`
- RMSE de bodyellipse
- temporal QC
- validez de la gota contralateral

### Independencia de las gotas

Si SAMPLE falta en un frame pero CONTROL es válido, CONTROL puede conservarse
para su análisis independiente.

Lo mismo ocurre en sentido inverso.

---

## M. QC temporal

BubbleCV calcula también metadatos de QC temporal.

En la versión actual son diagnósticos.

No modifican:

- `tracking_valid`;
- `geometry_quality_valid`;
- la máscara analítica central;
- dV/dt;
- binning independiente;
- OLS;
- Theil–Sen;
- K.

---

## N. Conteos del resumen de detección

Para cada gota se distinguen:

### `attempted`

Número de timestamps intentados.

### `detections`

Frames donde existe una detección para ese lado.

### `missing`

Frames sin detección para ese lado.

### `geometry_valid`

Detecciones con `geometry_quality_valid = True`.

### `geometry_rejected`

Detecciones con `geometry_quality_valid = False`.

Este conteo es diagnóstico y no debe confundirse con rechazo analítico.

### `analytical_valid`

Frames que pasan la máscara analítica real de ese lado.

### `qc_rejected`

Frames detectados pero no analíticamente válidos.

### `unusable`

Frames que no aportan al análisis independiente de ese lado.

---

## O. `results.csv`

`results.csv` conserva una fila por timestamp intentado.

Incluye:

- `frame_id`;
- `timestamp_s`;
- `control_detected`;
- `sample_detected`;
- mediciones de CONTROL;
- mediciones de SAMPLE;
- tracking QC;
- diagnósticos bodyellipse;
- geometry quality;
- BODY;
- contour consensus;
- QC temporal;
- tasas derivadas cuando corresponda.

Una gota ausente no provoca que se inventen valores geométricos para ese lado.

La detección válida de la gota contralateral se conserva.

---

## P. `binned.csv`: producto pareado

Este detalle es importante.

El archivo exportado como:

`binned.csv`

es un producto **pareado para comparación simultánea**.

Para construirlo BubbleCV utiliza la intersección:

`control_analytical_valid AND sample_analytical_valid`

Por tanto, una fila entra en este CSV sólo cuando ambas gotas son
analíticamente válidas en ese timestamp.

Dentro de cada bin se guardan estadísticas como:

- tiempo medio;
- número de puntos;
- media;
- desviación estándar;
- mediana de `radius_eq_mm2`.

---

## Q. Binning independiente para los ajustes

Aunque `binned.csv` es pareado, los ajustes de cada gota que aparecen en
`summary.csv` se calculan de forma **independiente**.

Para cada lado BubbleCV reconstruye sus propios bins usando únicamente su
máscara analítica.

Esto evita que la pérdida de SAMPLE elimine innecesariamente un punto válido
de CONTROL, y viceversa.

Por esta razón:

- el CSV binned sirve para comparación pareada;
- los fits binned del summary son por lado;
- `n_bins` del summary pertenece al análisis independiente de esa gota.

La mediana de `radius_eq_mm2` utilizada por Theil–Sen se almacena redondeada a
4 decimales antes del ajuste.

No se impone un mínimo adicional de puntos por bin.

---

## R. Ajustes de `r_eq²` frente al tiempo

Cuando se activa:

`--r2-fit`

BubbleCV calcula tres descripciones por gota.

### 1. OLS frame a frame

Columnas:

- `slope_radius2_mm2_s`
- `intercept_radius2_mm2`
- `r_squared_fit`

### 2. OLS binned independiente

Columnas:

- `binned_slope_radius2_mm2_s`
- `binned_intercept_radius2_mm2`
- `binned_r_squared`

### 3. Theil–Sen robusto binned independiente

Columnas:

- `robust_binned_slope_radius2_mm2_s`
- `robust_binned_intercept_radius2_mm2`
- `robust_binned_r_squared`
- `robust_binned_K_mm2_s`
- `robust_binned_n_pairwise_slopes`
- `robust_binned_statistic`
- `robust_binned_fit_method`

---

## S. Especificación del ajuste robusto Theil–Sen

Para todos los pares válidos `i < j` con tiempos distintos:

`slope_ij = (y_j - y_i) / (x_j - x_i)`

La pendiente robusta es:

`slope = median(slope_ij)`

El intercepto conjunto es:

`intercept = median(y - slope*x)`

El R² se calcula contra esa recta robusta.

La tasa K se define exactamente como:

`K = -slope`

### Importante

BubbleCV **no utiliza `abs(slope)`**.

Si la pendiente fuera positiva, K resultaría negativa.

Eso debe interpretarse como una señal que requiere revisión científica del
experimento o de la ventana analizada, no corregirse cambiando el signo.

---

## T. Interpretación de K

Para un régimen donde:

`r_eq²(t) = r0² - K*t`

K tiene unidades:

`mm²/s`

y describe el decaimiento de `r_eq²`.

No es equivalente directamente a:

`dV/dt`

que tiene unidades de volumen por tiempo.

La conversión de K a un coeficiente de difusión u otra magnitud física depende
del modelo adoptado y de las condiciones experimentales, por ejemplo:

- temperatura;
- humedad relativa;
- presión;
- composición;
- geometría y condiciones de frontera.

BubbleCV no impone ese modelo físico posterior.

---

## U. `summary.csv`

Contiene una fila para CONTROL y una para SAMPLE.

Además de los ajustes, conserva conteos independientes:

- `total_frames`
- `detected_frames`
- `missing_frames`
- `valid_frames`
- `qc_rejected_frames`
- `unusable_frames`
- `rejected_frames`

`valid_frames` representa la validez analítica del lado.

No equivale necesariamente a `geometry_quality_valid`.

---

## V. `run_manifest.json`

Las corridas realizadas mediante `run_bubblecv.py` guardan un manifest con:

### Entrada

- ruta absoluta del video;
- SHA-256;
- metadatos de ffprobe cuando están disponibles.

### Parámetros

- calibración;
- FPS;
- skip;
- BODY fijo/automático;
- clip limit;
- excentricidad;
- smooth;
- bin size;
- activación de R² fit.

### Entorno

- versión de Python;
- ejecutable de Python;
- plataforma;
- arquitectura;
- OpenCV;
- NumPy;
- Pandas;
- Matplotlib;
- commit Git;
- rama Git;
- tag exacto si existe.

### Ejecución

- comando;
- hora de inicio;
- hora de finalización;
- código de salida;
- estado final.

Este archivo permite conservar la identidad computacional de la corrida junto
con sus resultados.

---

## W. Cómo revisar una corrida

No existe un R² mínimo universal ni un porcentaje universal de rechazo válido
para todos los experimentos.

Revisa como mínimo:

1. identidad del video;
2. SHA-256 cuando se requiera trazabilidad;
3. FPS;
4. calibración;
5. posición CONTROL/SAMPLE;
6. BODY utilizado;
7. detecciones y frames ausentes;
8. `analytical_valid`;
9. `geometry_quality_valid` como diagnóstico;
10. contour consensus cuando sea aplicable;
11. evolución temporal de `r_eq²`;
12. número de bins;
13. OLS raw;
14. OLS binned;
15. Theil–Sen;
16. gráficas;
17. video original cuando aparezcan anomalías.

Una diferencia entre OLS y Theil–Sen no debe corregirse automáticamente.

Puede revelar sensibilidad a:

- outliers;
- regiones finales de la evaporación;
- cambios de régimen;
- fallos de detección;
- perturbaciones experimentales.

---

## X. Lo que NO debe hacerse

No:

- borrar manualmente filas porque el valor no gusta;
- modificar `clip-limit` para obtener otra K;
- cambiar el FPS para ajustar la pendiente;
- aumentar la excentricidad máxima después de observar el resultado;
- utilizar smooth para ocultar outliers;
- desplazar BODY buscando maximizar R²;
- interpretar automáticamente RMSE > 0.08 como rechazo analítico;
- desactivar contour consensus porque veta un resultado no deseado;
- utilizar la gota contralateral para invalidar una medición individual válida;
- sustituir la pendiente Theil–Sen por su valor absoluto.

Los cambios metodológicos deben justificarse antes de interpretar el resultado.

---

## Y. Validación clean-room de `v0.2.0`

La versión científica congelada `v0.2.0` fue reproducida desde:

- clon Git limpio;
- commit `7abca93448a4d46a9b75b59211d455f4a91328d3`;
- entorno virtual nuevo;
- dependencias instaladas desde cero;
- videos de entrada verificados mediante SHA-256.

La suite del release contiene:

`152 tests`

con:

`OK (skipped=1)`

El test omitido depende de un CSV histórico externo que no forma parte del
repositorio limpio.

### Matriz clean-room utilizada

Se reprodujeron cinco videos:

1. Corrida 3 — agua/agua
2. Corrida 4 — agua/agua
3. Hexadecanol V1
4. Hexadecanol V2
5. Hexadecanol V5

V3 y V4 de la serie de hexadecanol no se utilizaron porque esas preparaciones
experimentales no eran válidas para esta matriz de validación.

### Resultados robustos reproducidos

| Video | CONTROL K (mm²/s) | SAMPLE K (mm²/s) |
|---|---:|---:|
| Corrida 3 agua/agua | 0.000450281 | 0.000489056 |
| Corrida 4 agua/agua | 0.0004605329311211671 | 0.000444444444444446 |
| Hexadecanol V1 | 0.0004016666666666668 | 0.0004228571428571428 |
| Hexadecanol V2 | 0.00019451387846961736 | 0.00025846741045214327 |
| Hexadecanol V5 | 0.0005170910931174089 | 0.00008173913043478253 |

Estos valores son registros de reproducibilidad computacional de esas entradas
específicas.

No constituyen valores universales esperados para nuevos experimentos.

---

## Z. Prueba de aceptación del Guided Runner

Durante el desarrollo de `v0.2.1`, Hexadecanol V5 se ejecutó mediante
`run_bubblecv.py` utilizando los mismos parámetros del análisis clean-room.

Se compararon contra la ejecución directa de `v0.2.0`:

- `results.csv`
- `binned.csv`
- `summary.csv`

Los tres archivos resultaron:

- iguales mediante comparación exacta de DataFrames;
- idénticos byte por byte;
- con SHA-256 de salida idénticos.

Por tanto, en esa prueba de aceptación el Guided Runner funcionó como una capa
operativa reproducible sin modificar las salidas científicas de
`analyze_video.py`.

---

## Referencias internas

Guía rápida:

→ [`quickstart.md`](./quickstart.md)

Calibración:

→ [`calibration.md`](./calibration.md)

Conversión de video:

→ [`video_conversion.md`](./video_conversion.md)

Solución de problemas:

→ [`troubleshooting.md`](./troubleshooting.md)

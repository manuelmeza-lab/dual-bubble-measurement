# BubbleCV Dual — Guía rápida

Esta guía está pensada para realizar una primera corrida de BubbleCV sin
necesidad de conocer los argumentos internos de `analyze_video.py`.

---

## 1. Activar BubbleCV

Abre una Terminal dentro de la carpeta del proyecto.

### macOS / Linux

`source venv/bin/activate`

### Windows PowerShell

`venv\Scripts\Activate.ps1`

Cuando el entorno virtual esté activo aparecerá normalmente:

`(venv)`

al inicio de la línea de Terminal.

---

## 2. Información que necesitas antes de comenzar

Ten preparados:

1. El archivo de video.
2. La calibración óptica en **px/mm**.
3. Los FPS reales del archivo que será analizado.
4. La decisión experimental sobre BODY:
   - automático, o
   - frontera física fija previamente determinada.

No modifiques parámetros después de observar K o R² con el objetivo de
obtener un resultado más favorable.

---

## 3. Convención de las gotas

BubbleCV utiliza siempre:

- **gota izquierda = CONTROL**
- **gota derecha = SAMPLE**

Esta convención es fija.

Antes de analizar, verifica visualmente que el experimento respete esta
disposición.

---

## 4. Iniciar el Guided Runner

Con el entorno virtual activo ejecuta:

`python run_bubblecv.py`

BubbleCV realizará las preguntas una por una.

---

## 5. Seleccionar el video

Introduce la ruta completa del video.

Ejemplo en macOS:

`/Users/nombre/Videos/experimento01.mp4`

Ejemplo en Windows:

`C:\Users\nombre\Videos\experimento01.mp4`

Si `ffprobe` está disponible, BubbleCV mostrará información como:

- codec;
- resolución;
- `r_frame_rate`;
- `avg_frame_rate`;
- duración.

El FPS detectado sirve como ayuda y debe confirmarse de acuerdo con el archivo
y el protocolo experimental.

---

## 6. Introducir la calibración

BubbleCV solicita:

`Calibración (px/mm)`

Ejemplo:

`34.08`

La calibración debe corresponder a la misma configuración óptica utilizada
para adquirir el video.

Debe repetirse si cambia, por ejemplo:

- resolución;
- zoom;
- óptica;
- distancia de trabajo;
- posición o configuración de la cámara.

---

## 7. Confirmar los FPS

BubbleCV calcula el tiempo de cada frame utilizando:

`tiempo = número de frame / FPS`

Por tanto, un FPS incorrecto modifica directamente la escala temporal del
experimento.

Ejemplo:

`10`

No utilices un FPS genérico si el archivo real tiene otra cadencia.

---

## 8. Elegir `skip`

`skip` indica cada cuántos frames se realizará una medición.

Por ejemplo, para obtener aproximadamente una medición por segundo en un
video de 10 FPS puede utilizarse:

`skip = 10`

El Guided Runner propone inicialmente un valor cercano al FPS introducido.

---

## 9. Configurar BODY

BubbleCV pregunta por CONTROL y SAMPLE de forma independiente:

`1 = automático`

`2 = frontera física fija`

### BODY automático

BubbleCV determina automáticamente la transición cuello → cuerpo libre.

### BODY fijo

Se introduce una coordenada Y global en píxeles.

Debe utilizarse únicamente cuando existe una frontera física definida y
previamente establecida para ese experimento.

No muevas BODY buscando mejorar K, R² o la apariencia de la curva.

Cuando BODY es fijo, BubbleCV puede aplicar el QC adicional de
`contour_consensus`.

---

## 10. Parámetros del análisis

El Guided Runner propone los siguientes valores:

- `clip-limit = 3.0`
- `max eccentricity = 0.85`
- `smooth = 0`
- `bin size = 10 s`

Si el protocolo experimental no establece otra cosa, conserva los parámetros
predefinidos para la corrida.

No deben ajustarse después de observar el resultado con el objetivo de hacer
que K o R² parezcan mejores.

---

## 11. Carpeta de resultados

Por defecto BubbleCV propone una carpeta de la forma:

`results/NOMBRE_VIDEO_FECHA_HORA/`

Cada análisis utiliza una carpeta nueva.

El Guided Runner se niega a ejecutar si la carpeta elegida ya existe. Esto
reduce el riesgo de sobrescribir accidentalmente una corrida anterior.

---

## 12. Revisar antes de ejecutar

Antes de comenzar el análisis BubbleCV muestra:

- video;
- calibración;
- FPS;
- skip;
- BODY de CONTROL;
- BODY de SAMPLE;
- `clip-limit`;
- excentricidad máxima;
- smooth;
- tamaño de bin;
- carpeta de resultados;
- comando exacto que se ejecutará.

Revisa esta información antes de confirmar.

Para ejecutar responde:

`s`

Para cancelar responde:

`n`

Si cancelas, no se crea una carpeta de resultados.

---

## 13. Durante el análisis

BubbleCV procesa las dos gotas de forma independiente y muestra periódicamente
el avance.

Puede aparecer un mensaje indicando que una detección falló en algún frame
aislado.

Eso no significa automáticamente que toda la corrida haya fallado.

La evaluación debe hacerse con el resumen final y las variables de QC.

---

## 14. Finalización correcta

Al terminar, una ejecución correcta muestra:

`exit_code : 0`

Además, BubbleCV imprime:

- SHA-256 del video;
- ubicación del manifest;
- carpeta de resultados;
- contenido de `summary.csv`.

---

## 15. Archivos principales de salida

### `results.csv`

Contiene las mediciones frame a frame para CONTROL y SAMPLE, además de las
variables de diagnóstico y QC.

### `binned.csv`

Contiene los datos agrupados en intervalos temporales.

### `summary.csv`

Contiene el resumen del análisis independiente de evaporación para ambas
gotas.

Incluye:

- pendiente frame a frame;
- R² frame a frame;
- pendiente binned;
- R² binned;
- ajuste robusto Theil–Sen;
- K robusto;
- número de bins;
- conteos de frames.

### `run_manifest.json`

Registra de manera reproducible:

- ruta del video;
- SHA-256 del video;
- metadatos disponibles;
- parámetros utilizados;
- versiones de Python y dependencias;
- commit Git;
- rama Git;
- comando ejecutado;
- código de salida;
- fecha de inicio y finalización.

### Gráficas PNG

BubbleCV genera gráficas temporales para facilitar la inspección de las dos
gotas.

---

## 16. Qué valor de K revisar

La estimación robusta principal se encuentra en `summary.csv`:

`robust_binned_K_mm2_s`

Su correspondiente calidad de ajuste se encuentra en:

`robust_binned_r_squared`

El software también conserva ajustes OLS frame a frame y binned como
información diagnóstica.

K se expresa en:

`mm²/s`

y corresponde al decaimiento de `r_eq²` con el tiempo.

No debe confundirse directamente con `dV/dt`.

---

## 17. QC actual

### `tracking_valid`

Representa la validez básica de la detección y de las magnitudes geométricas.

### BODYELLIPSE RMSE

El residual RMSE del ajuste `bodyellipse` es un **diagnóstico geométrico**.

Un valor superior a 0.08 no elimina automáticamente un frame del análisis
científico en la versión actual.

### `contour_consensus`

Cuando BODY es fijo, BubbleCV utiliza un QC adicional basado en consenso del
contorno físico.

Cuando BODY es automático, este QC no es aplicable y no funciona como veto.

---

## 18. Qué revisar si algo parece extraño

Revisa primero:

- el video original;
- FPS;
- calibración;
- posición izquierda/derecha de las gotas;
- BODY utilizado;
- número de frames detectados;
- número de frames analíticamente válidos;
- evolución temporal de `r_eq²`;
- número de bins;
- gráficas;
- variables de QC.

No existe un R² mínimo universal válido para todos los experimentos.

---

## 19. Lo que no debe hacerse

No:

- borrar filas porque el resultado no gusta;
- modificar `clip-limit` para obtener un K distinto;
- cambiar la excentricidad máxima después de ver el resultado;
- utilizar smooth para ocultar outliers;
- desplazar BODY buscando mejorar R²;
- interpretar automáticamente RMSE > 0.08 como rechazo analítico;
- cambiar el FPS para hacer coincidir una pendiente esperada.

Los parámetros deben definirse por razones experimentales o metodológicas,
no por el resultado obtenido.

---

## 20. Terminar la sesión

Cuando acabes:

`deactivate`

Para una descripción técnica más completa consulta:

→ [`analysis_guide.md`](./analysis_guide.md)

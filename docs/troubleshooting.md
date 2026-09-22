# Solución de problemas — BubbleCV Dual

Esta guía cubre problemas frecuentes de instalación, video, detección y
control de calidad.

Para una primera corrida consulta:

→ [`quickstart.md`](./quickstart.md)

Para la semántica científica completa:

→ [`analysis_guide.md`](./analysis_guide.md)

---

## 1. Antes de modificar parámetros

Si una corrida produce un resultado extraño, no cambies inmediatamente
`clip-limit`, excentricidad, BODY, FPS o suavizado.

Primero revisa:

1. video de entrada;
2. FPS;
3. calibración;
4. posición CONTROL/SAMPLE;
5. BODY utilizado;
6. resumen de detección;
7. `results.csv`;
8. `summary.csv`;
9. gráficas;
10. QC geométrico y contour consensus.

Los parámetros no deben modificarse buscando mejorar K o R².

---

## 2. El entorno virtual no está activo

Síntomas frecuentes:

`No module named 'cv2'`

`No module named 'pandas'`

`No module named 'matplotlib'`

### macOS / Linux

Activa:

`source venv/bin/activate`

### Windows PowerShell

Activa:

`venv\Scripts\Activate.ps1`

Después comprueba:

`python -m pip check`

---

## 3. Dependencias faltantes

### macOS de referencia

Si estás reproduciendo el entorno validado:

`python -m pip install -r requirements-lock.txt`

### Instalación general

`python -m pip install -r requirements.txt`

Después:

`python -m pip check`

Una instalación sin conflictos debe mostrar:

`No broken requirements found.`

---

## 4. El video no existe

El Guided Runner muestra:

`ERROR: no existe el archivo`

Comprueba la ruta introducida.

En macOS puedes arrastrar un archivo desde Finder hacia Terminal para obtener
su ruta completa.

---

## 5. El video no puede abrirse

Puede existir un problema de codec o contenedor.

Se recomienda:

- MP4;
- H.264;
- CFR cuando el protocolo requiera cadencia constante.

Consulta:

→ [`video_conversion.md`](./video_conversion.md)

---

## 6. BubbleCV no muestra metadatos del video

El Guided Runner utiliza `ffprobe` cuando está disponible.

Comprueba:

`ffprobe -version`

En macOS puede instalarse mediante:

`brew install ffmpeg`

Si `ffprobe` no está disponible, el Guided Runner puede seguir funcionando,
pero no mostrará automáticamente codec, resolución, FPS y duración.

---

## 7. FPS incorrecto

BubbleCV calcula:

`timestamp_s = frame_num / fps`

Por ello, un FPS incorrecto escala directamente el eje temporal y las
pendientes.

No cambies el FPS para intentar obtener una K esperada.

Comprueba el archivo con:

`ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate,avg_frame_rate,duration -of default=noprint_wrappers=1 VIDEO.mp4`

Si el archivo es VFR y el protocolo requiere tiempo uniforme, conviértelo a
CFR antes del análisis.

---

## 8. Calibración incorrecta

La calibración está expresada en:

`px/mm`

Debe corresponder a la configuración óptica del video analizado.

Si cambió:

- zoom;
- resolución;
- óptica;
- distancia de trabajo;
- posición de cámara;

la calibración debe revisarse.

Consulta:

→ [`calibration.md`](./calibration.md)

---

## 9. Fallos de detección en frames aislados

Puede aparecer un mensaje como:

`ROI [control]: detection failed`

o:

`ROI [sample]: detection failed`

Eso significa que en ese frame concreto no sobrevivió una detección válida
para ese lado.

No significa automáticamente que toda la corrida haya fallado.

BubbleCV conserva la información de cada gota independientemente.

Por ejemplo:

- SAMPLE puede faltar;
- CONTROL puede seguir siendo analíticamente válido.

Y viceversa.

Revisa al final:

- `detections`;
- `missing`;
- `analytical_valid`;
- `qc_rejected`;
- `unusable`.

---

## 10. Muchas detecciones ausentes

Si `missing` es elevado, revisa:

- posición de la gota dentro de la ROI;
- tamaño aparente;
- foco;
- contraste;
- vibración;
- iluminación;
- desaparición física de la gota al final del experimento.

No aumentes arbitrariamente parámetros del detector para forzar detecciones.

---

## 11. BODY automático parece incorrecto

Con BODY automático, BubbleCV estima la transición cuello → cuerpo libre.

Revisa las variables:

- `body_start_y_global`;
- `body_start_y_local`;
- `body_max_width`.

Si necesitas una auditoría visual avanzada puedes ejecutar directamente
`analyze_video.py` con `--visualize`.

Ejemplo conceptual:

`python analyze_video.py --input VIDEO.mp4 --calibration PX_PER_MM --fps FPS --skip N --r2-fit --bin-size-s 10 --visualize --vis-dir DEBUG_FRAMES`

No fijes BODY únicamente porque otra coordenada produzca mejor R².

---

## 12. BODY fijo produce un error

Una coordenada BODY fija:

- debe ser un entero;
- representa Y global del frame;
- debe caer dentro del intervalo vertical de la ROI.

BubbleCV no corrige silenciosamente una frontera fuera de rango.

Ejemplo:

`--sample-body-start-y 56`

Si aparece un error de frontera, revisa que la coordenada corresponda
realmente al frame completo y no a un recorte local.

---

## 13. `geometry_quality_valid = False`

Esta bandera describe la calidad geométrica del ajuste `bodyellipse`.

El criterio diagnóstico actual utiliza:

`bodyellipse_residual_rmse <= 0.08`

Por ello:

`geometry_quality_valid = False`

indica que el ajuste presenta un residual RMSE superior al umbral o que el
residual requerido no está disponible.

### Importante

En la versión actual:

**`geometry_quality_valid = False` NO veta automáticamente la medición.**

No implica por sí solo que el frame quede fuera de:

- dV/dt;
- ajuste OLS;
- binning independiente;
- ajuste Theil–Sen;
- cálculo de K.

El RMSE se conserva como diagnóstico geométrico.

---

## 14. RMSE > 0.08 en algunos frames

No elimines automáticamente esos frames.

Utiliza el RMSE para localizar regiones que merecen inspección.

Revisa:

- contorno físico;
- forma de la gota;
- final de evaporación;
- vibración;
- desenfoque;
- reflejos;
- cambios bruscos de geometría.

Un grupo temporal de RMSE altos puede ser informativo aunque no actúe como
veto analítico.

---

## 15. `geometry_rejected` es alto pero `analytical_valid` también es alto

Esto puede ser completamente coherente con la semántica actual.

`geometry_rejected` cuenta detecciones con:

`geometry_quality_valid = False`

mientras que `analytical_valid` utiliza la máscara analítica real.

Por tanto, ambos números no representan la misma cosa.

No interpretes:

`geometry_rejected`

como sinónimo de:

`qc_rejected`

---

## 16. ¿Qué determina realmente la validez analítica?

Para cada gota, una observación es analíticamente válida cuando:

1. `tracking_valid == True`;
2. `radius_eq_mm2` está disponible;
3. y, si contour consensus es aplicable,
   `contour_consensus_valid == True`.

Conceptualmente:

`tracking_valid AND radius_available AND consensus_pass_if_applicable`

La validez de la otra gota no forma parte de esta decisión individual.

---

## 17. `tracking_valid = False`

Puede ocurrir por:

- excentricidad mayor al máximo permitido;
- diámetro equivalente ausente o no positivo;
- volumen ausente o no positivo.

Consulta:

`rejection_reason`

para conocer la causa registrada.

No incrementes `--max-eccentricity` después de observar el resultado para
recuperar artificialmente frames rechazados.

---

## 18. Contour consensus

Contour consensus es un QC diferente al RMSE de bodyellipse.

Sus columnas incluyen:

- `contour_consensus_applicable`;
- `contour_consensus_valid`;
- `contour_consensus_support_fraction`;
- `contour_consensus_n_inliers`;
- `contour_consensus_n_points`;
- `contour_consensus_rejection_reason`.

---

## 19. `contour_consensus_applicable = False`

Es esperado cuando el lado utiliza BODY automático.

En ese caso contour consensus:

- no es aplicable;
- no veta la medición.

No interpretes `contour_consensus_valid` aisladamente sin revisar primero
`contour_consensus_applicable`.

---

## 20. `contour_consensus_applicable = True` y `valid = False`

Esto ocurre con BODY fijo cuando el contorno físico no alcanza el consenso
geométrico requerido.

En ese caso sí existe un veto analítico para esa medición.

La medición original:

- no se borra;
- permanece en `results.csv`;
- puede auditarse posteriormente.

Revisa:

- `contour_consensus_support_fraction`;
- `contour_consensus_n_inliers`;
- `contour_consensus_n_points`;
- `contour_consensus_rejection_reason`.

No desactives este QC únicamente para recuperar una medición cuyo resultado
parece conveniente.

---

## 21. `low_consensus`

La razón:

`low_consensus`

indica que menos del soporte mínimo requerido pudo sostener una elipse
geométricamente plausible bajo el procedimiento de consenso.

El umbral científico congelado es:

`support_fraction >= 0.50`

con tolerancia radial de:

`2 px`

No modifiques esos valores buscando cambiar K.

---

## 22. `fewer_than_5_points`

Contour consensus necesita al menos cinco puntos para ajustar una elipse.

La razón:

`fewer_than_5_points`

significa que el contorno disponible no contiene suficientes puntos.

Debe revisarse la detección y el video; no deben inventarse puntos.

---

## 23. `no_plausible_model`

Significa que el procedimiento de consenso no encontró una elipse compatible
con las restricciones geométricas del detector.

Revisa:

- segmentación;
- BODY;
- borde físico;
- tamaño de la gota;
- deformación real;
- calidad óptica.

---

## 24. CONTROL y SAMPLE tienen diferente número de frames válidos

Es posible y esperado.

BubbleCV analiza cada lado independientemente.

Una detección ausente o inválida en SAMPLE no elimina automáticamente una
detección válida de CONTROL.

Por eso `summary.csv` puede mostrar diferentes:

- `detected_frames`;
- `missing_frames`;
- `valid_frames`;
- `qc_rejected_frames`;
- `unusable_frames`.

---

## 25. `binned.csv` tiene menos puntos de los esperados

`binned.csv` es un producto **pareado**.

Utiliza únicamente timestamps donde:

- CONTROL es analíticamente válido;
- SAMPLE es analíticamente válido.

Por ello puede contener menos información que los ajustes independientes
reportados en `summary.csv`.

Esto no es una contradicción.

---

## 26. `n_bins` del summary no parece coincidir con el CSV pareado

Los fits de `summary.csv` utilizan binning **independiente por gota**.

El archivo `binned.csv`, en cambio, utiliza validez pareada.

Por ello:

- `binned.csv` sirve para comparación simultánea;
- `n_bins` del summary corresponde al análisis independiente del lado.

---

## 27. Diferencia entre OLS y Theil–Sen

BubbleCV conserva:

- OLS frame a frame;
- OLS binned;
- Theil–Sen robusto binned.

Una diferencia entre ellos no debe corregirse automáticamente.

Puede indicar:

- outliers;
- cambio de régimen;
- región final de evaporación;
- perturbación experimental;
- detecciones atípicas.

Inspecciona la serie temporal antes de decidir si existe un problema.

---

## 28. R² bajo

No existe un R² mínimo universal para todos los experimentos.

Un R² bajo puede deberse a:

- comportamiento no lineal;
- cambio de régimen;
- ruido experimental;
- final de evaporación;
- perturbación mecánica;
- detección inestable.

No modifiques parámetros del software únicamente para elevar R².

---

## 29. K negativa

BubbleCV define exactamente:

`K = -slope`

No utiliza:

`abs(slope)`

Si `r_eq²` aumenta con el tiempo, la pendiente puede ser positiva y K
resultará negativa.

Eso requiere revisar la interpretación científica y la ventana temporal.

No debe cambiarse automáticamente el signo.

---

## 30. Saltos en `r_eq²`

Revisa el video alrededor del timestamp correspondiente.

Posibles causas:

- perturbación física;
- movimiento;
- cambio de iluminación;
- proximidad al final de evaporación;
- cambio de detección;
- deformación real de la gota.

Comprueba CONTROL y SAMPLE por separado.

---

## 31. La gota está casi evaporada

Cerca de la desaparición de una gota pueden aumentar:

- fallos de detección;
- residuos geométricos;
- cambios bruscos de forma;
- sensibilidad a ruido.

No debe asumirse que esa región pertenece al mismo régimen físico que el resto
de la corrida.

La elección de una ventana experimental debe justificarse por el protocolo,
no por el deseo de obtener una pendiente determinada.

---

## 32. `run_manifest.json` falta

El manifest sólo se genera cuando se utiliza:

`python run_bubblecv.py`

Una ejecución directa de:

`analyze_video.py`

no crea este archivo automáticamente.

---

## 33. La carpeta de resultados ya existe

El Guided Runner devuelve un error para evitar sobrescrituras accidentales.

Elige una carpeta nueva.

Por defecto se propone un nombre con:

- nombre del video;
- fecha;
- hora.

---

## 34. El SHA-256 tarda algunos segundos

Antes de ejecutar el análisis, el Guided Runner calcula la huella SHA-256 del
video.

En archivos grandes esto puede tardar un poco.

Es normal.

La huella permite verificar posteriormente que dos corridas utilizaron
exactamente los mismos bytes de entrada.

---

## 35. El análisis terminó con `exit_code` distinto de 0

Consulta:

- mensajes inmediatamente anteriores;
- existencia del video;
- permisos;
- calibración;
- parámetros BODY;
- disponibilidad de dependencias.

El `run_manifest.json` conserva el código de salida final cuando la carpeta de
resultados ya fue creada.

---

## 36. macOS: acceso a archivos

Si Terminal no puede acceder a una carpeta, revisa:

**Ajustes del Sistema → Privacidad y Seguridad**

y concede el permiso correspondiente cuando macOS lo solicite.

No es necesario conceder acceso total al disco de forma preventiva.

---

## 37. macOS: suspensión durante análisis largos

Consulta:

→ [`prevent_sleep.md`](./prevent_sleep.md)

---

## 38. Windows: PowerShell bloquea el entorno virtual

Puedes usar Command Prompt y activar mediante:

`venv\Scripts\activate.bat`

Consulta:

→ [`windows.md`](./windows.md)

---

## 39. Antes de repetir una corrida

Si vas a cambiar algún parámetro, registra primero:

- qué parámetro cambiarás;
- por qué;
- qué evidencia experimental justifica el cambio.

No hagas una serie de ajustes iterativos buscando la K o el R² que esperabas.

---

## 40. Recursos

Guía rápida:

→ [`quickstart.md`](./quickstart.md)

Manual de análisis:

→ [`analysis_guide.md`](./analysis_guide.md)

Calibración:

→ [`calibration.md`](./calibration.md)

Conversión de video:

→ [`video_conversion.md`](./video_conversion.md)

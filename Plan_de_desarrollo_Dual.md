# 1. Plan de Desarrollo: Sistema de Análisis de Evaporación Dual (Control vs. Muestra)

## Fase 1: Preprocesamiento y Calibración
*   **Conversión a Escala de Grises:** Extracción del canal con mejor contraste (probablemente el rojo o verde en este caso).
*   **Filtrado de Ruido:** Aplicación de filtros Gaussianos o Mediana para suavizar texturas del fondo.
*   **Calibración Espacial:** Establecer la relación **píxeles a milímetros** ($\text{px/mm}$) usando el diámetro conocido de tu esfera de referencia (4 mm).

## Fase 2: Segmentación y Extracción de Contornos Múltiples
*   **Aislamiento de la Aguja (ROI):** Filtrado morfológico para desconectar las gotas colgantes de los capilares.
*   **Umbralización Adaptativa:** Para separar las gotas del fondo anaranjado.
*   **Detección de Bordes (Doble):** Modificación del algoritmo para detectar y conservar estrictamente los **dos** contornos de mayor área.
*   **Clasificación Espacial:**
    *   Coordenada $X$ menor $\rightarrow$ Gota Izquierda (**Control**).
    *   Coordenada $X$ mayor $\rightarrow$ Gota Derecha (**Muestra**).

## Fase 3: Cálculo de Propiedades Geométricas Pareadas
*   **Ajuste de Elipse Dual:** Ajuste elíptico simultáneo e independiente para la gota control y la muestra.
*   **Cálculos Físicos (Para cada gota):**
    *   **Diámetro Equivalente ($D$):** Basado en el área del contorno.
    *   **Volumen ($V$):** $V = \frac{4}{3} \pi a b^2$ (asumiendo esferoide oblato/prolato).
    *   **Área Superficial y Excentricidad.**

## Fase 4: Procesamiento de Video y Exportación Combinada
*   **Iteración por Cuadros:** Aplicar las fases 2 y 3 a cada frame del video experimental.
*   **Almacenamiento de Datos (CSV Pareado):** Guardar resultados en un único archivo CSV con columnas emparejadas (`control_vol`, `sample_vol`, etc.) para facilitar la comparación termodinámica directa (ej. comparar tasas de evaporación $dV/dt$).

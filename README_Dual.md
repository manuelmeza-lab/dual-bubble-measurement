# BubbleCV Dual — Sistema de Análisis de Evaporación Simultánea (Control vs. Muestra)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python ≥ 3.9](https://img.shields.io/badge/Python-%E2%89%A53.9-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green.svg)](https://opencv.org/)

Herramienta de visión por computadora avanzada, derivada del proyecto `bubble-measurement`, diseñada específicamente para **medir y comparar simultáneamente dos gotas colgantes** (Control vs. Muestra) a partir de videos de microscopio. 

Este sistema aísla capilares, detecta múltiples contornos elípticos, los clasifica espacialmente (Izquierda = Control, Derecha = Muestra) y exporta métricas termodinámicas pareadas para facilitar el análisis comparativo de las tasas de evaporación.

---

## Tabla de Contenidos

1. [Qué hace este programa](#qué-hace-este-programa)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Estructura de Exportación (CSV Dual)](#estructura-de-exportación-csv-dual)
4. [Consideraciones sobre la Detección Dual](#consideraciones-sobre-la-detección-dual)

---

## Qué hace este programa

1. **Preprocesa** cada frame aislando el fondo y los capilares metálicos.
2. **Detecta Múltiples Gotas** aislando los dos contornos principales en el campo de visión.
3. **Clasifica Espacialmente** asignando la gota izquierda como `Control` y la derecha como `Muestra`.
4. **Calcula Propiedades Geométricas** de manera independiente para cada gota (Volumen, Radio, Excentricidad).
5. **Exporta un CSV Pareado** con métricas simultáneas para graficar comparativas de evaporación ($dV/dt$).

---

## Requisitos e instalación

*Mismos requisitos que el repositorio base `bubble-measurement`.*

1. Clonar el nuevo repositorio o copiar la carpeta base.
2. Activar un nuevo entorno virtual.
3. Instalar dependencias: `pip install -r requirements.txt`.

---

## Estructura de Exportación (CSV Dual)

El archivo CSV de resultados ha sido reestructurado. Ahora cada fila temporal contiene datos emparejados:

| Tiempo (s) | control_volume_mm3 | control_evap_rate | sample_volume_mm3 | sample_evap_rate |
|------------|--------------------|-------------------|-------------------|------------------|
| 1.0        | 10.02              | -0.05             | 9.98              | -0.12            |

---

## Consideraciones sobre la Detección Dual

*   **Alineación Horizontal:** Asegúrate de que ambas gotas estén suspendidas de capilares aproximadamente a la misma altura.
*   **Calibración Única:** El sistema utiliza un solo factor de calibración ($	ext{px/mm}$) derivado de un balín de 4 mm, el cual se aplica equitativamente a ambas gotas.

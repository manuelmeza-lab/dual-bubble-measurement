# Reconocimientos / Acknowledgments

## Repositorio de referencia

Este proyecto está basado en y deriva del trabajo original de:

**[Riperedo/bubble-measurement](https://github.com/Riperedo/bubble-measurement)**

> *Measurement of bubble sizes using videomicroscopy.*

El repositorio original implementa la detección de burbujas mediante
visión por computadora con OpenCV. Este proyecto adapta y extiende esa
base para el análisis de **evaporación de gotas** bajo condiciones de
baja iluminación con microscopio DinoLite, incorporando:

- Ajuste elíptico (en lugar de circular) para gotas deformadas
- Mejora de contraste adaptativa (CLAHE) para imágenes oscuras
- Cálculo de propiedades de esferoide (oblato/prolato)
- Análisis de series de tiempo con suavizado temporal
- CLI completa con soporte de calibración automática

---

## Licencia del proyecto original

El repositorio `Riperedo/bubble-measurement` está publicado bajo
licencia MIT. Consulta el archivo [`LICENSE`](./LICENSE) de este
repositorio.

---

*Si usas este código en un trabajo académico, considera citar también
el repositorio original.*

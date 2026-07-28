# Guía de Contribución

¡Gracias por tu interés en contribuir a BubbleCV!

## ¿Qué puedo aportar?

- Correcciones de errores en el código de detección o geometría
- Mejoras en la documentación (claridad, traducción, ejemplos)
- Nuevos métodos de detección o calibración
- Soporte para nuevos formatos de video o cámara
- Casos de prueba automatizados

## Cómo empezar

1. **Haz un fork** del repositorio en tu cuenta de GitHub
2. **Clona tu fork** localmente:
   ```bash
   git clone https://github.com/TU_USUARIO/bubble-measurement.git
   cd bubble-measurement
   ```
3. **Crea una rama** para tu cambio:
   ```bash
   git checkout -b feature/mi-mejora
   ```
4. **Crea el entorno virtual** e instala dependencias:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # macOS/Linux
   # venv\Scripts\activate    # Windows
   pip install -r requirements.txt
   ```
5. **Realiza tus cambios** y verifica que el código funciona
6. **Haz commit** con un mensaje descriptivo:
   ```bash
   git add .
   git commit -m "feat: descripción breve del cambio"
   ```
7. **Abre un Pull Request** hacia este repositorio

## Reglas importantes

### Qué NO subir
- Videos (`.mov`, `.mp4`, `.avi`, `.wmv`, `.mkv`)
- Imágenes experimentales reales
- Resultados en CSV o PNG generados por el análisis
- La carpeta `venv/`
- Carpetas con datos propios como `DinoLite/` o `resultados_*/`

Consulta el `.gitignore` del proyecto para la lista completa.

### Estilo de código
- Python 3.8+ compatible
- Mantén los docstrings existentes; agrega docstrings en funciones nuevas
- Sin dependencias externas fuera de `requirements.txt` (a menos que se justifique)

## Estructura del código

```
bubble_cv/
├── preprocessing.py   # CLAHE, selección de canal, filtrado
├── detection.py       # Hough Circle + ajuste elíptico
├── geometry.py        # Fórmulas de esferoide (volumen, área, excentricidad)
├── calibration.py     # Calibración px/mm
├── visualization.py   # Superposición de elipses y gráficas
└── io_utils.py        # Carga de imágenes/video, exportación CSV
```

## Reporte de errores

Si encuentras un error, abre un [Issue](../../issues) con:
- Sistema operativo y versión de Python
- Comando exacto que ejecutaste
- Mensaje de error completo
- Descripción de lo que esperabas que pasara

## Preguntas

Si eres estudiante y tienes dudas sobre cómo usar el sistema,
abre un Issue con la etiqueta `question`.

# Guía de instalación — macOS

Esta guía cubre la instalación de BubbleCV Dual en macOS.

El entorno de referencia de `v0.2.1` fue validado en macOS ARM64 con
Python 3.9.6.

---

## 1. Requisitos

Necesitas:

- macOS;
- Python ≥ 3.9;
- Git;
- Terminal;
- conexión a Internet durante la instalación.

`ffmpeg` y `ffprobe` son recomendables para inspeccionar y convertir videos.

---

## 2. Instalar Homebrew

Si todavía no tienes Homebrew, abre Terminal y ejecuta:

`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`

Después cierra y vuelve a abrir Terminal si el instalador lo solicita.

---

## 3. Instalar Python, Git y ffmpeg

Ejecuta:

`brew install python git ffmpeg`

Verifica:

`python3 --version`

`git --version`

`ffprobe -version`

---

## 4. Clonar BubbleCV

Ejecuta:

`git clone https://github.com/manuelmeza-lab/dual-bubble-measurement.git`

Después entra a la carpeta:

`cd dual-bubble-measurement`

---

## 5. Crear un entorno virtual

Dentro de la carpeta del proyecto ejecuta:

`python3 -m venv venv`

Actívalo:

`source venv/bin/activate`

Cuando esté activo aparecerá normalmente:

`(venv)`

al inicio de la línea de Terminal.

Cada vez que abras una Terminal nueva debes volver a activar el entorno.

Por ejemplo:

`cd /ruta/a/dual-bubble-measurement`

`source venv/bin/activate`

---

## 6. Actualizar pip

Con el entorno virtual activo:

`python -m pip install --upgrade pip`

---

## 7. Instalar dependencias

### Entorno reproducible de referencia

Para reproducir el entorno exacto utilizado durante la validación de
`v0.2.1`:

`python -m pip install -r requirements-lock.txt`

Este lock fue validado con:

- Python 3.9.6;
- macOS ARM64;
- OpenCV 5.0.0;
- NumPy 2.0.2;
- Pandas 2.3.3;
- Matplotlib 3.9.4.

### Instalación flexible

También existe:

`requirements.txt`

con las dependencias mínimas/flexibles del proyecto.

Para una computadora destinada al análisis reproducible se recomienda utilizar
`requirements-lock.txt` cuando sea compatible con la plataforma.

---

## 8. Comprobar dependencias

Ejecuta:

`python -m pip check`

En una instalación correcta debe aparecer:

`No broken requirements found.`

---

## 9. Ejecutar los tests

Ejecuta:

`python -m unittest discover -s tests -p 'test_*.py'`

La suite debe terminar con:

`OK`

Puede aparecer algún test marcado como `skipped` cuando depende de datos
históricos externos que no se incluyen en el repositorio.

---

## 10. Primera corrida

La forma recomendada de iniciar BubbleCV es:

`python run_bubblecv.py`

Después sigue la guía:

→ [`quickstart.md`](./quickstart.md)

El Guided Runner solicitará paso a paso:

- video;
- calibración;
- FPS;
- skip;
- BODY de CONTROL;
- BODY de SAMPLE;
- parámetros de análisis;
- carpeta de resultados.

Antes de ejecutar mostrará la configuración completa y pedirá confirmación.

---

## 11. Verificar ffprobe

Si BubbleCV no muestra información de codec, resolución o FPS, ejecuta:

`ffprobe -version`

Si no existe:

`brew install ffmpeg`

El Guided Runner puede funcionar sin `ffprobe`, pero disponer de esta
herramienta facilita la revisión del video y de su cadencia.

---

## 12. Problemas comunes

### `python3: command not found`

Comprueba Homebrew:

`brew --prefix`

En Macs con Apple Silicon suele encontrarse en:

`/opt/homebrew`

Si acabas de instalar Homebrew, cierra Terminal y vuelve a abrirla.

---

### `git: command not found`

Instala Git con:

`brew install git`

---

### El video no puede ser leído

Se recomienda utilizar MP4 con codec H.264.

Consulta:

→ [`video_conversion.md`](./video_conversion.md)

---

### macOS impide acceder a la carpeta del video

Revisa:

**Ajustes del Sistema → Privacidad y Seguridad**

y concede a Terminal permiso para acceder a la ubicación correspondiente si
macOS lo solicita.

No es necesario conceder acceso total al disco de forma preventiva si el
sistema no lo requiere.

---

### La computadora se suspende durante una corrida larga

Consulta:

→ [`prevent_sleep.md`](./prevent_sleep.md)

---

## 13. Resultados

El Guided Runner crea por defecto una carpeta dentro de:

`results/`

con un nombre basado en el video y la fecha/hora de ejecución.

Los archivos principales son:

- `results.csv`;
- `binned.csv`;
- `summary.csv`;
- `run_manifest.json`;
- gráficas PNG.

La carpeta `results/` está excluida del repositorio mediante `.gitignore`.

---

## 14. Desactivar el entorno

Cuando termines:

`deactivate`

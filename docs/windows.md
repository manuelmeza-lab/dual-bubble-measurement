# Guía de instalación — Windows

Esta guía cubre la instalación de BubbleCV Dual en Windows 10 y Windows 11.

El entorno exacto contenido en `requirements-lock.txt` de `v0.2.1` fue
validado de forma independiente en macOS ARM64 con Python 3.9.6.

La instalación en Windows está soportada por la especificación general de
dependencias, pero el lock exacto todavía no cuenta con una validación
clean-room independiente en Windows.

---

## 1. Requisitos

Necesitas:

- Windows 10 u 11;
- Python ≥ 3.9;
- PowerShell o Command Prompt;
- conexión a Internet durante la instalación.

Git es recomendable, aunque también puedes descargar el repositorio como ZIP.

`ffmpeg` y `ffprobe` son recomendables para inspeccionar y convertir videos.

---

## 2. Instalar Python

Descarga Python desde:

https://www.python.org/downloads/

Durante la instalación activa la opción:

`Add Python to PATH`

Después abre una nueva ventana de PowerShell y verifica:

`python --version`

Debe mostrar Python 3.9 o posterior.

---

## 3. Obtener BubbleCV

### Opción A — Git

Si tienes Git instalado:

`git clone https://github.com/manuelmeza-lab/dual-bubble-measurement.git`

Después:

`cd dual-bubble-measurement`

### Opción B — ZIP

En GitHub selecciona:

`Code → Download ZIP`

Descomprime el archivo y abre PowerShell dentro de la carpeta del proyecto.

---

## 4. Crear el entorno virtual

Desde la carpeta de BubbleCV ejecuta:

`python -m venv venv`

---

## 5. Activar el entorno

### PowerShell

`venv\Scripts\Activate.ps1`

### Command Prompt

`venv\Scripts\activate.bat`

Cuando esté activo aparecerá normalmente:

`(venv)`

al inicio de la línea de comandos.

Cada vez que abras una terminal nueva tendrás que volver a activar el entorno.

---

## 6. Si PowerShell bloquea la activación

Si PowerShell impide ejecutar `Activate.ps1`, una alternativa sencilla es
utilizar Command Prompt y activar con:

`venv\Scripts\activate.bat`

Si prefieres continuar en PowerShell, consulta la política de ejecución de tu
sistema antes de modificarla.

No es necesario cambiar políticas de seguridad globales para utilizar
BubbleCV.

---

## 7. Actualizar pip

Con el entorno activo:

`python -m pip install --upgrade pip`

---

## 8. Instalar dependencias

### Instalación general recomendada en Windows

Ejecuta:

`python -m pip install -r requirements.txt`

Después comprueba:

`python -m pip check`

### Sobre `requirements-lock.txt`

El archivo:

`requirements-lock.txt`

contiene las versiones exactas utilizadas en el entorno de referencia de
`v0.2.1`.

Ese lock fue reconstruido y validado con:

- Python 3.9.6;
- macOS ARM64;
- OpenCV 5.0.0;
- NumPy 2.0.2;
- Pandas 2.3.3;
- Matplotlib 3.9.4.

Todavía no debe interpretarse como un lock validado independientemente en
Windows.

---

## 9. Verificar la instalación

Primero:

`python -m pip check`

Una instalación sin conflictos debe mostrar:

`No broken requirements found.`

Después ejecuta:

`python -m unittest discover -s tests -p "test_*.py"`

La suite debe terminar con:

`OK`

Puede aparecer algún test marcado como `skipped` cuando depende de datos
históricos externos que no forman parte del repositorio.

---

## 10. Primera corrida

La forma recomendada de iniciar BubbleCV es:

`python run_bubblecv.py`

Después sigue:

→ [`quickstart.md`](./quickstart.md)

El Guided Runner solicitará:

- video;
- calibración;
- FPS;
- skip;
- BODY de CONTROL;
- BODY de SAMPLE;
- parámetros del análisis;
- carpeta de resultados.

Antes de ejecutar mostrará toda la configuración y pedirá confirmación.

---

## 11. Rutas de archivos

Ejemplo:

`C:\Users\Nombre\Videos\experimento01.mp4`

Las rutas con espacios pueden introducirse directamente cuando el Guided
Runner las solicita.

Ejemplo:

`C:\Users\Nombre\Mis Videos\experimento 01.mp4`

El Guided Runner construye internamente el comando como una lista de
argumentos y no utiliza `shell=True`.

---

## 12. ffmpeg y ffprobe

`ffprobe` permite revisar datos como FPS, duración, codec y resolución.

Si ya tienes ffmpeg instalado, verifica:

`ffprobe -version`

Si no está disponible, BubbleCV puede ejecutar el Guided Runner sin ffprobe,
pero no mostrará automáticamente esos metadatos.

Para conversión de videos consulta:

→ [`video_conversion.md`](./video_conversion.md)

---

## 13. Problemas comunes

### `python` no se reconoce como comando

Python probablemente no está en `PATH`.

La solución más sencilla es reinstalar Python y seleccionar:

`Add Python to PATH`

Después cierra y vuelve a abrir PowerShell.

---

### PowerShell no ejecuta `Activate.ps1`

Puedes utilizar Command Prompt:

`venv\Scripts\activate.bat`

Esto evita tener que modificar la política de ejecución de PowerShell.

---

### El video no puede ser leído

Se recomienda utilizar:

- contenedor MP4;
- codec H.264;
- cadencia CFR cuando el protocolo lo requiera.

Consulta:

→ [`video_conversion.md`](./video_conversion.md)

---

### La computadora entra en suspensión

Consulta:

→ [`prevent_sleep.md`](./prevent_sleep.md)

---

## 14. Resultados

El Guided Runner crea por defecto una carpeta dentro de:

`results/`

Los archivos principales son:

- `results.csv`;
- `binned.csv`;
- `summary.csv`;
- `run_manifest.json`;
- gráficas PNG.

La carpeta `results/` está excluida del repositorio mediante `.gitignore`.

---

## 15. Desactivar el entorno

Cuando termines:

`deactivate`

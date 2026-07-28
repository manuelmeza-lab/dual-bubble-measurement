# Guía de Instalación — macOS

Esta guía cubre la instalación completa de BubbleCV en macOS.

---

## Requisitos

- macOS 12 Monterey o posterior (también funciona en versiones anteriores)
- Terminal (viene incluida en macOS: `/Applications/Utilities/Terminal.app`)
- Python ≥ 3.9

---

## 1. Instalar Python

### Opción A — Homebrew (recomendada)

Homebrew es el gestor de paquetes más popular para macOS y facilita
la gestión de Python sin interferir con el Python del sistema.

```bash
# Instalar Homebrew si no está instalado
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Instalar Python
brew install python

# Verificar
python3 --version
```

> **Nota para Apple Silicon (M1/M2/M3):** Si usas un Mac con chip Apple
> Silicon, Homebrew se instala en `/opt/homebrew/`. Asegúrate de que
> `/opt/homebrew/bin` esté en tu `PATH`. El instalador de Homebrew lo
> configura automáticamente.

### Opción B — Instalador oficial

1. Descarga el instalador desde [python.org/downloads](https://www.python.org/downloads/)
2. Ejecuta el archivo `.pkg`
3. Al terminar, abre Terminal y verifica:
   ```bash
   python3 --version
   ```

---

## 2. Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/bubble-measurement.git
cd bubble-measurement
```

---

## 3. Crear y activar el entorno virtual

```bash
# Crear el entorno virtual
python3 -m venv venv

# Activar
source venv/bin/activate
```

Cuando el entorno está activo, verás `(venv)` al inicio del prompt:
```
(venv) nombre@mac bubble-measurement %
```

> **Importante:** Cada vez que abras una nueva ventana de Terminal,
> debes volver a activar el entorno virtual con `source venv/bin/activate`.

---

## 4. Instalar dependencias

Con el entorno virtual activo:

```bash
pip install -r requirements.txt
```

---

## 5. Verificar la instalación

```bash
python analyze_image.py --help
```

Deberías ver el mensaje de ayuda con todas las opciones disponibles.

---

## Problemas comunes en macOS

### Error: "python3: command not found"

Homebrew no está en el PATH. Agrega esto a tu `~/.zshrc`:
```bash
export PATH="/opt/homebrew/bin:$PATH"   # Apple Silicon
# o
export PATH="/usr/local/bin:$PATH"      # Intel
```
Luego: `source ~/.zshrc`

### Error: "Operation not permitted" al acceder a videos

macOS protege el acceso a ciertos directorios. Solución:
- Ve a **Ajustes del Sistema → Privacidad y Seguridad → Acceso total al disco**
- Agrega Terminal (o el editor que uses)

### Advertencia de "quarantine" al ejecutar scripts descargados

```bash
# Eliminar el atributo de cuarentena (ejecutar una vez en la carpeta del proyecto)
xattr -rd com.apple.quarantine /ruta/al/proyecto/
```

### Error al leer archivos `.mov` (codec no soportado)

```bash
# Instalar ffmpeg para soporte completo de codecs
brew install ffmpeg

# Reinstalar OpenCV con soporte headless (más compatible con codecs)
pip uninstall opencv-python
pip install opencv-python-headless
```

Consulta también → [`docs/video_conversion.md`](./video_conversion.md)

### La computadora se suspende durante análisis largos

Consulta → [`docs/prevent_sleep.md`](./prevent_sleep.md)

---

## Desactivar el entorno virtual

Cuando termines de trabajar:

```bash
deactivate
```

# Guía de Instalación — Windows

Esta guía cubre la instalación completa de BubbleCV en Windows 10 y 11.

---

## Requisitos

- Windows 10 (versión 1903 o posterior) o Windows 11
- PowerShell o Command Prompt (CMD)
- Python ≥ 3.9

---

## 1. Instalar Python

1. Descarga el instalador desde [python.org/downloads](https://www.python.org/downloads/)
   (elige la versión marcada como "Recommended")
2. Ejecuta el archivo `.exe`
3. **Importante:** En la primera pantalla del instalador, marca la casilla:
   ✅ **"Add Python to PATH"**

   Sin esta opción, Python no estará disponible en la terminal.

4. Haz clic en **"Install Now"**

5. Verifica abriendo PowerShell o CMD:
   ```
   python --version
   ```

---

## 2. Clonar el repositorio

Si tienes Git instalado:
```powershell
git clone https://github.com/TU_USUARIO/bubble-measurement.git
cd bubble-measurement
```

Si no tienes Git, descarga el ZIP desde GitHub:
- Haz clic en **Code → Download ZIP**
- Descomprime en la carpeta que prefieras

---

## 3. Crear y activar el entorno virtual

Abre PowerShell o CMD y navega a la carpeta del proyecto:

```powershell
cd C:\ruta\al\proyecto\bubble-measurement
```

Crea el entorno virtual:
```powershell
python -m venv venv
```

**Activa el entorno virtual:**

| Terminal | Comando |
|----------|---------|
| PowerShell | `venv\Scripts\Activate.ps1` |
| CMD | `venv\Scripts\activate.bat` |

Cuando el entorno está activo, verás `(venv)` al inicio del prompt:
```
(venv) PS C:\ruta\al\proyecto>
```

### Error de ejecución en PowerShell

Si PowerShell bloquea la ejecución del script, ejecuta **una sola vez**:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
Luego activa el entorno normalmente.

---

## 4. Instalar dependencias

Con el entorno virtual activo:

```powershell
pip install -r requirements.txt
```

---

## 5. Verificar la instalación

```powershell
python analyze_image.py --help
```

Deberías ver el mensaje de ayuda con todas las opciones disponibles.

---

## Rutas en Windows: diferencia importante

En Windows, las rutas usan `\` en lugar de `/`. Sin embargo, los scripts
de BubbleCV también aceptan `/` en la mayoría de los casos.

**Ejemplos de rutas válidas en Windows:**
```powershell
# Con backslash (estilo Windows)
python analyze_video.py --input path\to\video.mp4 --calibration PX_PER_MM

# Con slash (funciona en Python)
python analyze_video.py --input path/to/video.mp4 --calibration PX_PER_MM
```

Si la ruta contiene espacios, enciérrala en comillas:
```powershell
python analyze_video.py --input "C:\Mi Carpeta\video.mp4" --calibration PX_PER_MM
```

---

## Problemas comunes en Windows

### "python" no se reconoce como comando

Python no está en el PATH. Soluciones:

1. Reinstala Python marcando "Add Python to PATH"
2. O agrega Python al PATH manualmente:
   - Busca "Variables de entorno" en el menú de inicio
   - En "Variables del sistema" → `Path` → Agregar la carpeta de Python
     (generalmente `C:\Users\TU_USUARIO\AppData\Local\Programs\Python\Python3X\`)

### Error al leer archivos `.mov`

Los archivos `.mov` de iPhone o cámaras pueden tener codecs (H.265/HEVC)
que OpenCV no soporta directamente en Windows.

Solución: convertir el video a `.mp4` antes de analizarlo.
Consulta → [`docs/video_conversion.md`](./video_conversion.md)

### La computadora se suspende durante análisis largos

Consulta → [`docs/prevent_sleep.md`](./prevent_sleep.md)

---

## Desactivar el entorno virtual

```powershell
deactivate
```

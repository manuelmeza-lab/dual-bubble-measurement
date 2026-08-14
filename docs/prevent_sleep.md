# Prevención de Suspensión del Sistema

El análisis de videos largos puede tardar varios minutos o incluso horas.
Si el sistema operativo suspende la computadora durante ese tiempo, el
proceso se interrumpirá y perderás el progreso.

Esta guía explica cómo evitarlo en macOS y Windows.

---

## macOS

### Opción A — `caffeinate` (línea de comandos, recomendada)

`caffeinate` es una utilidad incluida en macOS que impide la suspensión
mientras un comando se está ejecutando.

**Úsalo directamente con tu análisis:**
```bash
caffeinate -i python analyze_video.py \
    --input path/to/video.mp4 \
    --calibration PX_PER_MM \
    --fps 30
```

| Bandera | Efecto |
|---------|--------|
| `-i` | Evita suspensión por inactividad |
| `-d` | Evita que la pantalla se apague |
| `-s` | Evita suspensión incluso en batería (con cuidado) |

**Para mantener la computadora despierta por un tiempo específico:**
```bash
# Mantener despierta por 2 horas (7200 segundos)
caffeinate -t 7200
```

**Para mantener despierta hasta que pulses Ctrl+C:**
```bash
caffeinate
```

### Opción B — Ajustes del Sistema (interfaz gráfica)

1. Abre **Ajustes del Sistema** (ícono del engranaje)
2. Selecciona **Batería** (o **Economizador de energía** en versiones antiguas)
3. En **"Evitar suspensión automática"**, aumenta el tiempo a 3-4 horas
4. Opcional: marca **"Evitar que el disco duro se suspenda"**

> Recuerda restablecer los ajustes normales después de terminar el análisis
> para no afectar la duración de la batería.

### Opción C — Ajustes rápidos desde el ícono de batería

En macOS Ventura o posterior:
- Haz clic en el ícono de batería en la barra de menús
- Selecciona **"Modo de bajo consumo"** → desactivar
- O usa **"Preferencias de energía"** para ajustar tiempos de suspensión

---

## Windows

### Opción A — Ajustes de energía (interfaz gráfica)

1. Busca **"Opciones de energía"** en el menú de inicio
2. Selecciona **"Cambiar la configuración del plan"**
3. En **"Apagar pantalla"** y **"Suspender equipo"**, selecciona **"Nunca"**
4. Haz clic en **"Guardar cambios"**

> Recuerda restablecer después de terminar.

### Opción B — `powercfg` (PowerShell, línea de comandos)

```powershell
# Evitar suspensión indefinidamente
powercfg /change standby-timeout-ac 0
powercfg /change standby-timeout-dc 0

# Restaurar (60 minutos = valor típico)
powercfg /change standby-timeout-ac 60
powercfg /change standby-timeout-dc 30
```

### Opción C — Caffeine (aplicación de terceros gratuita)

[Caffeine para Windows](https://www.zhornsoftware.co.uk/caffeine/) es una
aplicación mínima que simula pulsaciones de tecla cada 59 segundos para
mantener el sistema activo.

1. Descarga e instala Caffeine
2. Actívala antes de iniciar el análisis
3. Desactívala al terminar

---

## Ejecutar el análisis en segundo plano (avanzado)

Si cierras la terminal accidentalmente, el análisis se interrumpe.
Para evitarlo, puedes usar `nohup` en macOS/Linux:

```bash
# El proceso continúa aunque cierres la terminal
nohup caffeinate -i python analyze_video.py \
    --input path/to/video.mp4 \
    --calibration PX_PER_MM \
    --fps 30 \
    > analysis_log.txt 2>&1 &

# Ver el progreso en tiempo real
tail -f analysis_log.txt
```

---

## ¿Cuánto tarda el análisis?

El tiempo depende del número de frames procesados:

| Video | FPS | Duración | Frames analizados (`--skip 30`) | Tiempo estimado |
|-------|-----|----------|--------------------------------|-----------------|
| 1 min | 30 | 60 s | ~60 frames | < 1 min |
| 10 min | 30 | 600 s | ~600 frames | 2-5 min |
| 1 hora | 30 | 3600 s | ~3600 frames | 15-40 min |

Los tiempos varían según la resolución del video y el hardware de la computadora.

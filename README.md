# Audio Separator

App local para Mac: separar voz e instrumental, bajar audio de YouTube, convertir voz con RVC en disco y volver a unir.

No usa el Space de Hugging Face.

## Instalar en Mac

Para Apple Silicon, bajá el disco (trae Python y las librerías; no hace falta instalar nada más):

https://github.com/lucastomasi/audio-separator/releases/download/macos-arm64/Audio-Separator-arm64.dmg

Abrilo. La app está a la izquierda y Aplicaciones a la derecha. Arrastrá **Audio Separator** a **Aplicaciones**, cerrá el disco y no la abras desde adentro. `LEEME.txt` está en el mismo disco.

En Aplicaciones, **Control-clic** (clic derecho) sobre Audio Separator → **Abrir** → **Abrir**.

macOS puede decir que no puede verificar al desarrollador. La app no está firmada. Ese Control-clic → Abrir es el paso normal.

Si igual no abre, en Terminal:

```bash
xattr -dr com.apple.quarantine "/Applications/Audio Separator.app"
```

Volvé a abrirla con Control-clic → Abrir. La primera ventana puede tardar uno o dos minutos. Si falla, el detalle queda en `~/Library/Logs/Audio Separator/launch.log`.

### Modelos y procesador

La primera vez que separás, la app baja el modelo de separación. La primera vez que convertís, baja HuBERT y el estimador de pitch. Esos archivos no van en el disco: la app los baja sola.

Tu voz es un archivo `.pth` tuyo. En la app: **Abrí la carpeta de voces**, dejá el `.pth` ahí y tocá **Actualizá los modelos**.

No hace falta una GPU NVIDIA. La separación ONNX corre en CPU. La conversión de voz puede usar el chip de Apple si PyTorch lo detecta; si no, usa CPU y tarda más.

YouTube usa el ffmpeg que viene dentro del `.app`. Si una descarga falla, instalá ffmpeg en la Mac (`brew install ffmpeg`) y volvé a abrir.

### Armar el .app y el DMG

En una Mac, desde el repo:

```bash
bash macos/build_release.sh
```

Eso deja `dist/Audio Separator.app` y `dist/Audio-Separator-arm64.dmg`. El script embebe un CPython 3.12, compila `macos/launcher.c` como ejecutable Mach-O y arma el disco con la app a la izquierda y Aplicaciones a la derecha.

El ejecutable no es un script. `clang` lo genera en la Mac del armado y le pone una firma ad-hoc, sin certificado de desarrollador y sin notarización. Gatekeeper igual pide Control-clic → Abrir.

En una Mac Intel, el mismo comando genera el disco para esa máquina. El enlace de arriba es solo Apple Silicon.

`bash macos/build_app.sh --layout-only` crea la carpeta del `.app` sin Python y sin el binario.

## Desarrollo

```bash
.venv/bin/python desktop.py
```

## Modelos RVC

La app baja HuBERT y `rmvpe.pt` la primera vez que convertís. El `.pth` de la voz es tuyo: dejalo en `rvc_models/`. Sirve el de la carpeta `weights`. Si solo tenés el `G_` del entrenamiento, dejá el `config.json` de ese entrenamiento al lado.

Esos binarios quedan fuera de git. El pitch es rmvpe; no hace falta pyworld.

`infer-rvc-python` pide `pyworld==0.3.4`, que no tiene wheel para Python 3.12. Si `pip install -r requirements-macos.txt` se cae ahí, instalá ese paquete con `--no-deps` después del resto. La app reemplaza pyworld por un stub y no lo llama.
